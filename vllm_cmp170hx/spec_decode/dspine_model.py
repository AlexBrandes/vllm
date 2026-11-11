# SPDX-License-Identifier: Apache-2.0
"""DSpine drafter (tencent/Qwen3.8-27B-DSpine-b16) for the vllm-cmp170hx engine.

Experimental, off unless a DSpine drafter is configured. DSpine is a DFlash drafter (five Qwen3 layers, an
8,192-token sliding window, block-causal draft attention, target layers
1/16/31/46/61) plus a *code bridge*: after each active layer the residual
stream gets a low-rank write that carries the predecessor position's
predicted code into the next position ("adjacent causal injection").
Tencent's release runs it on SGLang with Hopper-only kernels; this module
ports the bridge onto vLLM's DFlash drafter, plus the greedy serial
last-write refinement (``dspine_tail``, on unless VLLM_CMP_DSPINE_TAIL=0): the
top-16 candidates of each proposal row, rescored with the last layer's write
recomputed from each predecessor candidate's exact code, then walked as a
16-state chain. Tencent's tail precomputes two vocabulary-sized tables
(762 MB BF16) from a dense LM head; here the same terms are formed for the
candidates only, with their LM-head rows rebuilt from the target's int4
Marlin head (``_marlin_a16_rows``), so it costs no resident memory. Not
ported: DSpark-style dynamic verification and the sampled chain (the engine
drafts greedily).

Weights: every Linear int4 g128
(compressed-tensors pack-quantized, as our DFlash2 W4A16 drafter), the
248,320 x 1,024 bridge codebook int8 per row (``codebook_q`` /
``codebook_scale``; only the anchor rows are gathered, so dequantising on
read is free), the small bridge projections BF16. ``architectures:
["DFlashDSpineDraftModel"]`` selects this class (plugin DRAFTER_OVERRIDES); the
speculative method stays ``dflash``.

The bridge works on whole request blocks: the DFlash speculator gives every
request exactly ``1 + num_speculative_tokens`` draft rows (row 0 the anchor),
so the served block is k + 1 (DSpine trained 16; any 2..16 is valid).

Bridge kernels adapted from Tencent/DSpine ``sglang_dspine/kernels/dspine.py``
(Apache-2.0, Copyright (C) 2026 Tencent).
"""
from __future__ import annotations

import os
from collections.abc import Iterable

import numpy as np

import torch
import torch.nn.functional as F
from torch import nn

from vllm.logger import init_logger
from vllm.model_executor.models import qwen3_dflash
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

from vllm_cmp170hx.spec_decode.dflash_draft import (
    DFlashDraftMixin,
    fitted_rope_positions,  # noqa: F401  (kept importable for the mixin)
)

logger = init_logger(__name__)
BRIDGE = "selected_prefix_bridge."


# --------------------------------------------------------------- kernels

@triton.jit
def _prepare_read(X, Residual, Raw, Normalized, RMS, H: tl.constexpr, EPS: tl.constexpr,
                  ADD: tl.constexpr, WIDTH: tl.constexpr):
    row = tl.program_id(0)
    h = tl.arange(0, WIDTH)
    x = tl.load(X + row * H + h, h < H, 0).to(tl.float32)
    if ADD:
        residual = tl.load(Residual + row * H + h, h < H, 0).to(tl.float32)
        x = (x + residual).to(X.dtype.element_ty).to(tl.float32)
    rms = tl.sqrt(tl.maximum(tl.sum(x * x, 0) / H, 1.1754943508222875e-38))
    tl.store(Raw + row * H + h, x, h < H)
    tl.store(Normalized + row * H + h, x / (rms + EPS), h < H)
    tl.store(RMS + row, rms)


@triton.jit
def _normalize_codes(Projected, AnchorCodes, Codes, C: tl.constexpr, T: tl.constexpr,
                     EPS: tl.constexpr, WIDTH: tl.constexpr):
    # AnchorCodes [rows / T, C]: the anchor token's codebook row per block,
    # already dequantised (int8 codebook)
    row = tl.program_id(0)
    k = tl.arange(0, WIDTH)
    if row % T == 0:
        z = tl.load(AnchorCodes + (row // T) * C + k, k < C, 0).to(tl.float32)
        z = z * tl.sqrt(float(C))
    else:
        z = tl.load(Projected + row * C + k, k < C, 0).to(tl.float32)
        norm = tl.sqrt(tl.sum(z * z, 0))
        z = tl.sqrt(float(C)) * z / (norm + EPS)
    tl.store(Codes + row * C + k, z, k < C)


@triton.jit
def _feature_message(Packed, Bias, Message, R: tl.constexpr, T: tl.constexpr,
                     WIDTH: tl.constexpr):
    row = tl.program_id(0)
    k = tl.arange(0, WIDTH)
    query = tl.load(Packed + row * (3 * R) + k, k < R, 0).to(tl.float32)
    prev = tl.maximum(row - 1, 0)
    sender = tl.load(Packed + prev * (3 * R) + R + k, k < R, 0).to(tl.float32)
    value = tl.load(Packed + prev * (3 * R) + 2 * R + k, k < R, 0).to(tl.float32)
    bias = tl.load(Bias + k, k < R, 0).to(tl.float32)
    gate = tl.sigmoid(query + sender + bias)
    message = tl.where(row % T == 0, 0.0, gate * value)
    tl.store(Message + row * R + k, message, k < R)


@triton.jit
def _feature_finish(Raw, Delta, RMS, Weight, Output, Normed, H: tl.constexpr,
                    T: tl.constexpr, NORM_EPS: tl.constexpr, WIDTH: tl.constexpr):
    row = tl.program_id(0)
    k = tl.arange(0, WIDTH)
    raw = tl.load(Raw + row * H + k, k < H, 0).to(tl.float32)
    delta = tl.load(Delta + row * H + k, k < H, 0).to(tl.float32)
    rms = tl.load(RMS + row)
    edge = row % T != 0
    update = rms * delta * tl.cast(edge, tl.float32)
    result = (raw + update).to(Output.dtype.element_ty)
    tl.store(Output + row * H + k, result, k < H)
    x = result.to(tl.float32)
    variance = tl.sum(x * x, 0) / H
    y = (x * tl.rsqrt(variance + NORM_EPS)).to(Output.dtype.element_ty)
    weight = tl.load(Weight + k, k < H, 0)
    y = (y.to(tl.float32) * weight.to(tl.float32)).to(Output.dtype.element_ty)
    tl.store(Normed + row * H + k, y, k < H)


@triton.jit
def _marlin_a16_rows_kernel(Q, S, InvW, InvS, Ids, Out, N, K, BK: tl.constexpr):
    """Rows ``Ids`` of an int4 W4A16 Marlin weight (K x N, uint4b8, group-128
    scales permuted by ``scale_perm``) rebuilt as [len(Ids), K]: output column
    v of the GEMM is vocabulary row v of the LM head."""
    i = tl.program_id(0)
    pk = tl.program_id(1)
    v = tl.load(Ids + i).to(tl.int64)
    k = pk * BK + tl.arange(0, BK)
    kt = k // 16
    kr = k % 16
    c = (v // 16) * 256 + kr * 16 + (v % 16)
    p = (c // 1024) * 1024 + tl.load(InvW + (c % 1024))
    word = tl.load(Q + kt.to(tl.int64) * (N * 2) + p // 8)
    nib = (word >> ((p % 8) * 4)) & 0xF
    g = k // 128
    scol = (v // 64) * 64 + tl.load(InvS + (v % 64))
    sc = tl.load(S + g.to(tl.int64) * N + scol).to(tl.float32)
    w = (nib.to(tl.float32) - 8.0) * sc
    tl.store(Out + i.to(tl.int64) * K + k, w.to(Out.dtype.element_ty))


_A16_PERMS: dict = {}


def _a16_inverse_perms(device):
    hit = _A16_PERMS.get(device)
    if hit is None:
        from vllm.model_executor.layers.quantization.utils.marlin_utils import get_scale_perms
        from vllm.model_executor.layers.quantization.utils.marlin_utils_test import (
            get_weight_perm)
        perm = get_weight_perm(4, False).numpy()
        inv_w = np.empty_like(perm)
        inv_w[perm] = np.arange(perm.size)
        sp = np.array(get_scale_perms()[0])
        inv_s = np.empty_like(sp)
        inv_s[sp] = np.arange(sp.size)
        hit = (torch.tensor(inv_w, dtype=torch.int32, device=device),
               torch.tensor(inv_s, dtype=torch.int32, device=device))
        _A16_PERMS[device] = hit
    return hit


def lm_head_rows(lm_head: nn.Module, ids: torch.Tensor, hidden: int,
                 dtype: torch.dtype) -> torch.Tensor:
    """[len(ids), hidden] rows of the (shared, possibly int4 Marlin) LM head."""
    q = getattr(lm_head, "weight_packed", None)
    sc = getattr(lm_head, "weight_scale", None)
    if q is None or sc is None or q.dtype != torch.int32:
        return lm_head.weight[ids].to(dtype)
    n_cols = sc.shape[1]
    inv_w, inv_s = _a16_inverse_perms(q.device)
    out = torch.empty((ids.numel(), hidden), dtype=dtype, device=q.device)
    if ids.numel():
        _marlin_a16_rows_kernel[(ids.numel(), hidden // 256)](
            q, sc, inv_w, inv_s, ids.reshape(-1), out, n_cols, hidden, BK=256)
    return out


def _bridge_write(hidden: torch.Tensor, residual: torch.Tensor, input_ids: torch.Tensor,
                  read_w: torch.Tensor, codebook_q: torch.Tensor,
                  codebook_scale: torch.Tensor, packed_w: torch.Tensor,
                  gate_b: torch.Tensor, write_w: torch.Tensor, norm_w: torch.Tensor,
                  block: int, eps: float, norm_eps: float) -> torch.Tensor:
    """One code write after a layer. Returns [3, m, H]: (next-norm(out), out,
    raw): out is the updated residual stream, raw the pre-write state
    (hidden + residual) the refinement tail reads for the last layer."""
    m, h = hidden.shape
    c = read_w.shape[0]
    r = gate_b.shape[0]
    out = torch.empty((3, m, h), dtype=hidden.dtype, device=hidden.device)
    if m == 0:
        return out
    raw = out[2]
    normalized = torch.empty_like(hidden)
    rms = torch.empty(m, device=hidden.device, dtype=torch.float32)
    _prepare_read[(m,)](hidden, residual, raw, normalized, rms, h, eps, True,
                        triton.next_power_of_2(h))
    projected = F.linear(normalized, read_w)
    anchors = input_ids.view(-1, block)[:, 0]
    anchor_codes = (codebook_q[anchors].to(torch.float32)
                    * codebook_scale[anchors].to(torch.float32)[:, None])
    codes = torch.empty_like(projected)
    _normalize_codes[(m,)](projected, anchor_codes, codes, c, block, eps,
                           triton.next_power_of_2(c))
    packed = F.linear(codes, packed_w)
    message = torch.empty((m, r), device=hidden.device, dtype=hidden.dtype)
    _feature_message[(m,)](packed, gate_b, message, r, block, triton.next_power_of_2(r))
    delta = F.linear(message, write_w)
    _feature_finish[(m,)](raw, delta, rms, norm_w, out[1], out[0], h, block, norm_eps,
                          triton.next_power_of_2(h))
    return out


def _bridge_write_fake(hidden, residual, input_ids, read_w, codebook_q, codebook_scale,
                       packed_w, gate_b, write_w, norm_w, block, eps, norm_eps):
    return torch.empty((3, *hidden.shape), dtype=hidden.dtype, device=hidden.device)


direct_register_custom_op("cmp_dspine_bridge_write", _bridge_write,
                          fake_impl=_bridge_write_fake)


# ---------------------------------------------------------------- bridge

class DSpineBridge(nn.Module):
    """The code bridge's weights (checkpoint names under ``selected_prefix_bridge.``)."""

    def __init__(self, config, dtype: torch.dtype):
        super().__init__()
        cfg = dict(getattr(config, "dflash_config", None) or {})
        bridge = cfg.get("selected_prefix_code_bridge") or {}
        if not bridge:
            raise ValueError("DSpine draft config lacks dflash_config.selected_prefix_code_bridge")
        if cfg.get("selected_prefix_terminal", True):
            raise ValueError("DSpine port supports terminal-disabled code bridges only")
        if cfg.get("selected_prefix_feature_write_mode") != "independent":
            raise ValueError("DSpine port requires independent learned writes")
        self.hidden = int(config.hidden_size)
        self.code = int(bridge["code_size"])
        self.rank = int(bridge["rank"])
        self.eps = float(bridge["eps"])
        self.num_layers = int(config.num_hidden_layers)
        self.active = frozenset(int(i) - 1 for i in cfg["selected_prefix_feature_layers"])
        if self.num_layers - 1 not in self.active:
            raise ValueError("DSpine port expects the last layer to write")
        h, c, r, n, v = self.hidden, self.code, self.rank, self.num_layers, int(config.vocab_size)

        def p(*shape, dt=dtype):
            return nn.Parameter(torch.empty(*shape, dtype=dt), requires_grad=False)

        self.codebook_q = p(v, c, dt=torch.int8)
        self.codebook_scale = p(v, dt=torch.float16)
        self.read_weight = p(c, h)
        self.early_read_weights = p(n - 1, c, h)
        self.feature_write_weights = p(n, h, r)
        self.gate_weight = p(r, c)
        self.gate_bias = p(r)
        self.predecessor_gate_weight = p(r, c)
        self.prev_weight = p(r, c)
        self.packed_weight = p(3 * r, c)

    def read_w(self, layer: int) -> torch.Tensor:
        return self.read_weight if layer == self.num_layers - 1 else self.early_read_weights[layer]

    def finish_load(self) -> None:
        self.packed_weight.data.copy_(torch.cat(
            (self.gate_weight.data, self.predecessor_gate_weight.data, self.prev_weight.data),
            dim=0))


_BRIDGE_NAMES = {
    "codebook_q": "codebook_q", "codebook_scale": "codebook_scale",
    "read_weight": "read_weight", "early_read_weights": "early_read_weights",
    "feature_write_weights": "feature_write_weights", "gate.weight": "gate_weight",
    "gate.bias": "gate_bias", "predecessor_gate.weight": "predecessor_gate_weight",
    "prev.weight": "prev_weight",
}
# present in the checkpoint, not used by the parallel draft (``out`` must be
# zero for a terminal-disabled bridge; the version tensors are checked)
_BRIDGE_IGNORED = {"out.weight", "independent_write_version", "terminal_disabled_version"}


# ----------------------------------------------------------------- model

class DSpineQwen3Model(qwen3_dflash.DFlashQwen3Model):
    def __init__(self, *, vllm_config, start_layer_id: int = 0, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, start_layer_id=start_layer_id, prefix=prefix)
        spec = vllm_config.speculative_config
        self.block = 1 + int(spec.num_speculative_tokens)
        trained = int(getattr(self.config, "block_size", 16) or 16)
        if not 2 <= self.block <= trained:
            raise ValueError(f"DSpine serves blocks of 2..{trained} rows; got {self.block}")
        self.selected_prefix_bridge = DSpineBridge(self.config, vllm_config.model_config.dtype)
        self.tail_enabled = os.environ.get("VLLM_CMP_DSPINE_TAIL", "1") == "1"
        cfg = dict(getattr(self.config, "dflash_config", None) or {})
        refine = cfg.get("selected_prefix_last_write_refine") or {}
        self.head_topk = int(refine.get("head_topk", 16))

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor,
                input_embeds: torch.Tensor | None = None) -> torch.Tensor:
        if input_embeds is None:
            input_embeds = self.embed_input_ids(input_ids)
        bridge = self.selected_prefix_bridge
        last = len(self.layers) - 1
        hidden = input_embeds
        residual = None
        normed_in = False
        raw_last = None
        for i, layer in enumerate(self.layers):
            if normed_in:
                x = hidden
            elif residual is None:
                residual = hidden
                x = layer.input_layernorm(hidden)
            else:
                x, residual = layer.input_layernorm(hidden, residual)
            x = layer.self_attn(positions=positions, hidden_states=x)
            x, residual = layer.post_attention_layernorm(x, residual)
            hidden = layer.mlp(x)
            normed_in = False
            if i in bridge.active:
                nxt = self.layers[i + 1].input_layernorm if i < last else self.norm
                both = torch.ops.vllm.cmp_dspine_bridge_write(
                    hidden, residual, input_ids, bridge.read_w(i), bridge.codebook_q,
                    bridge.codebook_scale, bridge.packed_weight, bridge.gate_bias,
                    bridge.feature_write_weights[i], nxt.weight, self.block, bridge.eps,
                    nxt.variance_epsilon)
                hidden, residual = both[0], both[1]
                if i == last:
                    raw_last = both[2]
                normed_in = True
        if not normed_in:
            hidden, _ = self.norm(hidden, residual)
        if self.tail_enabled and raw_last is not None:
            # [m, 2H]: the speculator splits it (dflash/speculator.py overlay)
            return torch.cat((hidden, raw_last), dim=-1)
        return hidden


class DSpineQwen3ForCausalLM(DFlashDraftMixin, qwen3_dflash.DFlashQwen3ForCausalLM):
    """DSpine drafter: DFlash layers + code bridge (``DFlashDSpineDraftModel``; vLLM prefixes DFlash-method drafters with DFlash)."""

    model_cls = DSpineQwen3Model

    def compute_logits(self, hidden_states: torch.Tensor):
        h = self.config.hidden_size
        if hidden_states.shape[-1] != h:
            hidden_states = hidden_states[..., :h]
        return super().compute_logits(hidden_states)

    @torch.no_grad()
    def dspine_tail(self, rows: torch.Tensor, raw: torch.Tensor, anchors: torch.Tensor,
                    steps: int) -> torch.Tensor:
        """Greedy serial last-write refinement (Tencent's ``RefinementTail.greedy``)
        for ``rows`` [R*steps, H] (the draft's normed proposal rows) with their
        pre-write states ``raw`` and the requests' anchor tokens ``anchors`` [R].
        Returns the refined draft tokens [R*steps] (int64). Capture-safe."""
        model = self.model
        br = model.selected_prefix_bridge
        n, h = rows.shape
        nreq = n // steps
        k = model.head_topk
        c = br.code
        logits = self.compute_logits(rows)
        cand = torch.topk(logits.float(), k, dim=-1).indices
        cand = torch.sort(cand, dim=-1).values                         # [n, K] ascending ids
        heads = lm_head_rows(self.lm_head, cand, h, torch.float32).view(n, k, h)
        normw = model.norm.weight.float()
        rawf = raw.float()
        rms = torch.sqrt(torch.clamp((rawf * rawf).mean(-1), min=1.1754943508222875e-38))
        normalized = rawf / (rms[:, None] + br.eps)
        proj = normalized @ br.read_weight.float().t()                  # [n, C]
        codes = (c ** 0.5) * proj / (proj.norm(dim=-1, keepdim=True) + br.eps)
        query = codes @ br.gate_weight.float().t()                      # [n, R]
        base = torch.einsum("nh,nkh->nk", rawf * normw, heads)          # [n, K]
        cand_v = cand.view(nreq, steps, k)
        sid = torch.empty_like(cand_v)
        sid[:, 0, :] = anchors.view(nreq, 1).to(sid.dtype)
        if steps > 1:
            sid[:, 1:, :] = cand_v[:, :-1, :]
        sid = sid.view(n, k)
        emb = (c ** 0.5) * (br.codebook_q[sid].float()
                            * br.codebook_scale[sid].float()[..., None])   # [n, K, C]
        tvg = emb @ br.predecessor_gate_weight.float().t()               # [n, K, R]
        tvv = emb @ br.prev_weight.float().t()
        msg = torch.sigmoid(query[:, None, :] + tvg + br.gate_bias.float()) * tvv
        write_w = br.feature_write_weights[br.num_layers - 1].float()    # [H, R]
        tg = (heads * normw) @ write_w                                   # [n, K, R]
        score = base[:, None, :] + rms[:, None, None] * (msg @ tg.transpose(1, 2))
        idx = score.argmax(-1).view(nreq, steps, k)                      # [R, steps, Kprev]
        out = torch.empty((nreq, steps), dtype=torch.int64, device=rows.device)
        j = idx[:, 0, :1]
        for i in range(steps):
            out[:, i] = cand_v[:, i].gather(1, j).squeeze(1)
            if i + 1 < steps:
                j = idx[:, i + 1].gather(1, j)
        return out.view(-1)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        bridge = self.model.selected_prefix_bridge
        loaded: set[str] = set()

        def rest():
            for name, tensor in weights:
                short = name.removeprefix("model.")
                if short.startswith(BRIDGE):
                    key = short[len(BRIDGE):]
                    if key in _BRIDGE_IGNORED:
                        if key == "out.weight" and tensor.count_nonzero().item():
                            raise ValueError("DSpine bridge: nonzero terminal projection")
                        if key.endswith("_version") and int(tensor.item()) != 1:
                            raise ValueError(f"DSpine bridge: unsupported {key}")
                        continue
                    target = _BRIDGE_NAMES.get(key)
                    if target is None:
                        raise ValueError(f"DSpine bridge: unexpected tensor {name}")
                    param = getattr(bridge, target)
                    if tuple(param.shape) != tuple(tensor.shape):
                        raise ValueError(f"DSpine bridge: {name} {tuple(tensor.shape)} "
                                         f"vs {tuple(param.shape)}")
                    param.data.copy_(tensor.to(param.dtype))
                    loaded.add(target)
                    continue
                if short in ("draft_block_causal_version", "last_write_refine_version"):
                    if int(tensor.item()) != 1:
                        raise ValueError(f"DSpine: unsupported {short}")
                    continue
                yield name, tensor

        super().load_weights(rest())
        missing = set(_BRIDGE_NAMES.values()) - loaded
        if missing:
            raise ValueError(f"DSpine bridge tensors missing: {sorted(missing)}")
        bridge.finish_load()
        logger.info("Loaded DSpine draft: block %d, active layers %s, code %d, rank %d, "
                    "int8 codebook %s", self.model.block,
                    sorted(i + 1 for i in bridge.active), bridge.code, bridge.rank,
                    tuple(bridge.codebook_q.shape))
