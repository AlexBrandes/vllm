# Development: the plugin and the vLLM branch

A vLLM plugin and a lightly modified vLLM for the NVIDIA CMP 170HX. Most of the
fork's code lives outside vLLM; the work comes as two parts:

- **The `vllm-cmp170hx` plugin** (this branch, Python package `vllm_cmp170hx`).
  vLLM loads it through its `vllm.general_plugins` entry point `cmp170hx` at
  startup. If the plugin is missing, the engine stops instead of quietly running
  upstream's path (`vllm_cmp170hx/guard.py`).
- **A vLLM branch** on an exact upstream commit: the branch
  `vllm/<first 10 characters of the upstream commit>`. It is upstream vLLM plus
  a short list of the fork's commits, one per feature. Most are hooks of a few
  lines that call into the plugin; a few are upstream bug fixes carried until
  upstream has them.

An engine is three pins put together: an upstream vLLM commit, a vLLM branch
commit and a plugin commit.

The layout follows Morrowmake's
[vllm-sm80](https://github.com/Morrowmake/vllm-cmp170hx), so changes move
between the two forks with a package rename and, for vLLM changes, a split
by file (see
[Contributing to vllm-sm80](#contributing-to-vllm-sm80)).

## Branches

| Branch | Holds |
|---|---|
| `plugin` (default) | the plugin: `vllm_cmp170hx/`, its tests, its tools, `FORK.md` (a generated map of every file and every change to upstream vLLM), and the fork's landing page and recipes |
| `vllm/a0d7356b98`, … | upstream vLLM at that commit plus the fork's commits. A newer upstream gets a new `vllm/` branch; older ones stay as long as a recipe pins them. |
| `main` | a plain mirror of upstream vLLM |

The current line is upstream `a0d7356b98` + `vllm/a0d7356b98` + `plugin`.

## Rules

- **Fork code goes in the plugin.** An upstream file is touched only where no
  extension point serves (plugin registration, a model-registry override, a
  scheduler or worker class setting), and then with a hook of a few lines that
  calls into `vllm_cmp170hx`.
- **One commit per feature on the vLLM branch.** Its message says what the
  change is for. A later change to the same feature amends that commit (send it
  as a `fixup!` commit so it autosquashes); a new feature gets a new commit.
- **Rebase, never merge.** The vLLM branch is always its upstream commit plus
  the fork's commits, so `git log <upstream>..vllm/<upstream10>` lists every
  change the fork makes to vLLM.
- **Every modified upstream source file says so.** Its first lines carry
  `# Modified by Alex Brandes for CMP 170HX support; see repository history.`
  (Apache-2.0 section 4(b); checked by
  `tests/cmp170hx/test_modification_notices.py`).
- **Every upstream object the plugin relies on without editing it is pinned.**
  `tests/cmp170hx/seams.yaml` records a fingerprint of each one (source hash and
  signature, or type and value), with a `why`. `test_seams.py` fails when
  upstream moves one.
- **Upstream fixes go upstream.** A carried fix names its upstream PR in its
  commit message and is dropped once the upstream base includes it.

## Moving to a newer upstream commit

1. Branch `vllm/<new10>` from the new upstream commit and cherry-pick the fork's
   commits onto it, oldest first. Conflicts are confined to the hooks. Drop the
   carried fixes the new base already has.
2. Assemble the engine (below) and run `tests/cmp170hx/test_seams.py`. Each
   failing row names the upstream object that moved and what the plugin depends
   on in it. Check the change against the row's `why`, adapt the plugin if
   needed, then pin the new base:

   ```bash
   python tests/cmp170hx/update_seams.py --new-base
   ```

   This writes `seams.<new12>.yaml`, the reviewed pins for that base;
   `seams.yaml` keeps the pins of the base named in `FORK.md`.
3. Run the plugin's tests, then the GPU validation, then regenerate `FORK.md`
   once the new line becomes the current one:

   ```bash
   python tools/cmp170hx/fork_map.py --vllm vllm/<new10>
   ```

## Assembling and testing an engine

```bash
python tools/cmp170hx/assemble.py /path/to/asm \
    --vllm vllm/a0d7356b98 --plugin plugin \
    --natives-from /path/to/site-packages/vllm   # optional
cd /path/to/asm/plugin
PYTHONPATH=/path/to/asm/engine:/path/to/asm/plugin python -m pytest tests/cmp170hx
python tools/cmp170hx/assemble.py /path/to/asm --remove
```

The engine and plugin directories are detached git worktrees of this clone.
`--natives-from` copies the compiled extensions of a vLLM build of the same
upstream commit, which a source tree does not have.

For an installed engine, build vLLM from the vLLM branch and install the
plugin beside it with `pip install --no-deps .` from this branch.

## Code shared with vllm-sm80

The plugin's sm_80 code for GLM-5.3-Flash (model, layer and op overrides, the
fork CUDA platform and scheduler, the DFlash drafters, the attention, indexer,
KDA, mHC, MoE and Marlin kernels, KV planning and the PP handler) is
vllm-sm80's, taken from its `plugin-dev` at `b2c9487dd5` and renamed to
`vllm_cmp170hx`; its queue hooks for those parts are commits on the vLLM
branch. Only what GLM-5.3 reaches is carried: DeepSeek-V4.1, Qwen4Exp and
vllm-sm80's launcher are not, and GLM-5.3 is the only architecture the plugin
replaces. EXL3 (`vllm_cmp170hx/kernels/exl3`, `csrc/exl3`) is this fork's own
line, not vllm-sm80's later EXL3.

To bring in later vllm-sm80 changes, diff their `plugin-dev` between
`b2c9487dd5` and the new commit, keep the files this plugin carries, rename
`vllm_sm80` to `vllm_cmp170hx` (and `tests/sm80` to `tests/cmp170hx`), and apply
the result with a three-way merge. Queue changes for upstream files go to the
matching feature commit on the vLLM branch.

## Contributing to vllm-sm80

Morrowmake's vllm-sm80 uses the same layout under `vllm_sm80/`. Their changes
to vLLM live on `queue/<upstream10>` branches with one commit per upstream file,
each named `queue: <path>`. To send a change there:

- **Plugin code:** move it from `vllm_cmp170hx/<path>` to `vllm_sm80/<path>`
  and target their `plugin-dev` branch. Seam rows move to their
  `tests/sm80/seams.<base12>.yaml`.
- **vLLM changes:** split the feature commit by file. A file their queue
  already touches becomes a `fixup! queue: <path>` of their commit; a new file
  becomes a `queue: <path>` commit. Where both forks hook the same upstream
  file, the hook functions have the same names and signatures, so only the
  import changes to `vllm_sm80`. Target their newest `queue/<upstream10>` branch.
- **Order:** their queue PR depends on the plugin PR that adds the functions it
  calls; say so in both, and get the plugin PR merged first.

## License

Apache-2.0, as upstream vLLM. See [NOTICE](../NOTICE) for attribution.
