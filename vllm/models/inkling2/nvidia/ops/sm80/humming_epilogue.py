"""Put the sm_80 fused SiTU epilogue header ahead of Humming's own headers.

Ported from snailbrainx/inkling-2-mini-sm80 (Apache-2.0, commit 8b998e0a2f, site/sitecustomize.py): the modified
humming/epilogue/gmem_writer.cuh adds the Inkling2 activation (softcapped SwiGLU) and routing weight to the first
expert GEMM's epilogue. It is a superset of Humming's header, so other kernels compile as before.
INKLING_SM80_NO_FUSED_SITU=1 leaves Humming's include path unchanged.
"""
import glob
import json
import os

_INSTALLED = False


def install() -> bool:
    global _INSTALLED
    if _INSTALLED:
        return True
    if os.environ.get("INKLING_SM80_NO_FUSED_SITU") == "1":
        return False
    patch_inc = os.path.join(os.path.dirname(os.path.abspath(__file__)), "humming_include")
    try:
        from humming.jit import compiler as module
    except Exception:
        return False
    comp = module.Compiler
    orig_inc = comp.include_dirs
    orig_mt = comp.cuh_last_update_time

    def _patched_inc():
        return [patch_inc] + orig_inc()

    def _patched_mt():
        data = json.loads(orig_mt())
        for fn in sorted(glob.glob(f"{patch_inc}/**/*.cuh", recursive=True)):
            data[fn] = os.stat(fn).st_mtime
        return json.dumps(data, ensure_ascii=False)

    comp.include_dirs = staticmethod(_patched_inc)
    comp.cuh_last_update_time = staticmethod(_patched_mt)
    _INSTALLED = True
    return True
