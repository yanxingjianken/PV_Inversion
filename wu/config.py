"""wu/config.py — Wu (Fortran SOR) pipeline config.

Re-exports the shared root-level config.py so every Wu step script can
simply do `from config import ...`. All Wu-specific values
(OMEGS, OMEGH, PART, THRSH, MAX_ITER, MAXT, INLIN, IBC, IQD, TSCAL, QSCAL)
are already defined in the root config; nothing extra needs to live here.
"""
import importlib.util
import sys
from pathlib import Path

_ROOT_CFG = Path(__file__).resolve().parent.parent / "config.py"
_spec = importlib.util.spec_from_file_location("_root_config", _ROOT_CFG)
_root_config = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_root_config)

# Re-export every public name from the root config
for _k, _v in vars(_root_config).items():
    if not _k.startswith("_"):
        globals()[_k] = _v
del _k, _v, _spec, _root_config

# Make project root importable for any submodule that does
# `from <some_module> import ...` where <some_module> lives at root
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.append(str(_ROOT))
