"""sh/config.py — SH-PPVI (spectral-vertical) pipeline config.

Re-exports the shared root-level config.py so every SH step script can
simply do `from config import ...`. SH-specific parameters
(currently `F0_DEG`) are appended below.

NOTE — SH-PPVI uses two leading-order approximations that Wu does NOT:
  (i)  β-plane thermal-wind link  δΦ ≈ f₀·δψ  (no Charney cross-term)
  (ii) spatial-mean coefficients  A_m, B_m  per pressure level
       (required to diagonalise in spherical-harmonic space)
A future TODO is to upgrade (ii) to spatially-varying A,B; see
sh/sh_ppvi/invert_piece.py docstring for details.
"""
import importlib.util
import sys
from pathlib import Path

_ROOT_CFG = Path(__file__).resolve().parent.parent / "config.py"
_spec = importlib.util.spec_from_file_location("_root_config", _ROOT_CFG)
_root_config = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_root_config)

# Re-export every public name from root config
for _k, _v in vars(_root_config).items():
    if not _k.startswith("_"):
        globals()[_k] = _v
del _k, _v, _spec, _root_config

# Make project root importable
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.append(str(_ROOT))

# ── SH-PPVI-specific ────────────────────────────────────────────────
F0_DEG = 45.0   # reference latitude for f₀ in the β-plane approximation
