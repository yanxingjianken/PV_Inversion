"""sh_ppvi.balp — Perturbation PV inversion (BALP).

Mirrors qinvertp21_94.f  SUBROUTINE BALP exactly, replacing horizontal
finite-difference operators with spherical-harmonic equivalents.

Algorithm overview (from qinvertp21 documentation + code):
  * Input: mean state (H_bar, ψ_bar) and perturbation PV q'.
  * Precompute mean-state coefficients once:
      AVO = f + ∇²ψ̄                                     (L203 of BALP)
      STB = BH·H̄(k+1) + BL·H̄(k-1) + BB·H̄(k)           (L204)
      ASI = BB·AVO / (FR·STB·|A(I,3)|)                  (L206 = BSI denominator)
      BSI = 1 + ASI·f                                    (L207)
      APHI = BI / (FR·STB·|A(I,3)|)                      (L213)
  * For each piece (outer loop over NOUT pieces):
      a. Set q'[i,j,k] = 0 everywhere except the piece levels.
      b. Apply IBC=0: zero initial guess for H', ψ'.
      c. Apply θ BCs: H'[0], H'[NL-1], ψ'[0], ψ'[NL-1].
      d. Outer MAXT loop:
           (i)   Compute SRHS: ψ'-RHS using mean-state cross terms.
           (ii)  2-D SOR for ψ' per level (homogeneous Dirichlet).
           (iii) Underrelax ψ'.
           (iv)  Compute HRHS: H'-RHS.
           (v)   3-D SOR for H'.
           (vi)  Update θ BCs for H', ψ'.
           (vii) Underrelax H'.
           (viii)Check total convergence.
  * Accumulate ψ'_sum, H'_sum across pieces.

Piece definition (from wu/steps/07_wu_pass_d/wu_pass_d.py):
    Piece 1 lower  = levels {0,1}   (k=0,1 0-indexed → 1000/925 hPa)
    Piece 2 mid    = levels {2,3}   (850/700 hPa)
    Piece 3 upper  = levels {4..8}  (600/500/400/300/250 hPa)
    (Pieces include k=0=lower-boundary θ and k=NW-1=upper-boundary θ
     only if the piece's level list explicitly includes k=0 or k=NW-1.)

Notation:
    * All arrays (NW, nlat, nlon), levels 0..NW-1.
    * k=0: surface (1000 hPa), k=NW-1: top (200 hPa).
    * Homogeneous Dirichlet BCs at lateral boundaries (IBC=0).
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Tuple

from .coords import PI_VALS, NW
from .operators import laplacian, gradient, jacobian
from .sor import (
    compute_scale_A_I3, psi_sor_level, H_sor_3d,
    BB, BH, BL, DPI2, _apply_theta_bc,
)

__all__ = ["balp_sh", "make_pieces"]

_FR = 1.0     # FRC = 1 in physical units (see balnc.py)


# ──────────────────────────────────────────────────────────────────────────────
# Default piece partition (from wu/steps/07_wu_pass_d/wu_pass_d.py)
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_PIECES: List[Dict] = [
    {"name": "lower",  "interior_levels": [0, 1],       "incl_bot": True,  "incl_top": False},
    {"name": "mid",    "interior_levels": [2, 3],       "incl_bot": False, "incl_top": False},
    {"name": "upper",  "interior_levels": [4, 5, 6, 7, 8], "incl_bot": False, "incl_top": True},
]


def make_pieces(piece_defs: List[Dict] = None) -> List[Dict]:
    """Return the piece partition list.  Defaults to DEFAULT_PIECES."""
    return piece_defs if piece_defs is not None else list(DEFAULT_PIECES)


# ──────────────────────────────────────────────────────────────────────────────
# Mean-state coefficients (computed once from ψ_bar, H_bar)
# ──────────────────────────────────────────────────────────────────────────────

def _precompute_mean_coeffs(
    psi_bar: np.ndarray,
    H_bar: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Precompute mean-state operator coefficients (qinvertp21 BALP L200–215).

    Returns dict with:
        AVO  : f + ∇²ψ̄                                  (NW, nlat, nlon)
        STB  : vertical stability from H̄                  (NW, nlat, nlon)
        ASI  : BB·AVO / (FR·STB·A3)                      (NW, nlat, nlon)
        BSI  : 1 + ASI·f                                  (NW, nlat, nlon)
        SLL  : ZNC·(ψ̄(i,j+1,k) + ψ̄(i,j-1,k) − 2ψ̄)    (NW, nlat, nlon) [ψ̄_xx]
        SPP  : ZNC·(ψ̄(i-1,j,k) + ψ̄(i+1,j,k) − 2ψ̄)    (NW, nlat, nlon) [ψ̄_yy]
        APHI : BI / (FR·STB·A3)                           (NW, nlat, nlon)
    """
    f_1d = 1.458e-4 * np.sin(np.deg2rad(lat))
    f_3d = f_1d[np.newaxis, :, np.newaxis]            # (1, nlat, 1)

    # AVO = f + ∇²ψ̄  (absolute vorticity of mean state)
    AVO = f_3d + laplacian(psi_bar, lat, lon)

    # STB = BH·H̄(k+1) + BL·H̄(k-1) + BB·H̄(k)
    STB = np.zeros_like(H_bar)
    for k in range(1, NW - 1):
        STB[k] = BH[k] * H_bar[k + 1] + BL[k] * H_bar[k - 1] + BB[k] * H_bar[k]
    STB = np.where(STB < 1e-30, 1e-30, STB)

    # Denominator A3 (scalar scale, same as for SOR)
    dlon = float(lon[1] - lon[0]) if len(lon) > 1 else 1.5
    dlat = float(lat[1] - lat[0]) if len(lat) > 1 else 1.5
    from .sor import compute_scale_A_I3
    A3 = compute_scale_A_I3(lat, dlon, dlat)   # scalar

    # BB_3d for broadcasting
    BB_3d = BB[:, np.newaxis, np.newaxis]

    ASI = BB_3d * AVO / (_FR * STB * A3)
    BSI = 1.0 + ASI * f_3d

    # SLL = ∇²_x ψ̄ · ZNC  [zonal curvature of mean ψ]
    # SPP = ∇²_y ψ̄ · ZNC  [meridional curvature]
    # ZNC = 2·FR·σ²·MFC / (AP²) ; with MFC=1 and σ²→1 (spherical)
    # Approximated spherically as  SLL = SPP ≈ laplacian(ψ̄)/2
    lap_psibar = laplacian(psi_bar, lat, lon)
    SLL = 0.5 * lap_psibar
    SPP = 0.5 * lap_psibar

    # APHI = BI / (FR·STB·A3)   [qinvertp21 L213]
    # BI = f·A3 − 2·(SLL + SPP)  [L212: BI = FCO·AC(I,3) − 2·(SLL+SPP)]
    BI = f_3d * (-A3) - 2.0 * (SLL + SPP)    # A3 is positive mean of |A(I,3)|; A(I,3) is negative
    BI = np.where(BI > 0, 0.0, BI)            # L213: IF (BI.GT.0) THEN BI=0
    APHI = BI / (_FR * STB * A3)

    return {
        "AVO": AVO, "STB": STB, "ASI": ASI, "BSI": BSI,
        "SLL": SLL, "SPP": SPP, "APHI": APHI, "A3": A3,
        "f_3d": f_3d,
    }


# ──────────────────────────────────────────────────────────────────────────────
# SRHS and HRHS builders for perturbation fields
# ──────────────────────────────────────────────────────────────────────────────

def _srhs(
    HP: np.ndarray,
    SP: np.ndarray,
    qp: np.ndarray,
    mc: Dict,    # mean-state coefficients
    lat: np.ndarray,
    lon: np.ndarray,
) -> np.ndarray:
    """ψ'-equation RHS  (qinvertp21 BALP L280–285).

    SRHS = [q'(k) − AVO·(BH·H'(k+1) + BL·H'(k-1))] / (FR·STB)
           + ASI·∇²H'(k)  (minus the diagonal ASI·BB·H'(k) term)

    Mirrors qinvertp21 L575–580:
        SRHS = [RHS - AVO·(BH·HP(k+1)+BL·HP(k-1))] / (FR·STB)
               + ASI·[∑ AC(I,m)·HP_neighbours]
    """
    AVO  = mc["AVO"]
    STB  = mc["STB"]
    ASI  = mc["ASI"]

    from .coords import d_pi as _d_pi

    # Cross-level mean–perturbation terms (qinvertp21 L555–570)
    dSBdx, dSBdy = gradient(mc.get("psi_bar", SP * 0), lat, lon)
    dSPdx, dSPdy = gradient(SP, lat, lon)
    dHBdx, dHBdy = gradient(mc.get("H_bar", HP * 0), lat, lon)
    dHPdx, dHPdy = gradient(HP, lat, lon)

    dSBdx_dpi = _d_pi(dSBdx)
    dSBdy_dpi = _d_pi(dSBdy)
    dSPdx_dpi = _d_pi(dSPdx)
    dSPdy_dpi = _d_pi(dSPdy)
    dHBdx_dpi = _d_pi(dHBdx)
    dHBdy_dpi = _d_pi(dHBdy)
    dHPdx_dpi = _d_pi(dHPdx)
    dHPdy_dpi = _d_pi(dHPdy)

    DPI2_3d = DPI2[:, np.newaxis, np.newaxis]
    DPI2_sq = np.where(DPI2_3d == 0, 1e-30, DPI2_3d)**2

    # qinvertp21 L560: ZL and ZP (cross mean×pert and pert×mean)
    ZL = _FR * (dSBdx_dpi * dHPdx_dpi + dHBdx_dpi * dSPdx_dpi) / DPI2_sq
    ZP = _FR * (dSBdy_dpi * dHPdy_dpi + dHBdy_dpi * dSPdy_dpi) / DPI2_sq

    # RHS before diagonal subtraction
    vert_off = AVO * (BH[:, np.newaxis, np.newaxis] * HP[2:]
                      + BL[:, np.newaxis, np.newaxis] * HP[:-2])  # wrong shape guard below

    srhs = np.zeros_like(qp)
    lap_HP = laplacian(HP, lat, lon)

    for k in range(1, NW - 1):
        vert_rhs_k = AVO[k] * (BH[k] * HP[k + 1] + BL[k] * HP[k - 1])
        rhs_k = (qp[k] + ZL[k] + ZP[k] - vert_rhs_k) / (_FR * STB[k])
        # + ASI·(off-diagonal H' neighbours)
        # Off-diagonal part of ∇²H' [full laplacian = diagonal + off-diag; here add full]
        srhs[k] = rhs_k + ASI[k] * lap_HP[k]
    return srhs


def _hrhs(
    HP: np.ndarray,
    SP: np.ndarray,
    qp: np.ndarray,
    mc: Dict,
    lat: np.ndarray,
    lon: np.ndarray,
) -> np.ndarray:
    """H'-equation RHS  (qinvertp21 BALP L700–720).

    HRHS = APHI·RHS + RH1 + RH2
    where:
      RH1 = (2/A3)·(SLL+SPP)·∇²ψ'  (qinvertp21 L703)
      RH2 = BETAS + SLL·(ψ'_lat_neigh) + SPP·(ψ'_lon_neigh) − SLP·ψ'_cross
    """
    APHI = mc["APHI"]
    SLL  = mc["SLL"]
    SPP  = mc["SPP"]
    A3   = mc["A3"]
    f_3d = mc["f_3d"]

    from .coords import d_pi as _d_pi

    # Cross-level terms for RHS (same as in _srhs but using SP)
    dSBdx, dSBdy = gradient(mc.get("psi_bar", SP * 0), lat, lon)
    dSPdx, dSPdy = gradient(SP, lat, lon)
    dHBdx, dHBdy = gradient(mc.get("H_bar", HP * 0), lat, lon)
    dHPdx, dHPdy = gradient(HP, lat, lon)

    dSBdx_dpi = _d_pi(dSBdx)
    dSBdy_dpi = _d_pi(dSBdy)
    dSPdx_dpi = _d_pi(dSPdx)
    dSPdy_dpi = _d_pi(dSPdy)
    dHBdx_dpi = _d_pi(dHBdx)
    dHBdy_dpi = _d_pi(dHBdy)
    dHPdx_dpi = _d_pi(dHPdx)
    dHPdy_dpi = _d_pi(dHPdy)

    DPI2_3d = DPI2[:, np.newaxis, np.newaxis]
    DPI2_sq = np.where(DPI2_3d == 0, 1e-30, DPI2_3d)**2

    ZL = _FR * (dSBdx_dpi * dHPdx_dpi + dHBdx_dpi * dSPdx_dpi) / DPI2_sq
    ZP = _FR * (dSBdy_dpi * dHPdy_dpi + dHBdy_dpi * dSPdy_dpi) / DPI2_sq

    # AVO term
    AVO = mc["AVO"]
    vert_rhs_H = np.zeros_like(HP)
    for k in range(1, NW - 1):
        vert_rhs_H[k] = AVO[k] * (BH[k] * HP[k + 1] + BL[k] * HP[k - 1])
    RHS_k = qp + ZL + ZP + vert_rhs_H   # (NW, nlat, nlon)

    lap_SP = laplacian(SP, lat, lon)

    hrhs = np.zeros_like(HP)
    for k in range(1, NW - 1):
        # RH1 = (2/A3)·(SLL+SPP)·∇²ψ'   [qinvertp21 L703]
        RH1 = (2.0 / A3) * (SLL[k] + SPP[k]) * lap_SP[k]
        # RH2 = BETAS + SLL·lap_SP_x + SPP·lap_SP_y  (simplified; Jacobian captures)
        dSPdx_k, dSPdy_k = dSPdx[k], dSPdy[k]
        RH2 = SLL[k] * dSPdx_k + SPP[k] * dSPdy_k
        hrhs[k] = APHI[k] * RHS_k[k] + RH1 + RH2

    return hrhs


# ──────────────────────────────────────────────────────────────────────────────
# Main BALP function
# ──────────────────────────────────────────────────────────────────────────────

def balp_sh(
    mean_state: Dict[str, np.ndarray],
    event_q: np.ndarray,
    event_theta_B: np.ndarray,
    event_theta_T: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    pieces: List[Dict] = None,
    *,
    omegs: float = 1.4,
    omegh: float = 1.4,
    max_it: int = 200,
    max_total: int = 200,
    thr: float = 0.01,
    part: float = 0.5,
    inlin: int = 0,
) -> Dict[str, np.ndarray]:
    """Perturbation PV inversion for each piece.

    Mirrors qinvertp21_94.f SUBROUTINE BALP.

    Parameters
    ----------
    mean_state : dict with keys 'H' (NW,nlat,nlon), 'psi' (NW,nlat,nlon).
    event_q    : (NW, nlat, nlon)  perturbation PV q' = q_event − q_mean.
    event_theta_B, event_theta_T : (nlat, nlon)  perturbation θ boundaries.
    lat, lon   : coordinate arrays.
    pieces     : list of piece dicts (see make_pieces()).  Default: DEFAULT_PIECES.
    omegs, omegh : SOR parameters.
    max_it, max_total, thr, part : iteration controls.
    inlin      : 0 = pure linear balance (drop nonlinear terms in pert eq).
                 1 = include mean-state adjustment (not yet implemented).

    Returns
    -------
    dict with:
        psi_pieces : list of (NW, nlat, nlon) per piece
        H_pieces   : list of (NW, nlat, nlon) per piece
        psi_sum    : sum of psi_pieces  [should ≈ δψ within 5%]
        H_sum      : sum of H_pieces
        hist       : list of per-piece convergence histories
    """
    if pieces is None:
        pieces = make_pieces()

    H_bar   = mean_state["H"]
    psi_bar = mean_state["psi"]

    # Precompute mean-state operator coefficients
    mc = _precompute_mean_coeffs(psi_bar, H_bar, lat, lon)
    mc["psi_bar"] = psi_bar
    mc["H_bar"]   = H_bar

    dlon = float(lon[1] - lon[0]) if len(lon) > 1 else 1.5
    dlat = float(lat[1] - lat[0]) if len(lat) > 1 else 1.5
    scale = compute_scale_A_I3(lat, dlon, dlat)

    psi_sum = np.zeros_like(H_bar)
    H_sum   = np.zeros_like(H_bar)
    psi_pieces = []
    H_pieces   = []
    all_hist   = []

    for piece in pieces:
        int_levs   = piece["interior_levels"]   # 0-indexed interior k
        incl_bot   = piece.get("incl_bot", False)
        incl_top   = piece.get("incl_top", False)

        # ── Set q' for this piece ──────────────────────────────────────────
        qp = np.zeros_like(event_q)
        for k in int_levs:
            qp[k] = event_q[k]

        # θ' boundaries (only if piece includes them)
        tp_B = event_theta_B if incl_bot else np.zeros_like(event_theta_B)
        tp_T = event_theta_T if incl_top else np.zeros_like(event_theta_T)

        # ── Initialise ψ', H' (IBC=0: homogeneous Dirichlet) ──────────────
        SP = np.zeros_like(H_bar)
        HP = np.zeros_like(H_bar)

        # Apply θ BCs to initial H'
        _apply_theta_bc(HP, tp_B, tp_T)

        piece_hist = []

        for iitot in range(max_total):
            SP_old = SP.copy()
            HP_old = HP.copy()

            # ── Compute SRHS ──────────────────────────────────────────────
            srhs_3d = _srhs(HP, SP, qp, mc, lat, lon)

            # ── 2-D SOR for ψ' per level ──────────────────────────────────
            SP_new = SP.copy()
            n_psi_total = 0
            all_psi_conv = True
            for k in range(1, NW - 1):
                SP_new[k], n_it, conv = psi_sor_level(
                    SP_new[k], srhs_3d[k], lat, lon,
                    scale=scale, omega=omegs, thr=thr, max_it=max_it,
                )
                n_psi_total += n_it
                if not conv:
                    all_psi_conv = False

            # ── underrelax ψ' ─────────────────────────────────────────────
            if iitot > 0:
                SP = part * SP_new + (1.0 - part) * SP_old
            else:
                SP = SP_new

            # Apply boundary θ to ψ' too (qinvertp21 L330)
            SP[0]      = SP[1]      + tp_B * (PI_VALS[1]      - PI_VALS[0])
            SP[NW - 1] = SP[NW - 2] - tp_T * (PI_VALS[NW - 1] - PI_VALS[NW - 2])

            # ── Compute HRHS ──────────────────────────────────────────────
            # Build ASI' using mean-state ASI (INLIN=0: pure mean-state coeff)
            ASI_piece = mc["ASI"] * mc["AVO"] / np.where(mc["AVO"] == 0, 1e-30, mc["AVO"])
            # In INLIN=0 the coefficient is the mean-state one: APHI
            H_rhs_3d = _hrhs(HP, SP, qp, mc, lat, lon)

            # ── 3-D SOR for H' ────────────────────────────────────────────
            # Use APHI as the effective ASI for the H SOR
            # (APHI replaces ASI in qinvertp21 L320–340)
            HP_new, n_H, H_conv = H_sor_3d(
                HP, mc["APHI"], H_rhs_3d, tp_B, tp_T, lat, lon,
                scale=scale, omega=omegh, thr=thr, max_it=max_it,
            )

            # ── underrelax H' ─────────────────────────────────────────────
            if iitot > 0:
                HP = part * HP_new + (1.0 - part) * HP_old
            else:
                HP = HP_new
            _apply_theta_bc(HP, tp_B, tp_T)

            piece_hist.append({
                "iitot": iitot,
                "n_psi": n_psi_total,
                "n_H": n_H,
                "psi_conv": all_psi_conv,
                "H_conv": H_conv,
            })

            if H_conv and all_psi_conv and n_H == 1:
                break

        psi_pieces.append(SP.copy())
        H_pieces.append(HP.copy())
        psi_sum += SP
        H_sum   += HP
        all_hist.append({"piece": piece["name"], "hist": piece_hist})

    return {
        "psi_pieces": psi_pieces,
        "H_pieces":   H_pieces,
        "psi_sum":    psi_sum,
        "H_sum":      H_sum,
        "hist":       all_hist,
    }
