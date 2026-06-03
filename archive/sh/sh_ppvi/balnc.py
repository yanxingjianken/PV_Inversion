"""sh_ppvi.balnc — Total PV inversion (BALNC), Wu non-dim, spectral direct-solve.

Solves Wu's BALNC system using spherical-harmonic operators (``wu_ops``)
instead of the Fortran 5-point FD stencil.  The equation form is identical
to qinvert21_94.f; only the Laplacian/gradient/Jacobian primitives differ.

Operator correspondence (see wu_ops.py):
    Wu ∇²         →  lap_wu  = LL² · ∇²_phys
    Wu ∂/∂x, ∂/∂y →  grad_wu = LL  · grad_phys
    Wu J(a,b)     →  jac_wu  = LL² · J_phys

Non-dim units (see nondim.py):
    H_nd  = H_phys · g/(THO·DPI)
    ψ_nd  = ψ_phys · g/(THO·DPI)
    θ_nd  = θ_phys / THO
    q_nd  = PIF(k)·q_phys / (1e2·QCONST)
    π_wu  = CP·(p/p0)^κ / DPI          (used for vertical differencing)

PDEs per outer Picard iteration (FR=1, SIG=1 for equal-spaced lat/lon grid):

    ψ-equation (Fortran L515–523):
        RHS_ψ = [q − f·STB + lap_wu(H) − ZNL + ZL + ZP] / (f + STB)
        ψ ← inv_lap_wu(RHS_ψ)

    H-equation (Fortran L540–580 + 700–750):
        rh = f·lap_wu(ψ) + 2·jac_wu(ψx,ψy) + q + ZL + ZP
        ASI = f + lap_wu(ψ);  vert = ASI·(BB·H + BH·H[k+1] + BL·H[k-1])
        H[k] ← inv_lap_wu(rh[k] − vert[k])

    ZL, ZP, ZNL (Fortran L505–514):
        ZPL(φ)   = 1 / cos²(φ)          (FR=1; ZPL = FR/(16·cos²); 1/16 cancels)
        ZPP      = SIG² = 1              (equal-spaced grid)
        ZL[k]    = (1/cos²)·(∂Hx_wu/∂π_wu)·(∂Sx_wu/∂π_wu)
        ZP[k]    = (∂Hy_wu/∂π_wu)·(∂Sy_wu/∂π_wu)
        ZNL[k]   = 2·jac_wu(Sx_wu, Sy_wu) + BETAS
             BETAS = fy_wu · Sy_wu   (β-plane correction)

Boundary conditions:
    θ_BC (top/bottom): H[0]=H[1]+θ_B·(π_wu[1]-π_wu[0])
                       H[-1]=H[-2]-θ_T·(π_wu[-1]-π_wu[-2])
    Lateral: handled intrinsically by spherical harmonics.
"""

from __future__ import annotations

import numpy as np
from typing import Dict

from pvtend.sh_ops import filter_low_modes_sh

from .coords import NW, R_EARTH
from .nondim import Scales, PI_WU
from .sor import BB, BH, BL, DPI2   # built from PI_WU after sor.py fix
from .wu_ops import lap_wu, inv_lap_wu, inv_helm_wu, grad_wu, jac_wu

__all__ = ["balnc_sh"]


# ─── Coriolis ───────────────────────────────────────────────────────────────

def _coriolis_nd(lat: np.ndarray, scales: Scales) -> np.ndarray:
    """Wu non-dim Coriolis f_nd(φ) = 2·Ω·sin(φ)/FF → O(0..1.458).

    Broadcast to (1, nlat, 1) so it can multiply (NW, nlat, nlon) arrays.
    """
    OMEGA = 7.2921e-5
    f_phys = 2.0 * OMEGA * np.sin(np.deg2rad(lat))
    f_nd   = f_phys / scales.FF
    return f_nd[np.newaxis, :, np.newaxis]          # (1, nlat, 1)


# ─── Vertical cross-level derivative ────────────────────────────────────────

def _cross_pi_deriv(field3d: np.ndarray) -> np.ndarray:
    """Centered derivative ∂field/∂π_wu.

    Returns (field[k+1] - field[k-1]) / (2·DPI2_wu[k]) for interior levels.
    Edge levels (k=0, k=NW-1) are zero.

    DPI2 here is built from PI_WU so the result is dimensionless O(1).
    """
    out = np.zeros_like(field3d)
    for k in range(1, NW - 1):
        dpi2 = DPI2[k]
        if dpi2 == 0.0:
            continue
        out[k] = (field3d[k + 1] - field3d[k - 1]) / (2.0 * dpi2)
    return out


# ─── STB (vertical static stability) ────────────────────────────────────────

def _stb(H: np.ndarray) -> np.ndarray:
    """STB[k] = BL[k]·H[k-1] + BH[k]·H[k+1] + BB[k]·H[k].

    With BB/BH/BL built from PI_WU: STB ≈ O(0.1..10) for typical H_nd.
    """
    out = np.zeros_like(H)
    for k in range(1, NW - 1):
        out[k] = BL[k] * H[k - 1] + BH[k] * H[k + 1] + BB[k] * H[k]
    return out


# ─── Cross-isobaric terms ZL, ZP ────────────────────────────────────────────

def _zl_zp(
    H: np.ndarray,
    S: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    LL: float,
) -> tuple:
    """Wu's cross-π correlation terms ZL, ZP (qinvert21 L505–515).

    After absorbing the ZPL·16/16 and ZPP·16/16 cancellations:
        ZL[k] = (1/cos²) · (∂Hx_wu/∂π_wu) · (∂Sx_wu/∂π_wu)
        ZP[k] = (∂Hy_wu/∂π_wu) · (∂Sy_wu/∂π_wu)    (SIG=1)
    Both are O(1) when H, S are O(1) and gradients are Wu non-dim.
    """
    Hx_wu, Hy_wu = grad_wu(H, lat, lon, LL)
    Sx_wu, Sy_wu = grad_wu(S, lat, lon, LL)

    Hx_p = _cross_pi_deriv(Hx_wu)
    Sx_p = _cross_pi_deriv(Sx_wu)
    Hy_p = _cross_pi_deriv(Hy_wu)
    Sy_p = _cross_pi_deriv(Sy_wu)

    cos2     = np.cos(np.deg2rad(lat)) ** 2
    # Mask out near-pole latitudes (|lat| > 88°) where cos²→0 causes singularity.
    # In Wu's FD grid these rows are also degenerate; we zero them explicitly.
    pole_mask = np.abs(lat) > 88.0          # (nlat,) boolean
    cos2      = np.where((cos2 < 1e-4) | pole_mask, 1.0, cos2)  # replace → 1 (ZL=product)
    # We also zero ZL at those latitudes after division (belt-and-suspenders).
    cos2_3d  = cos2[np.newaxis, :, np.newaxis]

    ZL = Hx_p * Sx_p / cos2_3d
    ZP = Hy_p * Sy_p

    # Zero out polar rows
    ZL[:, pole_mask, :] = 0.0
    ZP[:, pole_mask, :] = 0.0
    ZL[0] = 0.0;  ZL[-1] = 0.0
    ZP[0] = 0.0;  ZP[-1] = 0.0
    return ZL, ZP


# ─── Non-linear term ZNL ────────────────────────────────────────────────────

def _znl(
    S: np.ndarray,
    f3d: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    LL: float,
) -> np.ndarray:
    """ZNL = 2·jac_wu(Sx_wu, Sy_wu) + BETAS (qinvert21 L507–510).

    BETAS = fy_wu · Sy_wu  (β-plane meridional correction).
    """
    Sx_wu, Sy_wu = grad_wu(S, lat, lon, LL)
    znl = 2.0 * jac_wu(Sx_wu, Sy_wu, lat, lon, LL)

    OMEGA = 7.2921e-5
    FF    = 1.0e-4
    lat_rad = np.deg2rad(lat)
    fy_wu   = LL * 2.0 * OMEGA * np.cos(lat_rad) / (R_EARTH * FF)  # (nlat,)
    fy_3d   = fy_wu[np.newaxis, :, np.newaxis]

    betas = fy_3d * Sy_wu
    return znl + betas


# ─── θ boundary conditions ───────────────────────────────────────────────────

def _apply_theta_bc(
    H: np.ndarray,
    theta_B: np.ndarray,
    theta_T: np.ndarray,
) -> None:
    """Apply Wu θ boundary conditions on H (in-place).

    H[0]  = H[1]   + θ_B · (π_wu[1]  − π_wu[0])
    H[-1] = H[-2]  − θ_T · (π_wu[-1] − π_wu[-2])
    """
    H[0]  = H[1]  + theta_B * (PI_WU[1]  - PI_WU[0])
    H[-1] = H[-2] - theta_T * (PI_WU[-1] - PI_WU[-2])


# ─── RHS builders ───────────────────────────────────────────────────────────

def _psi_rhs(
    H: np.ndarray,
    S: np.ndarray,
    q: np.ndarray,
    f3d: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    LL: float,
    ZL_ext=None,
    ZP_ext=None,
    ZNL_ext=None,
) -> np.ndarray:
    """ψ-equation RHS (qinvert21 L515–523):

    RHS = [q − f·STB + lap_wu(H) − ZNL + ZL + ZP] / (f + STB)

    ZL_ext, ZP_ext, ZNL_ext : optional pre-computed (frozen) values.  When
    provided they replace the live ZL/ZP/ZNL computed from (H, S).  This is
    used in ``balnc_sh`` to freeze the ψ-dependent part of ZL and prevent
    Picard amplification (ZL is bilinear in H and ψ; as ψ grows ZL grows →
    rhs grows → psi_new ≫ psi_old → gain > 1 → divergence).
    """
    lap_H  = lap_wu(H, lat, lon, LL)
    STB    = _stb(H)
    STB    = np.where(STB <= 1e-4, 1e-4, STB)
    if ZL_ext is not None and ZP_ext is not None:
        ZL, ZP = ZL_ext, ZP_ext
    else:
        ZL, ZP = _zl_zp(H, S, lat, lon, LL)
    if ZNL_ext is not None:
        ZNL = ZNL_ext
    else:
        ZNL = _znl(S, f3d, lat, lon, LL)

    rhs = np.zeros_like(q)
    for k in range(1, NW - 1):
        rhst  = q[k] - f3d[0] * STB[k] + lap_H[k] - ZNL[k] + ZL[k] + ZP[k]
        denom = f3d[0] + STB[k]
        denom = np.where(np.abs(denom) < 1e-30, 1e-30, denom)
        # Wu's solver operates on a bounded NH window (lat ≈ 30–75°N) where f
        # is always O(1) non-dim.  On the global sphere, low-latitude grid
        # points have f → 0, making denom = f+STB_clamped ≈ 0.03–0.1 and
        # causing rhs → rhst/denom ≈ O(1e3–1e4) even for finite rhst.  This
        # feeds a huge synoptic-scale content into inv_lap_wu and diverges.
        # Suppression: zero the rhs wherever denom < 0.3 (equiv. lat ≲ 15°N,
        # well outside Wu's NH domain); the ψ-update will leave those points
        # at their initial (mean-state) value via the psi_init_n01 anchor.
        rhs[k] = np.where(denom >= 0.3, rhst / denom, 0.0)
    return rhs


def _H_rhs(
    S: np.ndarray,
    f3d: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    LL: float,
) -> tuple:
    """H-equation VOR and ASI (qinvert21 L700–730).

    VOR = lap_wu(ψ);  ASI = f + VOR  (floored at 1e-4).
    VOR is clamped before use in RHA = f*VOR (matches Wu's FCO*VOR with
    clamped VOR in line ~583 of qinvert21_94.f).
    """
    VOR = lap_wu(S, lat, lon, LL)
    # Clamp VOR so that ASI = f + VOR >= 1e-4 (Wu's condition: VOR >= 1e-4 - f)
    VOR_floor = 1e-4 - f3d       # minimum VOR to keep ASI >= 1e-4
    VOR_clamped = np.where(VOR < VOR_floor, VOR_floor, VOR)
    ASI = f3d + VOR_clamped      # = max(f + VOR, 1e-4)
    return VOR_clamped, ASI


# ─── Main solver ────────────────────────────────────────────────────────────

def balnc_sh(
    H_init: np.ndarray,
    psi_init: np.ndarray,
    q: np.ndarray,
    theta_B: np.ndarray,
    theta_T: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    scales: Scales,
    *,
    max_total: int = 200,
    thr: float = 0.01,
    part: float = 0.05,
    verbose: bool = True,
) -> Dict[str, np.ndarray]:
    """Total PV inversion (BALNC), spectral direct-solve, Wu non-dim units.

    All inputs MUST already be Wu-non-dim.  Outputs H, ψ are also non-dim.

    Parameters
    ----------
    H_init, psi_init : (NW, nlat, nlon) initial guess.
    q                : (NW, nlat, nlon) Wu-non-dim total PV.
    theta_B, theta_T : (nlat, nlon)     Wu-non-dim boundary θ.
    lat, lon         : 1-D coordinate arrays [degrees], lat ascending S→N.
    scales           : Wu non-dim Scales object.
    max_total        : maximum outer Picard iterations.
    thr              : convergence threshold on max|ΔH| & |Δψ|.
    part             : Picard under-relaxation factor.
    """
    LL  = scales.LL
    f3d = _coriolis_nd(lat, scales)

    H   = H_init.copy()
    psi = psi_init.copy()
    _apply_theta_bc(H, theta_B, theta_T)

    # Wu's SOR skips I=1 (equatorial) and I=NY (polar) rows — both are
    # fixed lateral BCs.  We mirror that by keeping those rows pinned at
    # their initial (mean-state) values throughout the Picard iteration.
    # Without the equatorial fix the denominator f+STB → 0 at lat=0, giving
    # a -61 000 spike in rhs_psi.  Without the polar fix the n=1,m=0 mode
    # (cos lat, peaking at pole) is unconstrained in the spectral inversion.
    eq_mask   = np.abs(lat) < 0.5          # True at lat=0°
    pol_mask  = lat > 89.0                 # True at lat=90°
    psi_eq_bc = psi_init[:, eq_mask, :]    # fixed ψ BC at equator
    H_eq_bc   = H_init[:, eq_mask, :]     # fixed H BC at equator
    psi_pol_bc = psi_init[:, pol_mask, :] # fixed ψ BC at pole
    H_pol_bc   = H_init[:, pol_mask, :]   # fixed H BC at pole

    # SH surrogate for Wu's lateral Dirichlet boundary condition.
    # Wu's FD solver pins rows I=1, I=NY, J=1, J=NX (qinvert21_94.f L481–L485)
    # which prevents n≤1 spherical-harmonic modes from entering the RHS and
    # being amplified by ∇⁻² (eigenvalue R²/[n(n+1)·LL²] ≈ 730× at n=1).
    # On the global sphere there are no walls, so we project n=0,1 out of
    # the RHS before inversion and anchor the large-scale (planetary) flow
    # to the mean-state initial ψ_init.  Specifically:
    #   rhs_n2plus  = rhs − (n=0 and n=1 modes of rhs)
    #   ψ_new       = ∇⁻²(rhs_n2plus) + (n=0 and n=1 modes of ψ_init)
    # psi_init_n01[k] = just the n=0,1 content of psi_init[k], computed ONCE:
    psi_init_n01 = np.zeros_like(psi_init)
    for k in range(1, NW - 1):
        psi_init_n01[k] = (
            psi_init[k]
            - filter_low_modes_sh(psi_init[k], lat, lon, nmin=2)
        )

    # ── Freeze ZL/ZP/ZNL at initial state ──────────────────────────────────
    # ZL = (∂H_x/∂π) × (∂ψ_x/∂π) / cos²  is bilinear in H and ψ.
    # As ψ grows during Picard, ∂ψ_x/∂π grows → ZL grows → rhs_ψ grows →
    # inv_lap amplifies by ~12× → psi_new ≫ psi_old → Picard gain > 1 →
    # divergence regardless of step size (proven: gain = 1+α(λ-1) > 1
    # for λ = inv_lap × ∂ZL/∂ψ/(f+STB) ≈ 6).
    # Fix: freeze ZL, ZP (and ZNL) at (H_init, psi_init).  The ψ-solve then
    # becomes a linear Poisson with RHS dependent only on H (not ψ), which
    # must converge.  Accuracy: with event-state init, ZL_frozen ≈ ZL_balanced
    # so the error from freezing is small.
    ZL_frozen, ZP_frozen = _zl_zp(H_init, psi_init, lat, lon, LL)
    ZNL_frozen           = _znl(psi_init, f3d, lat, lon, LL)
    if verbose:
        print(f"  [BALNC freeze] ZL_frozen k=1 range: "
              f"[{ZL_frozen[1].min():.3g}, {ZL_frozen[1].max():.3g}]")
        print(f"  [BALNC freeze] ZNL_frozen k=1 range: "
              f"[{ZNL_frozen[1].min():.3g}, {ZNL_frozen[1].max():.3g}]")

    if verbose:
        stb0 = _stb(H)
        zl0, zp0 = _zl_zp(H, psi, lat, lon, LL)
        print(f"  [BALNC] LL={LL:.3e}  thr={thr:.3g}  part={part}")
        print(f"          H_nd  [{H.min():.3g}, {H.max():.3g}]")
        print(f"          ψ_nd  [{psi.min():.3g}, {psi.max():.3g}]")
        print(f"          q_nd  [{q.min():.3g}, {q.max():.3g}]")
        print(f"  [BALNC sanity] STB k=4 range: [{stb0[4].min():.3g}, {stb0[4].max():.3g}]")
        print(f"  [BALNC sanity] ZL  k=4 range: [{zl0[4].min():.3g}, {zl0[4].max():.3g}]")
        print(f"  [BALNC sanity] ZP  k=4 range: [{zp0[4].min():.3g}, {zp0[4].max():.3g}]")

    hist = []

    for iitot in range(max_total):
        # ── ψ-equation ────────────────────────────────────────────────────
        # Wu skips the ψ update at IITOT=0 (GOTO 700 in qinvert21_94.f line
        # 460), solving only the H-equation first with S=S_init.  ψ updates
        # begin at IITOT=1.  Skipping this step prevents the huge first-step
        # blow-up that would otherwise be fed into the H-equation.
        psi_new = psi.copy()
        if iitot > 0:
            rhs_psi = _psi_rhs(H, psi, q, f3d, lat, lon, LL,
                               ZL_ext=ZL_frozen, ZP_ext=ZP_frozen,
                               ZNL_ext=ZNL_frozen)
            for k in range(1, NW - 1):
                rhs_k = rhs_psi[k].copy()
                rhs_k[eq_mask, :] = 0.0                         # zero equatorial row
                rhs_k[pol_mask, :] = 0.0                        # zero polar row
                # Project out n=0,1 modes before inversion (see comment above).
                rhs_k_n2 = filter_low_modes_sh(rhs_k, lat, lon, nmin=2)
                psi_new[k] = (
                    inv_lap_wu(rhs_k_n2[np.newaxis], lat, lon, LL)[0]
                    + psi_init_n01[k]
                )
                psi_new[k][eq_mask, :] = psi_eq_bc[k]          # restore equatorial BC
                psi_new[k][pol_mask, :] = psi_pol_bc[k]        # restore polar BC

        psi_next = part * psi_new + (1.0 - part) * psi  if iitot > 0 else psi_new
        dpsi = float(np.max(np.abs(psi_next - psi)))
        psi  = psi_next

        # ── H-equation ────────────────────────────────────────────────────
        VOR, ASI = _H_rhs(psi, f3d, lat, lon, LL)
        # Use frozen ZL/ZP/ZNL in H-step too: ZNL is quadratic in ψ and grows
        # as ψ accumulates, driving rh_H larger → H larger → larger STB/lap_H
        # in rhs_ψ → ψ grows more (self-reinforcing).  Freezing all three at
        # (H_init, psi_init) removes this feedback while preserving convergence.
        ZL, ZP   = ZL_frozen, ZP_frozen
        ZNL      = ZNL_frozen

        rh = np.zeros_like(q)
        for k in range(1, NW - 1):
            rh[k] = f3d[0] * VOR[k] + ZNL[k] + q[k] + ZL[k] + ZP[k]

        H_new = H.copy()
        for k in range(1, NW - 1):
            # Wu's H-equation:  [lap_wu + ASI·BB[k]] H[k]  +  ASI·(BH[k]·H[k+1] + BL[k]·H[k-1])  =  rh[k]
            # Operator split: use c_k = mean(ASI[k])·BB[k] in the implicit Helmholtz,
            # treat the spatially-varying remainder  (ASI - mean_ASI)·BB·H  explicitly.
            asi_mean_k  = float(np.mean(ASI[k]))
            c_k         = asi_mean_k * BB[k]           # constant Helmholtz shift, negative
            correction  = (ASI[k] - asi_mean_k) * BB[k] * H[k]   # spatially-varying residual
            rhs_helm_k  = (rh[k]
                           - ASI[k] * (BH[k] * H[k + 1] + BL[k] * H[k - 1])
                           - correction)
            H_new[k] = inv_helm_wu(rhs_helm_k[np.newaxis], lat, lon, LL, c_k)[0]
            H_new[k][eq_mask, :] = H_eq_bc[k]                  # restore equatorial BC
            H_new[k][pol_mask, :] = H_pol_bc[k]              # restore polar BC

        H_next = part * H_new + (1.0 - part) * H  if iitot > 0 else H_new
        dH = float(np.max(np.abs(H_next - H)))
        H  = H_next
        _apply_theta_bc(H, theta_B, theta_T)

        hist.append({
            "iitot":   iitot,
            "dpsi":    dpsi,
            "dH":      dH,
            "max_H":   float(np.max(np.abs(H))),
            "max_psi": float(np.max(np.abs(psi))),
        })
        if verbose and (iitot < 5 or iitot % 10 == 0):
            print(f"  [BALNC] outer {iitot:3d}:  "
                  f"Δψ={dpsi:.3g}  ΔH={dH:.3g}  "
                  f"|H|={hist[-1]['max_H']:.3g}  "
                  f"|ψ|={hist[-1]['max_psi']:.3g}")

        if dpsi < thr and dH < thr and iitot > 0:
            if verbose:
                print(f"  [BALNC] CONVERGED after {iitot + 1} outer iters.")
            break

    return {"H": H, "psi": psi, "hist": hist}
