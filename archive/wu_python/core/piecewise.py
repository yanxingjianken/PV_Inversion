# wu_python/core/piecewise.py — BALP: Piecewise perturbation inversion (Pass D)
"""Exact 1:1 port of qinvertp21_94.f subroutine BALP with numba acceleration.

ALL inputs to BALP are NON-DIMENSIONAL (Fortran convention):
  H_nd  = H_dim / HND            (HND = THO*DPI/G ≈ 28.3 m)
  ψ_nd  = ψ_dim / (1e5 * HND)    (matching Fortran write/read of ψ/1e5)
  Q_nd  = PIF*Q_dim/(1e2*QCONST) (same as BALNC)
  θ_nd  = θ_dim / THO            (THO = FF²*LL²/DPI)

Outputs are in the same non-dimensional units; re-dimensionalize by:
  ψ_dim = ψ_nd * (1e5 * HND)
  H_dim = H_nd * HND
"""
import numpy as np
from pathlib import Path
import sys

_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from wu_python.config import PIECES, OMEGS, OMEGH, PART, THRSH  # noqa: E402
from wu_python.core.grid import NY, NX, A, AP as AP_coslat, DP, DL, SIGM, FC as FC_1d  # noqa: E402
from wu_python.core.nondim import G, THO, DPI, PI_WU, BB, BH, BL, CP, FF  # noqa: E402
from wu_python.core.balance import QCONST  # correct QCONST from balance.py

try:
    from numba import njit
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False
    def njit(*args, **kw): return lambda f: f

NW = 10; NL = NW
HND = THO * DPI / G   # ~28.3 m
LL = 2e7/np.pi        # Earth radius
THO_val = FF*FF*LL*LL/DPI  # ≈ 5.556

FR = 1.0; SIG = SIGM
_DPI2 = np.array([(PI_WU[k+1]-PI_WU[k-1])/2.0 if 0<k<NW-1 else 0.0 for k in range(NW)])
FCO = np.tile(FC_1d[:, np.newaxis], (1, NX))
FCM = FCO
AC1, AC2, AC3, AC4, AC5 = A[:,0].copy(), A[:,1].copy(), A[:,2].copy(), A[:,3].copy(), A[:,4].copy()
AC3_2d = AC3[:, np.newaxis]
# Fortran BALP: APS = cos(lat) (map-factor, NOT meters!)
# ZNC = 2*FR*SIG²*MFC/(APS²), MFC=1 on lat/lon → ZNC = 2/cos²(lat)
APS_fortran = AP_coslat  # cos(lat), matches Fortran call: CALL BALP(..., AP, ...)
ZNC = (2.0 * FR * SIG * SIG / (APS_fortran * APS_fortran))[:, np.newaxis]


def _nondim_q(Q_pvu):
    """Non-dimensionalize PV: Q_nd = PIF*Q_PVU/(1e2*QCONST)."""
    nw, ny, nx = Q_pvu.shape
    Q_nd = np.zeros_like(Q_pvu)
    for k in range(1, nw - 1):
        PIF_k = (PI_WU[k] / CP) ** 2.5
        Q_nd[k] = PIF_k * Q_pvu[k] / (1e2 * QCONST)
    return Q_nd


def _compute_mean_coeffs(HB_nd, SBR_nd):
    """Pre-compute BALP coefficients from NON-DIMENSIONAL mean state."""
    SLL = np.zeros((NL, NY, NX)); SPP = np.zeros((NL, NY, NX))
    SLP = np.zeros((NL, NY, NX)); AVO = np.zeros((NL, NY, NX))
    STB = np.zeros((NL, NY, NX)); ASI = np.zeros((NL, NY, NX))
    BSI = np.zeros((NL, NY, NX)); APHI = np.zeros((NL, NY, NX))

    for k in range(2, NL - 1):
        SLL[k,1:-1,1:-1] = ZNC[1:-1]*(SBR_nd[k,1:-1,2:]+SBR_nd[k,1:-1,:-2]-2.0*SBR_nd[k,1:-1,1:-1])
        SPP[k,1:-1,1:-1] = ZNC[1:-1]*(SBR_nd[k,:-2,1:-1]+SBR_nd[k,2:,1:-1]-2.0*SBR_nd[k,1:-1,1:-1])
        SLP[k,1:-1,1:-1] = ZNC[1:-1]*(SBR_nd[k,:-2,2:]-SBR_nd[k,:-2,:-2]-SBR_nd[k,2:,2:]+SBR_nd[k,2:,:-2])/2.0
        AVO[k,1:-1,1:-1] = FCM[1:-1,1:-1] + FR*(
            AC1[1:-1,np.newaxis]*SBR_nd[k,:-2,1:-1] + AC2[1:-1,np.newaxis]*SBR_nd[k,1:-1,:-2]
            + AC3[1:-1,np.newaxis]*SBR_nd[k,1:-1,1:-1]
            + AC4[1:-1,np.newaxis]*SBR_nd[k,1:-1,2:] + AC5[1:-1,np.newaxis]*SBR_nd[k,2:,1:-1])
        STB[k] = BH[k]*HB_nd[k+1] + BL[k]*HB_nd[k-1] + BB[k]*HB_nd[k]
        AVO[k] = np.maximum(AVO[k], 0.01)
        STB[k] = np.maximum(STB[k], 0.01)
        ASI[k] = BB[k]*AVO[k]/(FR*STB[k]*AC3_2d)
        BSI[k] = 1.0 + ASI[k]*FCO
        bi = FCO*AC3_2d - 2.0*(SLL[k]+SPP[k])
        bi = np.minimum(bi, 0.0)
        APHI[k] = bi/(FR*STB[k]*AC3_2d)
    return {'SLL':SLL,'SPP':SPP,'SLP':SLP,'AVO':AVO,'STB':STB,'ASI':ASI,'BSI':BSI,'APHI':APHI}


@njit
def _psi_sor_level(SP_k, SRHS_k, SLL_k, SPP_k, SLP_k, ASI_k, BSI_k,
                   FCO, AC1, AC2, AC3, AC4, AC5, SIG, omegs, thrsh, maxx):
    """Numba-accelerated 2D ψ SOR for one level k."""
    ny, nx = SP_k.shape
    for itc in range(maxx):
        zc = True
        for j in range(2, nx - 1):
            for i in range(2, ny - 1):
                RSA = (AC1[i]*SP_k[i-1,j] + AC2[i]*SP_k[i,j-1] + AC3[i]*SP_k[i,j]
                       + AC4[i]*SP_k[i,j+1] + AC5[i]*SP_k[i+1,j])
                SXX = SP_k[i,j+1] + SP_k[i,j-1] - 2.0*SP_k[i,j]
                SYY = SP_k[i-1,j] + SP_k[i+1,j] - 2.0*SP_k[i,j]
                SXY = (SP_k[i-1,j+1] - SP_k[i+1,j+1] - SP_k[i-1,j-1] + SP_k[i+1,j-1])/4.0
                BETAS = (SIG*SIG*(FCO[i-1,j]-FCO[i+1,j])*(SP_k[i-1,j]-SP_k[i+1,j])/4.0
                         + (FCO[i,j+1]-FCO[i,j-1])*(SP_k[i,j+1]-SP_k[i,j-1])/4.0)
                RS = (BSI_k[i,j]*RSA + ASI_k[i,j]*(
                    BETAS + SLL_k[i,j]*SYY + SPP_k[i,j]*SXX - SLP_k[i,j]*SXY) - SRHS_k[i,j])
                DLS = BSI_k[i,j]*AC3[i] - 2.0*ASI_k[i,j]*(SLL_k[i,j]+SPP_k[i,j])
                ZSI = SP_k[i,j]
                SP_k[i,j] = ZSI - omegs*RS/DLS
                if abs(SP_k[i,j] - ZSI) > thrsh: zc = False
        if zc: break
    return SP_k


def _balp_solve(SP_nd, HP_nd, QP_nd, TP_nd, coeffs, HB_nd, SBR_nd,
                omegs=OMEGS, omegh=OMEGH, part=PART, thrsh=THRSH,
                maxx=200, maxxt=200):
    """Core BALP solver — all inputs NON-DIMENSIONAL."""
    SLL=coeffs['SLL']; SPP=coeffs['SPP']; SLP=coeffs['SLP']; AVO=coeffs['AVO']
    STB=coeffs['STB']; ASI=coeffs['ASI']; BSI=coeffs['BSI']; APHI=coeffs['APHI']

    for iitot in range(maxxt):
        OS = SP_nd.copy(); OH = HP_nd.copy()

        # ── ψ RHS (SRHS) ──
        SRHS = np.zeros((NL, NY, NX)); RHS = np.zeros((NL, NY, NX))
        for k in range(2, NL - 1):
            for j in range(2, NX - 1):
                for i in range(2, NY - 1):
                    dp4 = 4.0*_DPI2[k]; aps2 = APS_fortran[i]*APS_fortran[i]
                    R1BS=(SBR_nd[k+1,i,j+1]-SBR_nd[k+1,i,j-1]-SBR_nd[k-1,i,j+1]+SBR_nd[k-1,i,j-1])/dp4
                    R1BH=(HB_nd[k+1,i,j+1]-HB_nd[k+1,i,j-1]-HB_nd[k-1,i,j+1]+HB_nd[k-1,i,j-1])/dp4
                    R1PS=(SP_nd[k+1,i,j+1]-SP_nd[k+1,i,j-1]-SP_nd[k-1,i,j+1]+SP_nd[k-1,i,j-1])/dp4
                    R1PH=(HP_nd[k+1,i,j+1]-HP_nd[k+1,i,j-1]-HP_nd[k-1,i,j+1]+HP_nd[k-1,i,j-1])/dp4
                    R2BS=(SBR_nd[k+1,i-1,j]-SBR_nd[k+1,i+1,j]-SBR_nd[k-1,i-1,j]+SBR_nd[k-1,i+1,j])/dp4
                    R2BH=(HB_nd[k+1,i-1,j]-HB_nd[k+1,i+1,j]-HB_nd[k-1,i-1,j]+HB_nd[k-1,i+1,j])/dp4
                    R2PS=(SP_nd[k+1,i-1,j]-SP_nd[k+1,i+1,j]-SP_nd[k-1,i-1,j]+SP_nd[k-1,i+1,j])/dp4
                    R2PH=(HP_nd[k+1,i-1,j]-HP_nd[k+1,i+1,j]-HP_nd[k-1,i-1,j]+HP_nd[k-1,i+1,j])/dp4
                    RHS[k,i,j] = QP_nd[k,i,j] + FR*((R1BS*R1PH+R1BH*R1PS)/aps2
                                                      + SIG*SIG*(R2BS*R2PH+R2BH*R2PS))
                    SRHS[k,i,j] = ((RHS[k,i,j] - AVO[k,i,j]*(
                        BH[k]*HP_nd[k+1,i,j]+BL[k]*HP_nd[k-1,i,j]))/(FR*STB[k,i,j])
                        + ASI[k,i,j]*(AC1[i]*HP_nd[k,i-1,j]+AC2[i]*HP_nd[k,i,j-1]
                                      + AC4[i]*HP_nd[k,i,j+1]+AC5[i]*HP_nd[k,i+1,j]))

        # ── ψ SOR per level (numba-accelerated) ──
        if _HAS_NUMBA:
            for k in range(2, NL - 1):
                SP_nd[k] = _psi_sor_level(SP_nd[k], SRHS[k], SLL[k], SPP[k], SLP[k],
                                          ASI[k], BSI[k], FCO, AC1, AC2, AC3, AC4, AC5,
                                          SIG, omegs, thrsh, maxx)
        else:
            for k in range(2, NL - 1):
                for itc in range(maxx):
                    zc = True
                    for j in range(2, NX - 1):
                        for i in range(2, NY - 1):
                            RSA = (AC1[i]*SP_nd[k,i-1,j]+AC2[i]*SP_nd[k,i,j-1]+AC3[i]*SP_nd[k,i,j]
                                   +AC4[i]*SP_nd[k,i,j+1]+AC5[i]*SP_nd[k,i+1,j])
                            SXX = SP_nd[k,i,j+1]+SP_nd[k,i,j-1]-2.0*SP_nd[k,i,j]
                            SYY = SP_nd[k,i-1,j]+SP_nd[k,i+1,j]-2.0*SP_nd[k,i,j]
                            SXY = (SP_nd[k,i-1,j+1]-SP_nd[k,i+1,j+1]-SP_nd[k,i-1,j-1]+SP_nd[k,i+1,j-1])/4.0
                            BETAS = (SIG*SIG*(FCO[i-1,j]-FCO[i+1,j])*(SP_nd[k,i-1,j]-SP_nd[k,i+1,j])/4.0
                                     +(FCO[i,j+1]-FCO[i,j-1])*(SP_nd[k,i,j+1]-SP_nd[k,i,j-1])/4.0)
                            RS = (BSI[k,i,j]*RSA+ASI[k,i,j]*(
                                BETAS+SLL[k,i,j]*SYY+SPP[k,i,j]*SXX-SLP[k,i,j]*SXY)-SRHS[k,i,j])
                            DLS = BSI[k,i,j]*AC3[i]-2.0*ASI[k,i,j]*(SLL[k,i,j]+SPP[k,i,j])
                            ZSI = SP_nd[k,i,j]; SP_nd[k,i,j] = ZSI-omegs*RS/DLS
                            if abs(SP_nd[k,i,j]-ZSI)>thrsh: zc=False
                    if zc: break

        if iitot > 0:
            SP_nd[1:-1,1:-1,1:-1] = part*SP_nd[1:-1,1:-1,1:-1]+(1.0-part)*OS[1:-1,1:-1,1:-1]

        # ── H RHS (HRHS) ──
        HRHS = np.zeros((NL, NY, NX))
        for k in range(2, NL - 1):
            for j in range(2, NX - 1):
                for i in range(2, NY - 1):
                    RH1 = (2.0/AC3[i])*(SLL[k,i,j]+SPP[k,i,j])*(
                        AC1[i]*SP_nd[k,i-1,j]+AC2[i]*SP_nd[k,i,j-1]+AC4[i]*SP_nd[k,i,j+1]+AC5[i]*SP_nd[k,i+1,j])
                    BETAS = (SIG*SIG*(FCO[i-1,j]-FCO[i+1,j])*(SP_nd[k,i-1,j]-SP_nd[k,i+1,j])/4.0
                             +(FCO[i,j+1]-FCO[i,j-1])*(SP_nd[k,i,j+1]-SP_nd[k,i,j-1])/4.0)
                    SXYp = (SP_nd[k,i-1,j+1]-SP_nd[k,i+1,j+1]-SP_nd[k,i-1,j-1]+SP_nd[k,i+1,j-1])/4.0
                    RH2 = (BETAS+SLL[k,i,j]*(SP_nd[k,i-1,j]+SP_nd[k,i+1,j])
                           +SPP[k,i,j]*(SP_nd[k,i,j-1]+SP_nd[k,i,j+1])-SLP[k,i,j]*SXYp)
                    HRHS[k,i,j] = APHI[k,i,j]*RHS[k,i,j]+RH1+RH2

        # ── H SOR (3D) ──
        for itc in range(maxx):
            hc = True
            for k in range(2, NL - 1):
                for j in range(2, NX - 1):
                    for i in range(2, NY - 1):
                        avo=AVO[k,i,j]; aph=APHI[k,i,j]; ac3=AC3[i]
                        if k == 2:
                            RS = (AC1[i]*HP_nd[k,i-1,j]+AC2[i]*HP_nd[k,i,j-1]+(ac3+aph*(BB[k]+BL[k])*avo)*HP_nd[k,i,j]
                                  +AC4[i]*HP_nd[k,i,j+1]+AC5[i]*HP_nd[k,i+1,j]
                                  +aph*avo*(BH[k]*HP_nd[k+1,i,j]+TP_nd[i,j,0]/_DPI2[k])-HRHS[k,i,j])
                            diag = ac3+aph*(BB[k]+BL[k])*avo
                        elif k == NL - 2:
                            RS = (AC1[i]*HP_nd[k,i-1,j]+AC2[i]*HP_nd[k,i,j-1]+(ac3+aph*(BB[k]+BH[k])*avo)*HP_nd[k,i,j]
                                  +AC4[i]*HP_nd[k,i,j+1]+AC5[i]*HP_nd[k,i+1,j]
                                  +aph*avo*(BL[k]*HP_nd[k-1,i,j]-TP_nd[i,j,1]/_DPI2[k])-HRHS[k,i,j])
                            diag = ac3+aph*(BB[k]+BH[k])*avo
                        else:
                            RS = (AC1[i]*HP_nd[k,i-1,j]+AC2[i]*HP_nd[k,i,j-1]+(ac3+aph*BB[k]*avo)*HP_nd[k,i,j]
                                  +AC4[i]*HP_nd[k,i,j+1]+AC5[i]*HP_nd[k,i+1,j]
                                  +aph*avo*(BH[k]*HP_nd[k+1,i,j]+BL[k]*HP_nd[k-1,i,j])-HRHS[k,i,j])
                            diag = ac3+aph*BB[k]*avo
                        ZM = HP_nd[k,i,j]; HP_nd[k,i,j] = ZM - omegh*RS/diag
                        if abs(HP_nd[k,i,j]-ZM) > thrsh: hc = False
            if hc: break

        if iitot > 0:
            HP_nd[1:-1,1:-1,1:-1] = part*HP_nd[1:-1,1:-1,1:-1]+(1.0-part)*OH[1:-1,1:-1,1:-1]

        # θ BCs (non-dimensional)
        dpe_bot = PI_WU[2]-PI_WU[1]; dpe_top = PI_WU[NL-1]-PI_WU[NL-2]
        HP_nd[1] = HP_nd[2] + TP_nd[:,:,0]*dpe_bot; HP_nd[NL-1] = HP_nd[NL-2] - TP_nd[:,:,1]*dpe_top
        SP_nd[1] = SP_nd[2] + TP_nd[:,:,0]*dpe_bot; SP_nd[NL-1] = SP_nd[NL-2] - TP_nd[:,:,1]*dpe_top

        if np.any(~np.isfinite(SP_nd)) or np.any(~np.isfinite(HP_nd)):
            print(f"  [BALP] NaN at outer iter {iitot+1}"); return False, iitot+1

    return False, maxxt


def balp_pieces(Q_event, Q_mean, PSI_mean, H_mean,
                THA_event_K=None, THA_mean_K=None, pieces=None, **kw):
    """Piecewise perturbation PV inversion (Pass D)."""
    if pieces is None: pieces = PIECES
    nw = PSI_mean.shape[0]; ny = PSI_mean.shape[1]; nx = PSI_mean.shape[2]
    def _pad(Q):
        if Q.shape[0] == nw-2:
            o = np.zeros((nw,ny,nx)); o[1:-1] = Q; return o
        return Q.copy()
    Qe, Qm = _pad(Q_event), _pad(Q_mean)

    # Non-dimensionalize all inputs (Fortran BALP convention)
    Qe_nd = _nondim_q(Qe)
    Qm_nd = _nondim_q(Qm)
    HB_nd = H_mean / HND                            # H_nd = H_dim / HND
    SBR_nd = PSI_mean / (1e5 * HND)                 # ψ_nd = ψ_dim / (1e5*HND)

    print("  [BALP] Computing mean-state coefficients (non-dim)...")
    coeffs = _compute_mean_coeffs(HB_nd, SBR_nd)
    print("  [BALP] Coefficients ready.")

    Qa_nd = Qe_nd - Qm_nd
    PSI_p = np.zeros((len(pieces), nw, ny, nx))
    H_p = np.zeros((len(pieces), nw, ny, nx))
    TP_nd = np.zeros((ny, nx, 2))

    for ip, cfg in enumerate(pieces):
        ll = cfg['levels']; print(f"  [BALP] Piece {ip+1}/{len(pieces)}  levels={ll}")
        QP_nd = np.zeros((nw, ny, nx))
        for kp in ll:
            kf = kp
            if 0 < kf < nw-1: QP_nd[kf] = Qa_nd[kf]

        SP_nd = np.zeros((nw, ny, nx)); HP_nd = np.zeros((nw, ny, nx))
        cv, ni = _balp_solve(SP_nd, HP_nd, QP_nd, TP_nd, coeffs, HB_nd, SBR_nd, **kw)
        # Re-dimensionalize: ψ_dim = ψ_nd * 1e5 * HND, H_dim = H_nd * HND
        PSI_p[ip] = SP_nd * (1e5 * HND)
        H_p[ip] = HP_nd * HND
        print(f"    → {'converged' if cv else f'max_iters({ni})'}  ψ′=[{PSI_p[ip].min():.2e},{PSI_p[ip].max():.2e}]")
    return PSI_p, H_p
