import re
with open("sh/sh_ppvi/invert_piece.py", "r") as f:
    text = f.read()

# Let's add a check for inf/nan in _F_charney
old_eval = """    def _F_charney(dpsi_flat: np.ndarray) -> np.ndarray:
        dpsi   = dpsi_flat.reshape(shape3d)
        charney_rhs_val = _lin_charney_rhs(dpsi, psi_bar, lat, lon)
        
        # Invert level by level
        dPhi   = np.zeros_like(dpsi)
        for k in range(shape3d[0]):
            dPhi[k] = laplacian_inv(charney_rhs_val[k:k+1], lat, lon, parity="scalar")[0]
            
        dPhi   = _apply_anomaly_bc(dPhi)
        lap_dp = laplacian(dpsi, lat, lon)
        q_si   = coeff_grid * (A_avo_full * d_pi_pi(dPhi) + B_stb_full * lap_dp)
        return q_si.ravel()"""

new_eval = """    def _F_charney(dpsi_flat: np.ndarray) -> np.ndarray:
        dpsi   = dpsi_flat.reshape(shape3d)
        charney_rhs_val = _lin_charney_rhs(dpsi, psi_bar, lat, lon)
        
        # Invert level by level
        dPhi   = np.zeros_like(dpsi)
        for k in range(shape3d[0]):
            dPhi[k] = laplacian_inv(charney_rhs_val[k:k+1], lat, lon, parity="scalar")[0]
            
        dPhi   = _apply_anomaly_bc(dPhi)
        lap_dp = laplacian(dpsi, lat, lon)
        q_si   = coeff_grid * (A_avo_full * d_pi_pi(dPhi) + B_stb_full * lap_dp)
        if np.any(np.isnan(q_si)) or np.any(np.isinf(q_si)):
            print("[lgmres] WARNING: NaN or Inf detected in _F_charney output!")
            q_si = np.nan_to_num(q_si, nan=0.0, posinf=1e10, neginf=-1e10)
        return q_si.ravel()"""

text = text.replace(old_eval, new_eval)

with open("sh/sh_ppvi/invert_piece.py", "w") as f:
    f.write(text)
