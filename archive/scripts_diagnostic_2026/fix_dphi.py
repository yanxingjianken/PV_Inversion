import re
with open("sh/sh_ppvi/invert_piece.py", "r") as f:
    text = f.read()

# Fix the fast exit delta_Phi
old_eval_1 = """        delta_Phi = _apply_anomaly_bc(f0 * x_sol)"""
new_eval_1 = """        charney_rhs_val = _lin_charney_rhs(x_sol, psi_bar, lat, lon)
        delta_Phi = np.zeros_like(x_sol)
        for k in range(shape3d[0]):
            delta_Phi[k] = laplacian_inv(charney_rhs_val[k:k+1], lat, lon, parity="scalar")[0]
        delta_Phi = _apply_anomaly_bc(delta_Phi)"""
text = text.replace(old_eval_1, new_eval_1)

# Fix the main output delta_Phi
old_eval_2 = """    # δΦ from f₀-plane balance (consistent with A_f0 operator)
    delta_Phi = _apply_anomaly_bc(f0 * delta_psi)"""
new_eval_2 = """    # δΦ from Charney balance
    charney_rhs_val = _lin_charney_rhs(delta_psi, psi_bar, lat, lon)
    delta_Phi = np.zeros_like(delta_psi)
    for k in range(shape3d[0]):
        delta_Phi[k] = laplacian_inv(charney_rhs_val[k:k+1], lat, lon, parity="scalar")[0]
    delta_Phi = _apply_anomaly_bc(delta_Phi)"""
text = text.replace(old_eval_2, new_eval_2)

with open("sh/sh_ppvi/invert_piece.py", "w") as f:
    f.write(text)
