import re
with open("sh/sh_ppvi/invert_piece.py", "r") as f:
    text = f.read()

# I need to verify that B_m (which multiplies D2) matches the operator D2Phi used in _F_charney.
# In _solve_precond, M = lam * diag(A_m) + f0 * diag(B_m) * D2.
# A_m corresponds to B_stb, which multiplies laplacian(dpsi).
# B_m corresponds to A_avo, which multiplies D2(dPhi).
# If we replace dPhi = f0*dpsi with charney, maybe the preconditioner also needs adjusting. 
# For now, let's keep f0 in the preconditioner.

# Maybe the inner Charney inversion fails to find proper scale?
# In the original f0-plane: dPhi = f0 * dpsi
# In charney: lap(dPhi) = f * lap(dpsi) + beta * dpsi_y + J + J.
# So dPhi ~ f * dpsi. Yes, it should scale correctly.
# But laplacian_inv zeroes the horizontal mean? Yes.

# What about the fact that charney RHS is passed K slices? 
# Wait, dPhi[k] = laplacian_inv(charney_rhs_val[k:k+1], lat, lon)[0]
# Is that correct? Yes.
