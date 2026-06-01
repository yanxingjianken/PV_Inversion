import re

with open("sh/sh_ppvi/invert_piece.py", "r") as f:
    text = f.read()

# Remove _direct_solve 
text = re.sub(r'# ─────────────────────────────────────────────────────────────\n#  Public API\n# ─────────────────────────────────────────────────────────────.*?def invert_piece\(', '# ─────────────────────────────────────────────────────────────\n#  Public API\n# ─────────────────────────────────────────────────────────────\n\ndef invert_piece(', text, flags=re.DOTALL)

# Replace the docstring of invert_piece.
orig_docstring = '''    """Invert a PV anomaly q′ → (δψ, δΦ) via direct spectral solve.

    Uses the horizontal-dominant term of the linearised ertel_q_sh operator:

        δψ = ∇⁻²(q′ × 1e-6 / A_safe)   where A = g·κ·π/(p·cp) · ∂²Φ̄/∂π²

    Properties:
      - Exact additivity: Σ δψ_piece = δψ_total (linearity of ∇⁻²).
      - Consistent with invert_full direct solve (same A formula).
      - q′ must be computed with ertel_q_sh for numerical consistency with A.
      - One-step solve: no iteration required.

    Args:
        q_prime:    PV anomaly (NW, nlat, nlon) [PVU]; zero outside the piece.
                    Must be computed as ertel_q_sh(psi_ev, Phi_ev) - ertel_q_sh(psi_bar, Phi_bar).
        mean_state: dict with 'psi' and 'Phi' (NW, nlat, nlon) — from invert_full.
        lat:        ascending latitude (degrees), NH grid.
        lon:        longitude (degrees).
        verbose:    print solver progress.
        tol:        kept for API compatibility; not used in direct solve.
        maxit:      kept for API compatibility; not used in direct solve.
        omega:      kept for API compatibility; not used.
        f0_deg:     reference latitude for f₀-plane δΦ (output only).'''

new_docstring = '''    """Invert a PV anomaly q′ → (δψ, δΦ) via LGMRES with Linearized Charney balance.

    Uses the full linearised ertel_q_sh operator with Charney balance mapping δψ → δΦ:

        δΦ = ∇⁻²( _lin_charney_rhs(δψ, ψ̄) )
        q_pred = coeff · (A_avo_full · d²(δΦ)/dπ² + B_stb_full · ∇²δψ)

    Then LGMRES minimizes ‖q_pred - q′‖.

    Args:
        q_prime:    PV anomaly (NW, nlat, nlon) [PVU]; zero outside the piece.
                    Must be computed as ertel_q_sh(psi_ev, Phi_ev) - ertel_q_sh(psi_bar, Phi_bar).
        mean_state: dict with 'psi' and 'Phi' (NW, nlat, nlon).
        lat:        ascending latitude (degrees), NH grid.
        lon:        longitude (degrees).
        verbose:    print solver progress.
        tol:        LGMRES relative tolerance.
        maxit:      maximum LGMRES iterations.
        omega:      kept for API compatibility; not used.
        f0_deg:     reference latitude for preconditioner f₀.'''

text = text.replace(orig_docstring, new_docstring)

# Change invert_piece body to call _piecewise_lgmres_solve:
orig_body = '''    sol = _direct_solve(
        q_prime, Phi_bar, lat, lon,
        f0_deg=f0_deg, verbose=verbose,
    )

    delta_psi = np.asarray(sol["delta_psi"])
    delta_Phi = np.asarray(sol["delta_Phi"])
    q_pred    = np.asarray(sol["q_pred"])

    dpsi_x, dpsi_y = gradient(delta_psi, lat, lon)

    return {
        "psi"              : delta_psi,
        "Phi"              : delta_Phi,
        "u"                : -dpsi_y,
        "v"                : dpsi_x,
        "q_pred"           : q_pred,
        "residual_history" : sol["residual_history"],
        "n_iter"           : sol["n_iter"],
        "converged"        : sol["converged"],
    }'''

new_body = '''    sol = _piecewise_lgmres_solve(
        q_prime, mean_state['psi'], Phi_bar, lat, lon,
        f0_deg=f0_deg, tol=tol, maxit=maxit, omega=omega, verbose=verbose,
    )

    delta_psi = np.asarray(sol["delta_psi"])
    delta_Phi = np.asarray(sol["delta_Phi"])
    q_pred    = np.asarray(sol["q_pred"])

    dpsi_x, dpsi_y = gradient(delta_psi, lat, lon)

    return {
        "psi"              : delta_psi,
        "Phi"              : delta_Phi,
        "u"                : -dpsi_y,
        "v"                : dpsi_x,
        "q_pred"           : q_pred,
        "residual_history" : sol.get("residual_history", []),
        "n_iter"           : sol.get("n_iter", 1),
        "converged"        : sol.get("converged", True),
    }'''

text = text.replace(orig_body, new_body)

# Replace f0 operator with charney in _piecewise_lgmres_solve
orig_f0_op = '''    # ── f₀-plane forward operator: δψ → q_pred_SI ───────────────────────
    # Significantly cheaper than full Charney (~0.02 s/call vs ~0.11 s).
    # dPhi = f₀·δψ  →  d²Phi/dpi² = f₀·D₂·δψ  (avoids laplacian_inv call).
    def _F_f0(dpsi_flat: np.ndarray) -> np.ndarray:
        dpsi   = dpsi_flat.reshape(shape3d)
        dPhi   = f0 * dpsi
        dPhi   = _apply_anomaly_bc(dPhi)
        lap_dp = laplacian(dpsi, lat, lon)
        q_si   = coeff_grid * (A_avo_full * d_pi_pi(dPhi) + B_stb_full * lap_dp)
        return q_si.ravel()

    # ── Right-preconditioned operator: A_right(y) = A_f0(M_spec⁻¹(y)) ──
    # Solves A_f0·M_spec⁻¹·y = b.  Solution: x = M_spec⁻¹·y_sol.
    def _F_right(y_flat: np.ndarray) -> np.ndarray:
        return _F_f0(_M_spectral(y_flat))'''

new_charney_op = '''    # ── Linearized Charney forward operator: δψ → q_pred_SI ──────────────
    # Relates dPhi to dpsi via linearized Charney balance: dPhi = ∇⁻²(_lin_charney_rhs)
    def _F_charney(dpsi_flat: np.ndarray) -> np.ndarray:
        dpsi   = dpsi_flat.reshape(shape3d)
        charney_rhs_val = _lin_charney_rhs(dpsi, psi_bar, lat, lon)
        
        # Invert level by level
        dPhi   = np.zeros_like(dpsi)
        for k in range(shape3d[0]):
            dPhi[k] = laplacian_inv(charney_rhs_val[k:k+1], lat, lon, parity="scalar")[0]
            
        dPhi   = _apply_anomaly_bc(dPhi)
        lap_dp = laplacian(dpsi, lat, lon)
        q_si   = coeff_grid * (A_avo_full * d_pi_pi(dPhi) + B_stb_full * lap_dp)
        return q_si.ravel()

    # ── Right-preconditioned operator: A_right(y) = A_F(M_spec⁻¹(y)) ──
    # Solves A_F·M_spec⁻¹·y = b.  Solution: x = M_spec⁻¹·y_sol.
    def _F_right(y_flat: np.ndarray) -> np.ndarray:
        return _F_charney(_M_spectral(y_flat))'''

text = text.replace(orig_f0_op, new_charney_op)

# Also fix the f0 evaluation when computing norms:
text = text.replace('_F_f0(delta_psi.ravel())', '_F_charney(delta_psi.ravel())')
# Replace r0_rel text and docstring prints
text = text.replace('(f₀-plane, RIGHT precond)', '(Linearized Charney, RIGHT precond)')

# Replace _F_f0 references in docstring of _piecewise_lgmres_solve:
doc_pattern = r'Uses f₀-PLANE balance: δΦ = f₀·δψ.  This makes the operator F: δψ → q_pred.*?computation in step 13.'
doc_repl = r'Uses LINEARIZED CHARNEY balance: δΦ = ∇⁻²(_lin_charney_rhs). This couples the vertical derivative ∂²δΦ/∂π² fully with the horizontal mean-state gradients.'
text = re.sub(doc_pattern, doc_repl, text, flags=re.DOTALL)

# Fix top-level module docstring:
top_doc_pattern = r'"""sh_ppvi.invert_piece — piecewise PV inversion via direct spectral solve.*?Public API\n----------'
top_doc_repl = r'"""sh_ppvi.invert_piece — piecewise PV inversion via LGMRES with Linearized Charney.\n\nGiven the mean state (ψ̄, Φ̄) and a PV anomaly q′ supported on one piece,\nsolves for the anomaly streamfunction δψ.\n\nMethod — Linearized Charney LGMRES\n--------------------------------\nUses the linearised ertel_q_sh formula evaluated with Charney balance:\n\n    δΦ = ∇⁻²( _lin_charney_rhs(δψ, ψ̄) )\n    q_pred = coeff · (A_avo · ∂²δΦ/∂π² + B_stb · ∇²δψ)\n\nLGMRES finds δψ that minimizes ‖q_pred - q′‖. Preconditioned by level-mean spectral solve.\n\nPublic API\n----------'
text = re.sub(top_doc_pattern, top_doc_repl, text, flags=re.DOTALL)

with open("sh/sh_ppvi/invert_piece.py", "w") as f:
    f.write(text)

