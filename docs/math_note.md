# The equations as this package solves them

Davis & Emanuel (1991) posed the inversion on a limited-area window in Cartesian-like
coordinates. Moving it to the whole sphere changes three things that are easy to get wrong and one
that is easy to miss entirely. This note derives all four, because each was got wrong at least once
while building the package and the tests that caught them only make sense against the derivation.

Reference: Davis, C. A. and K. A. Emanuel, 1991: Potential vorticity diagnostics of cyclogenesis.
*Monthly Weather Review*, **119**, 1929–1953.

## 1. The vertical coordinate and the boundary temperature

The Exner function Π = c_p (p/p₀)^κ, κ = R_d/c_p = 2/7, with hydrostatic balance

    ∂Φ/∂Π = −θ.

Π *decreases* upward, so with Φ increasing upward θ is positive, and every difference stencil
built on Π carries that sign. Interior potential vorticity lives on levels 1 … N−2; the outermost
two levels carry boundary potential temperature instead, and enter as hydrostatic ghosts

    Φ₀ = Φ₁ + θ_B (Π₁ − Π₀),      Φ_{N−1} = Φ_{N−2} − θ_T (Π_{N−1} − Π_{N−2}).

Substituting these into the end rows of the second-difference stencil folds one coupling weight
onto the diagonal and moves the temperature to the right-hand side divided by `dpi2`, using the
identity `bl₁ (Π₁ − Π₀) = 1/dpi2₁` (asserted in `tests/test_levels.py`). Keeping the folding and
the source apart is what lets a piece be driven by one boundary temperature alone.

The same ghost rule applies inside the *first* differences of the cross terms. That is the easy one
to miss: a centred difference at the first interior level reaches the ghost too, so the boundary
temperature enters the cross terms as well as the static stability.

## 2. The potential vorticity, and the one unit conversion

The hydrostatic Ertel potential vorticity is

    q_SI = −g (ζ + f) ∂θ/∂p − g (∂u/∂p ∂θ/∂y − ∂v/∂p ∂θ/∂x).

With dΠ/dp = κΠ/p and θ = −∂Φ/∂Π this becomes

    q_SI = g κ (Π/p) [ (f + ∇²ψ) Φ_ΠΠ − ∇_h(ψ_Π)·∇_h(Φ_Π) ],

so the bracket — what the inversion actually solves for — is

    q̂ = q_SI · p / (g κ Π) = q_SI · (p/p₀)^{1−κ} · p₀/(κ g c_p).

That single factor is the whole of the Wu non-dimensionalisation once DPI, FF, THO and LL have
cancelled; everything inside the solver is SI. With κ = 2/7 the exponent 1−κ equals 5κ/2 — the same
coincidence that makes the Wu-to-SI conversion a flat factor of 1e-8 at every level. The physical
form is the one to keep, since only it survives a change of κ.

## 3. The balance equation is not the Cartesian one

Take the divergence of the vector-invariant momentum equation for a nondivergent wind
V = k̂ × ∇ψ and set the divergence tendency to zero:

    ∂V/∂t = −(ζ + f) k̂ × V − ∇(Φ + |V|²/2),   k̂ × V = −∇ψ,

    ⇒  ∇²Φ = ∇·[(f + ζ) ∇ψ] − ½ ∇²|∇ψ|².        (★)

This is exact and holds on any surface. It is **not** the textbook

    ∇²Φ = ∇·(f ∇ψ) + 2(ψ_xx ψ_yy − ψ_xy²).       (plane)

On a curved surface the Laplacian does not commute with the gradient. The Bochner identity gives
∇²(∇ψ) = ∇(∇²ψ) + K∇ψ with K = 1/a², and expanding both sides of (★):

    ∇·(ζ∇ψ) = ∇ζ·∇ψ + ζ²,
    ½∇²|∇ψ|² = ∇ψ·∇(∇²ψ) + ‖Hess ψ‖² + K|∇ψ|²,
    ⇒ ∇·(ζ∇ψ) − ½∇²|∇ψ|² = ζ² − ‖Hess ψ‖² − K|∇ψ|² = 2 det(Hess ψ) − |∇ψ|²/a².

So the Cartesian form is the plane approximation of (★), short by |V|²/a² — for a 30 m/s flow about
2 % of the balance terms, systematic and one-signed. This package solves (★).

**Closed-form check.** For solid-body rotation ψ = C sinφ: ζ = −2C sinφ/a², the deformation
vanishes, so 2 det(Hess) = ζ²/2, and |∇ψ|² = C² cos²φ/a². Then

    N(ψ) = 2 det(Hess) − |∇ψ|²/a² = C²(3 sin²φ − 1)/a⁴,

which the code reproduces to 1e-9 (`tests/test_sphere.py::test_solid_body_rotation_against_closed_form`).
Evaluating the same expression directly from (★) by hand gives the identical result, which is what
pinned the sign of the curvature term.

Because (★) is built from gradients, a divergence and a Laplacian — all spectral — no metric factor
is written out anywhere, and none can be dropped. The rotation-invariance test (the same geodesic
vortex at 45°, 60°, 80°, 89° and 90° N gives the same cap integral to seven figures) is what
demonstrates that.

## 4. Metric terms, for the diagnostics that do need them

From the strain-rate tensor in orthogonal curvilinear coordinates with scale factors a cosφ and a:

    div = u_x + v_y − v tanφ/a          ζ  = v_x − u_y + u tanφ/a
    D1  = u_x − v_y − v tanφ/a          D2 = v_x + u_y + u tanφ/a

with subscripts denoting the eastward and northward derivative components. Solid-body rotation is
the discriminating case: it has D1 = D2 = 0, which fails outright if either tanφ term is dropped or
carries the wrong sign — both of which happened while writing this.

These appear only in the deformation diagnostic. The production path never touches them.

## 5. The perturbation system, and why the pieces add

Freeze the coefficients at ψ̃ = ψ̄ + ½ψ′ and Φ̃ = Φ̄ + ½Φ′. Both nonlinearities are quadratic or
bilinear, so for a quadratic N and a bilinear B

    N(x̄ + x′) − N(x̄) = DN(x̄ + ½x′)[x′],
    B(x̄+x′, ȳ+y′) − B(x̄, ȳ) = B(x′, ȳ+½y′) + B(x̄+½x′, y′),

both exactly. The midpoint linearisation is therefore not an approximation, and pieces sharing the
frozen operator sum to the all-sources solution rather than approximating it. This is Davis's
`INLIN=1` convention; it is the whole reason a piecewise inversion means anything.

The two equations on the interior levels are

    E1:  ∇²Φ′ − { ∇·[(f* + ζ̃)∇ψ′] + ∇·(ζ′∇ψ̃) − ∇²(∇ψ̃·∇ψ′) } = 0,
    E2:  AVO·Φ′_ΠΠ + STB·∇²ψ′ − ∇_h(ψ̃_Π)·∇_h(Φ′_Π) − ∇_h(Φ̃_Π)·∇_h(ψ′_Π) = q̂′,

with AVO = f* + ∇²ψ̃ and STB = Φ̃_ΠΠ, both floored to keep the system elliptic.

**What is absent on purpose.** The Fortran's ASI, BSI, APHI and BI contain the discrete five-point
diagonal AC(I,3), because the elimination of one field into the other was performed at the discrete
level as an iteration accelerator. Substituting the converged discrete balance equation back into
the ψ-step residual collapses every AC3-containing term and returns exactly E1 and E2. They are
therefore not physics and have no continuum counterpart; the coupled Krylov solve supplies the
simultaneous coupling they were approximating.

## 6. The hemisphere, and why the mirror runs twice

The data are northern-hemisphere only; the operator is global. Two requirements: the inversion must
stay elliptic, and the invented hemisphere must not leak into the answer.

Use f* = 2Ω√(sin²φ + sin²φ₀), even in latitude and smooth at the equator. Then if every coefficient
and every source is even, the operator commutes with φ → −φ, the solution is exactly even, and its
northern half solves the northern problem under a homogeneous Neumann equator. The southern half is
determined by the northern one and carries no independent information.

Getting there takes two mirrors with different parities:

1. Mirror the **wind as a vector** — eastward even, northward odd — and take its vorticity. Only
   this parity gives a mirrored field whose vorticity agrees with the hemisphere's own.
2. Mirror **that vorticity as an even scalar**, and derive the streamfunction, wind, and potential
   vorticity from it.

Stopping after step 1 leaves ζ odd, so AVO = |f| + ζ is not even, ellipticity is lost in the south,
and the scaffold contaminates the north at order ζ/f. The wind implied by step 2 is eastward-odd and
northward-even — the mirror image of a circulation with the *same* sense of rotation, which is what
|f| describes.

The remaining defect is that an even mirror is only continuous where the mean zonal wind does not
vanish at the equator, leaving a vortex sheet in ζ̃. The coefficients — never the sources, never the
solution — are tapered to their smooth equatorial limits across a band, which keeps the operator
shared between pieces and so leaves the exact closure untouched. The taper is a real modification of
the system and is off by default for states that are already smooth across the equator: measured on
a global synthetic case, the untapered solve is exact to rounding while the tapered one leaves about
half a metre of equivalent geopotential height.

## 7. Gauge

The operator has an exact null space: a per-level constant added to ψ′ (which carries no wind, and
which every term annihilates), and a single global constant added to Φ′ (annihilated because the
vertical stencil's rows sum to zero). The balance equation is a sum of divergences, so its n = 0
component vanishes identically for any input and carries no information — that slot is reused to pin
each level's mean streamfunction, which is precisely the direction it corresponds to. A rank-one
term on the potential-vorticity row removes the remaining constant.

What the Fortran's Dirichlet walls did implicitly — pinning those constants, and absorbing any
global imbalance between the area-mean potential vorticity and the boundary temperature through
boundary fluxes — the sphere must now do explicitly. The gauges above handle the first; the second
appears as a compatibility condition on the n = 0 column, whose removed magnitude is reported rather
than discarded silently.

## 8. Ellipticity of the balance row, and the taper as a weight on products

Two things about the linearised balance row were found on real events and are now built in.

**The row is elliptic only where vorticity beats deformation.** The quadratic part of the balance
equation, polarised over two streamfunctions, is (from §3, using 4 det(Hess) = ζ² − D₁² − D₂²)

    B(a, b) = ∇·(ζ_a ∇b) + ∇·(ζ_b ∇a) − ∇²(∇a·∇b) = ζ_a ζ_b − (D₁ᵃD₁ᵇ + D₂ᵃD₂ᵇ) − 2∇a·∇b/a²,

so the symbol of the tangent at a reference ψ̃ has eigenvalues f + ζ̃ ∓ D̃. Wherever the reference
deformation exceeds the absolute vorticity — strain-dominated flow, about a sixth of the jet level
on a winter day — the linearised system is indefinite there. Krylov solvers often still converge on
it, but a Newton iteration whose path crosses a fold of such a system oscillates without descending,
which is how a handful of events out of fifty used to end at the step cap. The limited-area codes
meet a related condition with a clamp on their balance-equation coefficient.

Here the deformation part alone is scaled. With ζ̃ζ′ evaluated pointwise and the rest as the exact
divergence form,

    T[ψ′] = w ζ̃ ζ′ + w s [ B(ψ̃, ψ′) − ζ̃ ζ′ ],      s = min(1, (1 − m) AVO / (w D̃)),

the symbol becomes AVO ∓ s w D̃ ≥ m·AVO, the vorticity part that gradient-wind balance lives on is
untouched, and the form stays symmetric because s and w are frozen fields multiplying a symmetric
product. Scaling the whole quadratic part instead, ζ̃ζ′ included, removes the relative vorticity
from the effective Coriolis parameter in every strain region and weakened the balanced total flow
by a tenth on a test event; it is not what is done.

**The limiter is a safety net, brought in only on failure.** With the taper made symmetric (below),
the source evaluated with the operator's own stencils (§2) and the gauge constant removed before the
first step, the fifty test events — including every one that used to stall — converge in four Newton
steps with the balance equation solved exactly as posed. The limiter therefore starts as one
everywhere and is switched on, from the observed state and the current iterate, only when an inner
solve fails to converge or a line search fails: the two signs that the iteration has met a fold. On
the fifty events it was needed by two. While it is fixed, the residual is exactly quadratic and the
Jacobian its derivative, so Newton stays quadratic on the regularised system; every refresh is
counted and reported, and so is the area fraction where the returned state's linearised balance row
is still not elliptic. The piecewise pass takes the same limiter, so its operator is the tangent of
the same system at the midpoint. Where the limiter acts, the solved balance equation is the posed
one with a fraction 1 − s of its deformation and curvature part removed; the balance residual
reported against the posed equations shows exactly that.

**The taper multiplies products, not the reference.** The earlier construction tapered the
reference vorticity and rebuilt the reference streamfunction from it, so the row used two different
flows and its bilinear form was not symmetric: the midpoint identity of §5 failed by a factor of up
to two per Newton step in the band and the pieces no longer summed to the balanced perturbation.
With the weight inside every product — w in ∇·(wζ_a∇b), in ∇²(w∇a·∇b), and, for the static
stability tapered towards its level mean, the partner term (1 − w) ζ̃ ⟨S′⟩ on the potential-vorticity
row — the tapered system is again an exact quadratic and the identity holds to rounding with the
taper on (`tests/test_ellipticity_taper.py`).

Two smaller consistency points came with this. The reference state's boundary levels are set to the
operator's ghosts before the operator is frozen, since a reference that mixes observed boundary
levels with a balanced state's ghosts is the tangent of neither. And the boundary temperatures pass
through the spectrum before entering the right-hand side, so both sides of the bilinear form see the
same resolved temperature.
