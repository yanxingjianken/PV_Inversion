# talia

The Davis and Emanuel piecewise potential-vorticity inversion as a limited-area
Fortran chain, ported to gfortran from the original inv3d sources.

Three programs, run in order:

| program | pass | what it does |
|---|---|---|
| `src/pvpialln_94UV.f` | A, B | Ertel potential vorticity from the state, and the streamfunction from the vorticity |
| `src/qinvert21_94.f` | C | the total inversion: the nonlinear balance system, by relaxation |
| `src/qinvertp21_94.f` | D | the perturbation inversion, piece by piece |

The lateral boundary condition of the perturbation inversion is the `IBC` field of
its deck, and all three of its settings work here:

| `IBC` | condition on the walls |
|---|---|
| 0 | zero |
| 1 | the full perturbation, less whatever earlier pieces in the same run have already accounted for |
| 2 | read from a file, so an outer solve can drive an inner one |

`NOTES.md` is the interface contract: deck fields, file layouts and scalings, the
`IGFIL` format for nesting, and the regression against the shipped outputs.

## Building

`make` generates the grid-size variants into `build/gen/` and compiles them.
Grid dimensions are compile-time `PARAMETER`s in every program *and* in the
`BALNC` and `BALP` subroutines, so a new grid needs a new binary.

## Running

`scripts/prep_event.py` cuts an event and its window out of model output and
writes the fixed-format `.grid` files; `scripts/run_chain.py` writes the decks and
runs the three passes in order.
