"""Orchestrate the Davis/Wu inversion chain on a prepared event directory.

A prepared event directory (see prep_event.py) holds inner/ and outer/
subdirectories of fixed-format .grid files.  This module drives the three
Fortran passes inside one domain directory, always with cwd set to that
directory and bare relative filenames (the qinvertp IGFIL slot is
CHARACTER*30, see NOTES.md section 6):

  pass A/B  pvpialln  : 20 half-day .grid -> meanq, meanh, evq.out, evh.out
  pass C    qinvert   : evh.out + evq.out -> bal.out (total balanced fields)
  pass D    qinvertp  : one piecewise perturbation inversion per call

Pass-D piece indices (9-level build): 1 = bottom boundary theta (1000 hPa),
2..8 = interior PV at 850,700,500,400,300,250,200 hPa, 9 = top boundary
theta (100 hPa).

The zero-source ("wall") solve passes the mean q file as the total q file,
so q' and theta' vanish identically while FNM(4)=bal.out keeps the full
perturbation on the lateral boundary (IBC=1) - the solution is then the
pure boundary-forced harmonic response W.

All file layouts and scalings follow NOTES.md: q.out (THB,THT,Q[7]),
h.out (H,psi/1e5,U,V,TH), meanh (Hm,psim/1e5), bal.out (HZ,SI/1e5),
pert.out (HP in m, SP in 1e5 m2/s).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from talia_io import read_out_blocks

NL = 9
PLEV = [1000, 850, 700, 500, 400, 300, 250, 200, 100]
ALL_LEVELS = ",".join(str(k) for k in range(1, NL + 1))

_FAIL_MARKERS = ("TOO MANY", "Input data file headers are inconsistent")


def _run(binary: Path, deck_text: str, cwd: Path, tag: str,
         success_marker: str | None) -> str:
    deck = cwd / f"deck_{tag}.in"
    log = cwd / f"log_{tag}.txt"
    deck.write_text(deck_text)
    with deck.open() as fin, log.open("w") as fout:
        subprocess.run([str(binary)], stdin=fin, stdout=fout,
                       stderr=subprocess.STDOUT, cwd=cwd, check=True)
    text = log.read_text()
    for marker in _FAIL_MARKERS:
        if marker in text:
            raise RuntimeError(f"{tag}: solver reported '{marker}' ({log})")
    if success_marker and success_marker not in text:
        raise RuntimeError(f"{tag}: missing '{success_marker}' ({log})")
    return text


def run_passab(domain_dir: Path, binary: Path, event_index: int) -> None:
    grids = sorted(domain_dir.glob("*.grid"))
    if not grids:
        raise FileNotFoundError(f"no .grid files in {domain_dir}")
    lines = ["meanq", "meanh", str(len(grids))]
    lines += [g.name for g in grids]
    lines += ["1", str(event_index), "evq.out", "evh.out"]
    _run(binary, "\n".join(lines) + "\n", domain_dir, "passab",
         success_marker="psi converged")


def run_passc(domain_dir: Path, binary: Path,
              maxit: int = 2000, omegas: float = 1.85, omegah: float = 1.75,
              prt: float = 0.5, thr: float = 0.1, qmin: float = 0.01) -> None:
    deck = "\n".join([
        str(maxit), str(maxit), str(omegas), str(omegah), str(prt), str(thr),
        "'evh.out'", "'evq.out'", "'bal.out'",
        "1",            # IMAP spherical
        str(qmin),
        "1",            # INF=1: init file holds H then psi
    ]) + "\n"
    _run(binary, deck, domain_dir, "passc", success_marker="TOTAL CONVERGENCE")


def run_passd(domain_dir: Path, binary: Path, tag: str, qlv: list[int],
              ibc: int, igfil: str | None = None,
              total_q: str = "evq.out",
              omegas: float = 1.85, omegah: float = 1.75,
              prt: float = 0.8, thrsh: float = 0.1) -> Path:
    """One piecewise inversion; returns the pert output path.

    qlv     : list of piece indices for this inversion (see module docstring)
    ibc     : 0 homogeneous, 1 full-perturbation walls, 2 walls from igfil
    igfil   : bare filename of the initial-guess/boundary file (ibc=2 only)
    total_q : 'evq.out' normally; 'meanq' for the zero-source wall solve
    """
    if total_q == "meanq":
        # gfortran refuses to open one file on two units (FNM(1) is already
        # meanq), so the zero-source solve reads a copy.
        copy = domain_dir / "meanq_w"
        copy.write_bytes((domain_dir / "meanq").read_bytes())
        total_q = "meanq_w"
    out = f"pert_{tag}.out"
    lines = [
        str(omegas), str(omegah), str(prt), str(thrsh), "1.", "1.",
        "'meanq'", "'meanh'", f"'{total_q}'", "'bal.out'", f"'{out}'",
        "1",            # IMAP
        "1",            # INLIN
        "0",            # IQD
        f"{NL},{ALL_LEVELS}",
        f"{NL},{ALL_LEVELS}",
        "1",            # NOUT
        f"{len(qlv)},{','.join(str(k) for k in qlv)}",
        str(ibc),
    ]
    if ibc == 2:
        if igfil is None:
            raise ValueError("ibc=2 requires igfil")
        if len(igfil) > 30:
            raise ValueError(f"IGFIL name exceeds CHARACTER*30: {igfil}")
        lines.append(f"'{igfil}'")
    _run(binary, "\n".join(lines) + "\n", domain_dir, tag,
         success_marker="TOTAL CONVERGENCE")
    return domain_dir / out


# ---------------------------------------------------------------- parsers

def read_pert(path: Path, ny: int, nx: int) -> dict[str, np.ndarray]:
    """HP (m) and SP (1e5 m2/s), each (NL, ny, nx)."""
    return read_out_blocks(path, [("HP", NL, ny, nx), ("SP", NL, ny, nx)])


def read_bal(path: Path, ny: int, nx: int) -> dict[str, np.ndarray]:
    return read_out_blocks(path, [("HZ", NL, ny, nx), ("SI", NL, ny, nx)])


def read_meanh(path: Path, ny: int, nx: int) -> dict[str, np.ndarray]:
    return read_out_blocks(path, [("Hm", NL, ny, nx), ("psim", NL, ny, nx)])


def full_pert(domain_dir: Path, ny: int, nx: int) -> dict[str, np.ndarray]:
    """Reference perturbation: balanced total minus time mean.

    Same units as pert.out: HP in m, SP in 1e5 m2/s.
    """
    bal = read_bal(domain_dir / "bal.out", ny, nx)
    mean = read_meanh(domain_dir / "meanh", ny, nx)
    return {"HP": bal["HZ"] - mean["Hm"], "SP": bal["SI"] - mean["psim"]}


def write_igfil(path: Path, header: str, hp: np.ndarray,
                sp: np.ndarray) -> None:
    """IGFIL for IBC=2: header, HP all NL levels (m), SP all NL levels
    (1e5 m2/s), free-format (space-separated) so negative values never
    touch."""
    with path.open("w") as fh:
        fh.write(header.rstrip("\n") + "\n")
        for arr in (hp, sp):
            if arr.shape[0] != NL:
                raise ValueError("IGFIL needs all NL levels")
            for k in range(NL):
                for row in arr[k]:
                    fh.write(" ".join(f"{v:.4f}" for v in row) + "\n")
