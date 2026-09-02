"""Composite the fifty events: windowed chain against global inversion.

Both rows answer the same question -- which source induced which part of the
anomalous flow -- and differ in how the elliptic problem was closed.  The windowed
chain solves on a fixed 10.5 to 85.5 N band with lateral walls at ibc=1, so part
of its answer is the response to those walls and is carried as a wall piece.  The
global solve has no boundary, so it has no wall: the decomposition is the sources
themselves.  The wall is therefore merged with the residual in the figure, both
being the part not attributed to a potential-vorticity source.

Three things are done differently from the windowed-versus-nested comparison this
reproduces, all because the two rows now come from different codes:

*Only the three-dimensional keys are read.*  The two column averages do not share
a convention -- the windowed pipeline divides by the full weight sum, so a row
whose pieces are all missing comes out as an exact zero rather than as missing --
and a difference of that kind would appear as a stripe at the band edge and read
as physics.  The average is recomputed here, once, for both.

*One weight set.*  The event's own geopotential height, taken from the production
file, weights both rows.  It belongs to the event, not to either solver.

*One denominator.*  The two codes reconstruct the observed anomaly slightly
differently -- measured over these fifty events, 3.0 m/s apart on a 20.1 m/s
signal, correlation 0.99 -- so shares against each code's own anomaly would
differ by that much for reasons belonging to neither decomposition.  The production
anomaly is used for both, and absolute amplitudes are reported alongside.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

EXP = Path(__file__).resolve().parent
PROD = Path(
    "/net/flood/data2/users/x_yan/pvtend/outputs/cesm6hourly_blocking/peak/dh=+0"
)
MINE = EXP / "npz" / "peak" / "dh=+0"
PIECES = ["surface", "lower", "upper_p", "upper_e"]
WAVG_LEVELS = [400, 300, 250, 200]
H_SCALE = 7000.0
SHAPE = (65, 97)


def column_average(field3d: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted average over the upper-tropospheric levels.

    Divided by the weight that was actually valid at each point.  Dividing by the
    full sum instead fills a column that has no data at all with an exact zero,
    which is indistinguishable from a real cancellation.
    """
    valid = np.isfinite(field3d)
    weighted = np.where(valid, field3d * weights, 0.0).sum(axis=0)
    norm = np.where(valid, weights, 0.0).sum(axis=0)
    out = np.full(field3d.shape[1:], np.nan)
    good = norm > 0
    out[good] = weighted[good] / norm[good]
    return out


def load(
    path: Path, terms: list[str], weights: np.ndarray, index: list[int]
) -> dict:
    """Column-averaged wind for each term, from the three-dimensional keys.

    ``index`` selects the levels the average runs over, so the field and the
    weights are cut to the same levels in the same order.
    """
    out = {}
    with np.load(path) as z:
        for term in terms:
            stem = (
                "rot_anom_residual_ppvi"
                if term == "residual"
                else ("rot_anom" if term == "observed" else f"rot_anom_ppvi_{term}")
            )
            out[term] = tuple(
                column_average(np.asarray(z[f"{c}_{stem}_3d"], float)[index], weights)
                for c in ("u", "v")
            )
    return out


def weights_from(path: Path) -> tuple[np.ndarray, list[int]]:
    """Density weights for the column average, and the levels they belong to."""
    with np.load(path) as z:
        levels = [int(v) for v in z["levels"]]
        height = np.asarray(z["z_3d"], float)
    index = [levels.index(p) for p in WAVG_LEVELS]
    return np.exp(-height[index] / H_SCALE), index


def _assert_levels_agree(events) -> None:
    """The two stores must list the levels in the same order.

    They are combined by position, so a different order would mix levels while
    leaving every field looking entirely reasonable.
    """
    row = events.iloc[0]
    name = f"track_{row.track_id}_{row.ts.strftime('%Y%m%d%H')}_dh+0.npz"
    with np.load(PROD / name) as a, np.load(MINE / name) as b:
        old = [int(v) for v in a["levels"]]
        new = [int(v) for v in b["meta_level_pressures"]]
    if old != new:
        raise ValueError(f"level order differs: production {old}, global {new}")


def main() -> int:
    if "--plot-only" in sys.argv:
        stored = dict(np.load(EXP / "composites.npz"))
        metrics = json.loads((EXP / "composite_metrics.json").read_text())
        _plot(stored, metrics, EXP / "fig1_composite_flow.png")
        return 0
    events = pd.read_csv(EXP / "events_50.csv")
    events["ts"] = pd.to_datetime(events.base_ts)
    _assert_levels_agree(events)
    windowed_terms = PIECES + ["wall", "residual", "observed"]
    global_terms = PIECES + ["residual", "observed"]

    sums = {
        "windowed": {t: [np.zeros(SHAPE), np.zeros(SHAPE)] for t in windowed_terms},
        "global": {t: [np.zeros(SHAPE), np.zeros(SHAPE)] for t in global_terms},
    }
    count = np.zeros(SHAPE)
    used, skipped = 0, []

    for _, row in events.iterrows():
        name = f"track_{row.track_id}_{row.ts.strftime('%Y%m%d%H')}_dh+0.npz"
        try:
            weights, index = weights_from(PROD / name)
            windowed = load(PROD / name, windowed_terms, weights, index)
            # The same weights and the same levels: both files store the nine Wu
            # levels bottom-up, which is asserted before the loop rather than
            # assumed here.
            mine = load(MINE / name, global_terms, weights, index)
        except (FileNotFoundError, KeyError) as error:
            skipped.append(f"{row.track_id}: {error}")
            continue

        mask = np.ones(SHAPE, bool)
        for scheme in (windowed, mine):
            for u, v in scheme.values():
                mask &= np.isfinite(u) & np.isfinite(v)
        for label, scheme in (("windowed", windowed), ("global", mine)):
            for term, (u, v) in scheme.items():
                sums[label][term][0] += np.where(mask, u, 0.0)
                sums[label][term][1] += np.where(mask, v, 0.0)
        count += mask
        used += 1

    if skipped:
        print(f"skipped {len(skipped)}: {skipped[:4]}")
    valid = count > 0
    comp = {}
    for label in sums:
        for term in sums[label]:
            for component, values in zip("uv", sums[label][term]):
                comp[f"{label}_{term}_{component}"] = np.where(
                    valid, values / np.maximum(count, 1), np.nan
                )
    for component in "uv":
        comp[f"windowed_unattributed_{component}"] = (
            comp[f"windowed_wall_{component}"] + comp[f"windowed_residual_{component}"]
        )
        comp[f"global_unattributed_{component}"] = comp[f"global_residual_{component}"]
    np.savez(EXP / "composites.npz", count=count, n_used=used, **comp)

    trim = np.s_[2:-2, 2:-2]

    def rms(label, term):
        u = comp[f"{label}_{term}_u"][trim]
        v = comp[f"{label}_{term}_v"][trim]
        return float(np.sqrt(np.nanmean(u**2 + v**2)))

    reference = rms("windowed", "observed")
    metrics = {
        "n_events": used,
        "reference_anomaly_rms": reference,
        "global_own_anomaly_rms": rms("global", "observed"),
        "windowed": {t: rms("windowed", t) for t in windowed_terms + ["unattributed"]},
        "global": {t: rms("global", t) for t in global_terms + ["unattributed"]},
    }
    metrics["windowed_share"] = {
        t: metrics["windowed"][t] / reference for t in metrics["windowed"]
    }
    metrics["global_share"] = {
        t: metrics["global"][t] / reference for t in metrics["global"]
    }
    (EXP / "composite_metrics.json").write_text(json.dumps(metrics, indent=1))

    print(f"\ncomposite of {used} events; reference anomaly {reference:.2f} m/s "
          f"(the global reconstruction of it: {metrics['global_own_anomaly_rms']:.2f})")
    print(f"\n{'term':14s} {'windowed':>10s} {'global':>10s} {'diff':>10s}   (m/s)")
    for term in PIECES + ["unattributed"]:
        w = metrics["windowed"][term]
        g = metrics["global"][term]
        print(f"{term:14s} {w:10.3f} {g:10.3f} {g - w:+10.3f}")
    print(f"{'  of which wall':14s} {metrics['windowed']['wall']:10.3f} "
          f"{0.0:10.3f} {-metrics['windowed']['wall']:+10.3f}")
    _plot(comp, metrics, EXP / "fig1_composite_flow.png")
    return 0


def _plot(comp, metrics, path) -> None:
    """Three rows, five columns, with the two methods sharing a scale.

    Within a column the two method rows are drawn on one colour range, because the
    figure exists to be compared down the column and two independently scaled
    panels cannot be. The range is set from the two of them jointly at a high
    percentile rather than at the maximum: the windowed field runs away near its
    walls, and letting that set the range would flatten everything the figure is
    about. The difference row keeps its own range, being a different quantity.
    """
    columns = PIECES + ["unattributed"]
    titles = {
        "surface": "surface theta",
        "lower": "lower PV (850-500)",
        "upper_p": "upper PV, planetary",
        "upper_e": "upper PV, eddy",
        "unattributed": "not attributed\n(wall + residual)",
    }
    rows = [
        ("windowed", "windowed chain, ibc=1 + wall"),
        ("global", "global inversion, no wall"),
        ("difference", "global - windowed"),
    ]
    fig, axes = plt.subplots(3, 5, figsize=(19, 10), constrained_layout=True)
    yy, xx = np.mgrid[0 : SHAPE[0], 0 : SHAPE[1]]
    step = 4
    limits = {}
    for term in columns:
        both = np.concatenate(
            [
                np.hypot(comp[f"{lab}_{term}_u"], comp[f"{lab}_{term}_v"]).ravel()
                for lab in ("windowed", "global")
            ]
        )
        limits[term] = float(np.nanpercentile(both, 98))
        limits[("difference", term)] = float(
            np.nanpercentile(
                np.hypot(
                    comp[f"global_{term}_u"] - comp[f"windowed_{term}_u"],
                    comp[f"global_{term}_v"] - comp[f"windowed_{term}_v"],
                ),
                98,
            )
        )
    for r, (label, row_title) in enumerate(rows):
        for c, term in enumerate(columns):
            axis = axes[r, c]
            if label == "difference":
                u = comp[f"global_{term}_u"] - comp[f"windowed_{term}_u"]
                v = comp[f"global_{term}_v"] - comp[f"windowed_{term}_v"]
                vmax = limits[("difference", term)]
            else:
                u = comp[f"{label}_{term}_u"]
                v = comp[f"{label}_{term}_v"]
                vmax = limits[term]
            speed = np.hypot(u, v)
            image = axis.pcolormesh(
                speed, cmap="magma_r", vmin=0.0, vmax=vmax, shading="auto"
            )
            axis.quiver(
                xx[::step, ::step],
                yy[::step, ::step],
                u[::step, ::step],
                v[::step, ::step],
                scale=120,
                width=0.005,
                color="0.15",
            )
            axis.plot(SHAPE[1] // 2, SHAPE[0] // 2, "c+", markersize=10, mew=2)
            axis.set_xticks([0, 24, 48, 72, 96])
            axis.set_xticklabels(["-60", "-30", "0", "30", "60"])
            axis.set_yticks([0, 16, 32, 48, 64])
            axis.set_yticklabels(["-30", "-15", "0", "15", "30"])
            if r == 2:
                axis.set_xlabel("longitude from centre [deg]")
            if c == 0:
                axis.set_ylabel(f"{row_title}\nlatitude from centre [deg]")
            if r == 0:
                axis.set_title(titles[term])
            above = float(np.nanmean(speed > vmax))
            if above > 0.02:
                axis.text(
                    0.02, 0.02, f"{above:.0%} above scale",
                    transform=axis.transAxes, fontsize=7, color="0.25",
                )
            fig.colorbar(image, ax=axis, label="|v| [m s$^{-1}$]")
    fig.suptitle(
        f"Induced flow by source, {metrics['n_events']} blocking peaks, "
        f"400-200 hPa: a window with walls against the whole sphere",
        fontsize=14,
    )
    fig.savefig(path, dpi=140)
    print(f"wrote {path}")


if __name__ == "__main__":
    raise SystemExit(main())
