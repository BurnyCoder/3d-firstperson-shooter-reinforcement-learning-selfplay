"""Local: shared four-panel PNG/SVG rendering; global: loss, reward and activity plots use one transparent implementation.

Original glue uses https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.plot.html
and https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.savefig.html .
"""

import numpy as np  # Local: preserve missing observations as line gaps.

COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9")  # Local: stable contrasting run identities.


def metric_series(rows, key):
    """Local: preserve genuine zero and missingness; global: missing metrics are never successful zero error."""
    values = np.asarray([row.get(key, np.nan) for row in rows], dtype=float)
    return np.where(np.isfinite(values), values, np.nan)  # Local: infinities/NaNs are gaps, never clipped into finite values.


def plot_curves(records, destination, title, metrics, footer):
    """Local: export four separately scaled diagnostics; global: never average different runs or imply model uncertainty."""
    import matplotlib  # Local: reporting alone imports the graphics backend.
    matplotlib.use("Agg")  # Local: CPU artifact export never opens the game or consumes GPU memory.
    from matplotlib import pyplot as plt

    with plt.rc_context({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False}):
        figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
        for axis, (key, caption, unit, limits) in zip(axes.flat, metrics, strict=True):
            for index, record in enumerate(records):
                values = metric_series(record["rows"], key)  # Local: current metric's missingness cannot hide other recorded panels.
                if np.isfinite(values).any():
                    x = [row["frames"] / 1e6 for row in record["rows"]]  # Local: compare equal environment exposure.
                    axis.plot(x, values, color=COLORS[index % len(COLORS)], linewidth=1.15,
                              label=record["receipt"]["label"])  # Local: show every interval without smoothing or error bands.
            axis.set(title=caption, xlabel="Arena steps (millions)", ylabel=unit, xlim=(0, None))
            axis.title.set_fontweight("bold")  # Local: distinguish panel names from numerical scale labels.
            axis.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))  # Local: tiny coefficients remain readable.
            axis.grid(alpha=.2)  # Local: a light grid supports comparisons without obscuring raw data.
            if limits is not None:
                axis.set_ylim(*limits)  # Local: physical fractions use the full zero-to-one range.
            if axis.lines:
                axis.legend(fontsize=9, frameon=False)
            else:
                axis.text(.5, .5, "Not recorded", transform=axis.transAxes, ha="center", color="#666666")
        figure.suptitle(title, fontsize=16, fontweight="bold", x=.075, ha="left")
        figure.text(.075, .025, footer, fontsize=9)  # Local: interpretation warnings travel with exported figures.
        figure.subplots_adjust(left=.10, right=.98, bottom=.12, top=.89, hspace=.40, wspace=.28)
        for suffix in ("png", "svg"):
            figure.savefig(destination.with_suffix(f".{suffix}"), dpi=160, facecolor="white")  # Local: preview plus scalable publication artifact.
        plt.close(figure)  # Local: release figure memory before processing another run.
