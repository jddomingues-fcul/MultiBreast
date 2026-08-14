from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np


def tolerance_points(
    diffs: np.ndarray, k_max: int = 6
) -> tuple[np.ndarray, np.ndarray]:
    abs_err = np.abs(diffs)
    ks = np.arange(0, k_max + 1)
    ys = np.array([(abs_err <= k).mean() if diffs.size > 0 else 0.0 for k in ks])
    return ks, ys


def plot_tolerance_curve(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    groups: Sequence[str] | None = None,
    filename: str | None = None,
    title: str | None = "Cumulative Tolerance Curve (|pred - true| ≤ k)",
    k_max: int = 6,
    figsize=(6.4, 4.8),
    dpi=300,
):

    diffs = [pred - true for pred, true in zip(y_pred, y_true)]

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111)
    if groups is not None:
        agg_groups = {}
        for gr, df in zip(groups, diffs):
            agg_groups.setdefault(gr, []).append(df)
        for g, ds in agg_groups.items():
            diffs = np.asarray(ds, dtype=int)
            ks, ys = tolerance_points(diffs, k_max=k_max)
            label = str(g) if g is not None else "None"
            ax.plot(ks, ys, marker="o", label=label)
        ax.legend(title="Group")
    else:
        diffs = np.asarray(diffs, dtype=int)
        ks, ys = tolerance_points(diffs, k_max=k_max)
        ax.plot(ks, ys, marker="o")

    ax.set_xlabel("Tolerance k")
    ax.set_ylabel("Fraction with |error| ≤ k")
    ax.set_xticks(np.arange(0, k_max + 1))
    ax.set_ylim(0, 1.1)
    ax.hlines(
        y=1.0, xmin=0, xmax=k_max, colors="gray", linestyles="dashed", linewidth=0.8
    )
    if title:
        ax.set_title(title)
    fig.tight_layout()
    if filename:
        fig.savefig(f"{filename}.png", bbox_inches="tight")
    return fig
