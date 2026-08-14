from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np


def plot_error_distance_violin(
    true_labels: Sequence[int],
    diffs: Sequence[int],
    classes_order: Sequence[int] | None = None,
    out_prefix: str | None = None,
    title: str | None = "BI-RADS Prediction Error Distribution by True Class",
    ylim: tuple[float, float] | None = None,
    stagger_labels: bool = True,
    figsize=(8, 4.8),
    dpi=300,
):
    y_true = np.asarray(true_labels, dtype=int)
    d = np.asarray(diffs, dtype=int)
    assert y_true.shape == d.shape

    classes = (
        np.unique(y_true)
        if classes_order is None
        else np.asarray(classes_order, dtype=int)
    )
    data = [d[y_true == c] for c in classes]

    # Determine y-limits
    if ylim is None:
        all_vals = (
            np.concatenate([x for x in data if x.size > 0])
            if any(len(x) > 0 for x in data)
            else np.array([0])
        )
        y_min, y_max = all_vals.min(), all_vals.max()
        if y_min == y_max:
            y_min, y_max = y_min - 1, y_max + 1
        span = max(abs(y_min), abs(y_max))
        y_lo, y_hi = -span - 0.5, span + 0.5
    else:
        y_lo, y_hi = ylim

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111)

    ax.violinplot(
        data,
        positions=np.arange(1, len(classes) + 1),
        showmeans=True,
        showmedians=False,
        showextrema=False,
    )
    ax.axhline(0, linestyle="--", linewidth=1)

    ax.set_xlabel("True BI-RADS class")
    ax.set_ylabel("Error (pred - true)")
    ax.set_xticks(np.arange(1, len(classes) + 1))
    ax.set_xticklabels([str(c) for c in classes])
    ax.set_ylim(y_lo, y_hi + 0.3)
    if title:
        ax.set_title(title)

    rng = y_hi - y_lo
    for i, (c, vals) in enumerate(zip(classes, data), start=1):
        n = int(vals.size)
        if n == 0:
            txt = "n=0"
            y_txt = y_hi - 0.05 * rng
        else:
            within1 = float(np.mean(np.abs(vals) <= 1))
            pct = int(round(100 * within1))
            txt = f"{pct}% ≤±1  (n={n})"
            local_max = vals.max()
            y_txt = max(local_max + 0.06 * rng, y_hi - 0.05 * rng)
        if stagger_labels and (i % 2 == 0):
            y_txt -= 0.05 * rng  # stagger every other label
        ax.text(i, y_txt, txt, ha="center", va="bottom")

    ax.set_xlim(-1.0, len(classes) + 1.0)
    fig.tight_layout()

    if out_prefix:
        fig.savefig(f"{out_prefix}.png", bbox_inches="tight")
    return fig
