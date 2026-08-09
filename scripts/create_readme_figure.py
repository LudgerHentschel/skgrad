"""Create the two-panel ReLU-gradient figure used in the README.

Run from anywhere with::

    python scripts/create_readme_figure.py

The script writes SVG and PNG versions to ``docs/``. Matplotlib is required
only to regenerate the figure; it is not a runtime dependency of skgrad.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUTPUT_STEM = "relu-analytic-gradients"


def relu(values: np.ndarray) -> np.ndarray:
    return np.maximum(values, 0.0)


def network_values(
    x: np.ndarray,
    weights: np.ndarray,
    biases: np.ndarray,
    output_weights: np.ndarray,
    intercept: float,
) -> tuple[np.ndarray, np.ndarray]:
    hidden = relu(x[:, None] * weights + biases)
    contributions = hidden * output_weights
    return intercept + contributions.sum(axis=1), contributions


def network_gradients(
    x: np.ndarray,
    weights: np.ndarray,
    biases: np.ndarray,
    output_weights: np.ndarray,
) -> np.ndarray:
    active = x[:, None] * weights + biases > 0.0
    return (active * weights * output_weights).sum(axis=1)


def create_figure() -> plt.Figure:
    # Three fitted hidden units. Their activation switches occur at the
    # deliberately separated x coordinates below.
    thresholds = np.array([-1.4, 0.4, 1.5])
    weights = np.array([1.0, -1.0, 1.0])
    biases = -weights * thresholds
    output_weights = np.array([1.0, 0.65, -0.55])
    intercept = 0.30

    x = np.linspace(-3.0, 3.0, 1_500)
    grid = np.linspace(-2.75, 2.75, 12)
    values, contributions = network_values(
        x, weights, biases, output_weights, intercept
    )
    grid_values, _ = network_values(
        grid, weights, biases, output_weights, intercept
    )
    gradients = network_gradients(x, weights, biases, output_weights)
    grid_gradients = network_gradients(grid, weights, biases, output_weights)

    colors = {
        "network": "#175D8D",
        "gradient": "#D95F0E",
        "unit_1": "#5E3C99",
        "unit_2": "#1B9E77",
        "unit_3": "#7570B3",
        "grid": "#8793A1",
        "text": "#20262E",
        "frame": "#C8CFD8",
    }

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 12,
            "axes.labelsize": 10.5,
            "axes.edgecolor": colors["frame"],
            "axes.labelcolor": colors["text"],
            "xtick.color": colors["text"],
            "ytick.color": colors["text"],
            "text.color": colors["text"],
        }
    )

    figure, (value_axis, gradient_axis) = plt.subplots(
        2,
        1,
        figsize=(8.2, 6.5),
        sharex=True,
        gridspec_kw={"height_ratios": [1.45, 1.0], "hspace": 0.13},
    )
    figure.patch.set_facecolor("white")

    unit_colors = [colors["unit_1"], colors["unit_2"], colors["unit_3"]]
    unit_labels = ["hidden-unit contributions", "_nolegend_", "_nolegend_"]
    for index, (unit_color, label) in enumerate(zip(unit_colors, unit_labels)):
        value_axis.plot(
            x,
            contributions[:, index],
            color=unit_color,
            linewidth=1.35,
            linestyle=(0, (4, 3)),
            alpha=0.72,
            label=label,
        )

    value_axis.plot(
        x,
        values,
        color=colors["network"],
        linewidth=3.0,
        label=r"network output $f(x)$",
        zorder=4,
    )
    value_axis.scatter(
        grid,
        grid_values,
        s=34,
        color=colors["network"],
        edgecolor="white",
        linewidth=0.8,
        zorder=6,
        label="batched evaluation grid",
    )

    # Short tangent segments show the exact local slope at every grid point.
    tangent_half_width = 0.12
    for grid_x, grid_y, slope in zip(grid, grid_values, grid_gradients):
        tangent_x = np.array([grid_x - tangent_half_width, grid_x + tangent_half_width])
        tangent_y = grid_y + slope * (tangent_x - grid_x)
        value_axis.plot(
            tangent_x,
            tangent_y,
            color=colors["gradient"],
            linewidth=2.0,
            solid_capstyle="round",
            zorder=5,
        )

    gradient_axis.plot(
        x,
        gradients,
        color=colors["gradient"],
        linewidth=2.8,
        drawstyle="steps-post",
        label=r"analytic gradient $f'(x)$",
        zorder=4,
    )
    gradient_axis.scatter(
        grid,
        grid_gradients,
        s=36,
        color=colors["gradient"],
        edgecolor="white",
        linewidth=0.8,
        zorder=6,
        label=r"returned gradients $f'(x_i)$",
    )

    for axis in (value_axis, gradient_axis):
        axis.set_facecolor("white")
        axis.grid(axis="y", color=colors["frame"], linewidth=0.7, alpha=0.45)
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines[["left", "bottom"]].set_linewidth(0.8)
        for grid_x in grid:
            axis.axvline(grid_x, color=colors["grid"], linewidth=0.45, alpha=0.14)
        for threshold, unit_color in zip(thresholds, unit_colors):
            axis.axvline(
                threshold,
                color=unit_color,
                linewidth=1.1,
                linestyle=(0, (2, 3)),
                alpha=0.65,
            )

    value_axis.axhline(0.0, color=colors["frame"], linewidth=0.8)
    gradient_axis.axhline(0.0, color=colors["frame"], linewidth=0.8)
    value_axis.set_ylabel("Model output")
    gradient_axis.set_ylabel("Input gradient")
    gradient_axis.set_xlabel("Input feature  x")
    value_axis.set_xlim(-3.0, 3.0)

    value_axis.set_title(
        "1. A forward pass identifies the active ReLU units",
        loc="left",
        fontweight="semibold",
        pad=10,
    )
    gradient_axis.set_title(
        "2. A reverse pass combines their known slopes",
        loc="left",
        fontweight="semibold",
        pad=9,
    )

    value_axis.text(
        0.01,
        0.96,
        r"$f(x)=c+\sum_k v_k\,\mathrm{ReLU}(w_kx+b_k)$",
        transform=value_axis.transAxes,
        ha="left",
        va="top",
        fontsize=10.5,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 2.5},
    )
    gradient_axis.text(
        0.01,
        0.94,
        r"$f'(x)=\sum_{k:\,w_kx+b_k>0} v_kw_k$",
        transform=gradient_axis.transAxes,
        ha="left",
        va="top",
        fontsize=10.5,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 2.5},
    )

    value_axis.legend(
        loc="upper right",
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.9,
        ncol=1,
        handlelength=2.5,
        fontsize=8.5,
    )
    gradient_axis.legend(
        loc="lower right",
        frameon=False,
        fontsize=8.7,
        handlelength=2.5,
    )

    figure.suptitle(
        "How skgrad differentiates a fitted ReLU network",
        x=0.105,
        y=0.995,
        ha="left",
        fontsize=14.5,
        fontweight="bold",
        color=colors["text"],
    )
    figure.text(
        0.105,
        0.015,
        "Dashed verticals mark activation switches; orange tangents and dots are exact gradients at the batched evaluation grid.",
        ha="left",
        va="bottom",
        fontsize=8.7,
        color="#4E5965",
    )
    figure.subplots_adjust(left=0.105, right=0.985, top=0.92, bottom=0.155)
    return figure


def main() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    output_directory = repository_root / "docs"
    output_directory.mkdir(parents=True, exist_ok=True)
    figure = create_figure()
    figure.savefig(
        output_directory / f"{OUTPUT_STEM}.svg",
        format="svg",
        facecolor="white",
    )
    figure.savefig(
        output_directory / f"{OUTPUT_STEM}.png",
        format="png",
        dpi=180,
        facecolor="white",
    )
    plt.close(figure)


if __name__ == "__main__":
    main()
