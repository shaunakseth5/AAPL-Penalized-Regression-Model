"""Error pattern heatmap: error percentage by month × day of month."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_error_heatmap(
    comparison_df: pd.DataFrame,
) -> plt.Figure:
    """Heatmap showing prediction error patterns across the year.

    Each cell shows the mean error percentage for a (month, day-of-month) bin,
    revealing whether certain days consistently over/under-predict.

    Args:
        comparison_df: DataFrame with Date, Error_Pct columns.

    Returns:
        matplotlib Figure object.
    """
    df = comparison_df.copy()
    df["Day"] = df["Date"].dt.day

    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    pivot_data = df.pivot_table(
        index="Month", columns="Day", values="Error_Pct", aggfunc="mean"
    )

    # Only include months that exist in data
    active_months = sorted(pivot_data.index)
    month_labels = [month_names[m - 1] for m in active_months if m <= 12]

    fig, ax = plt.subplots(figsize=(16, 8))

    im = ax.imshow(pivot_data.loc[active_months], cmap="RdYlGn_r",
                   aspect="auto", vmin=-15, vmax=15)

    cbar = plt.colorbar(im, ax=ax, label="Error Percentage (%)")

    ax.set_title("Prediction Error Heatmap by Day of Month (2019)", fontsize=14, fontweight="bold")
    ax.set_yticks(range(len(month_labels)))
    ax.set_yticklabels(month_labels)

    x_tick_positions = [d for d in range(1, 31, 5)]
    ax.set_xticks(x_tick_positions)
    ax.set_xticklabels([str(d) for d in x_tick_positions])
    ax.set_xlabel("Day of Month", fontsize=12)
    ax.set_ylabel("Month", fontsize=12)

    plt.tight_layout()
    return fig
