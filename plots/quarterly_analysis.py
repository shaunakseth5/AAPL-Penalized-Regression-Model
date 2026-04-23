"""Quarterly error analysis and returns comparison visualization."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_quarterly_analysis(
    quarterly_metrics: pd.DataFrame,
    monthly_metrics: pd.DataFrame | None = None,
) -> plt.Figure:
    """Three-panel figure showing quarterly error and returns analysis.

    Panels:
      1. Quarterly MAPE bar chart with overall MAPE reference line
      2. Quarterly actual vs predicted return comparison (grouped bars)
      3. Monthly MAPE bar chart (if monthly data provided)

    Args:
        quarterly_metrics: DataFrame with Quarter, MAPE, Actual_Return, Predicted_Return columns.
        monthly_metrics: Optional DataFrame with MonthName, MAPE columns.

    Returns:
        matplotlib Figure object.
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 15))

    overall_mape = quarterly_metrics["MAPE"].mean() if not quarterly_metrics.empty else None

    # ── Panel 1: Quarterly MAPE ──
    ax1 = axes[0]
    bars = ax1.bar(quarterly_metrics["Quarter"], quarterly_metrics["MAPE"], color="skyblue", alpha=0.8)
    if overall_mape is not None:
        ax1.axhline(y=overall_mape, color="red", linestyle="--", label=f"Overall MAPE: {overall_mape:.2f}%")

    ax1.set_title("Prediction Error by Quarter (MAPE)", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Mean Absolute Percentage Error (%)", fontsize=12)
    ax1.grid(True, alpha=0.3, axis="y")
    if overall_mape is not None:
        ax1.legend()

    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                 f"{height:.2f}%", ha="center", va="bottom", fontsize=9)

    # ── Panel 2: Quarterly Returns ──
    ax2 = axes[1]
    x = np.arange(len(quarterly_metrics))
    width = 0.35
    ax2.bar(x - width / 2, quarterly_metrics["Actual_Return"], width,
            label="Actual", color="blue", alpha=0.7)
    ax2.bar(x + width / 2, quarterly_metrics["Predicted_Return"], width,
            label="Predicted", color="red", alpha=0.7)
    ax2.axhline(y=0, color="black", linewidth=0.5)

    ax2.set_title("Quarterly Returns: Actual vs. Predicted", fontsize=14, fontweight="bold")
    ax2.set_ylabel("Quarterly Return (%)", fontsize=12)
    ax2.set_xticks(x, [f"Q{q}" for q in quarterly_metrics["Quarter"]])
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.legend()

    # Value labels on bars
    for i, v in enumerate(quarterly_metrics["Actual_Return"]):
        offset = 1 if v >= 0 else -2
        ax2.text(i - width / 2, v + offset, f"{v:.1f}%", ha="center", fontsize=9)
    for i, v in enumerate(quarterly_metrics["Predicted_Return"] or []):
        offset = 1 if v >= 0 else -2
        ax2.text(i + width / 2, v + offset, f"{v:.1f}%", ha="center", fontsize=9)

    # ── Panel 3: Monthly MAPE ──
    if monthly_metrics is not None and not monthly_metrics.empty:
        ax3 = axes[2]
        months_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        monthly_subset = monthly_metrics.dropna(subset=["MAPE"])

        if not monthly_subset.empty:
            bars = ax3.bar(monthly_subset["MonthName"], monthly_subset["MAPE"], color="lightgreen", alpha=0.8)
            if overall_mape is not None:
                ax3.axhline(y=overall_mape, color="red", linestyle="--", label=f"Overall MAPE: {overall_mape:.2f}%")

        ax3.set_title("Prediction Error by Month (MAPE)", fontsize=14, fontweight="bold")
        ax3.set_ylabel("Mean Absolute Percentage Error (%)", fontsize=12)
        ax3.grid(True, alpha=0.3, axis="y")
        if overall_mape is not None:
            ax3.legend()

        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                     f"{height:.2f}%", ha="center", va="bottom", fontsize=9)
    else:
        axes[2].axis("off")

    plt.tight_layout()
    return fig
