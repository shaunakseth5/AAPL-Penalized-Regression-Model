"""Actual vs predicted prices visualization with quarter separators."""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd


def plot_price_comparison(
    comparison_df: pd.DataFrame,
) -> plt.Figure:
    """Plot actual and predicted closing prices side-by-side.

    Two subplots:
      1. Price line chart with quarter separators
      2. Prediction error bar chart (green=overprediction, red=underprediction)

    Args:
        comparison_df: DataFrame with Date, Actual_Close, Predicted_Close, Error, Quarter columns.

    Returns:
        matplotlib Figure object.
    """
    fig, axes = plt.subplots(2, 1, figsize=(18, 10), sharex=True)

    # ── Price chart ──
    ax1 = axes[0]
    ax1.plot(comparison_df["Date"], comparison_df["Actual_Close"],
             "b-", linewidth=2, label="Actual Close")
    ax1.plot(comparison_df["Date"], comparison_df["Predicted_Close"],
             "r--", linewidth=1.5, label="Predicted Close")

    # Quarter separators
    for q in range(2, 5):
        q_start = comparison_df[comparison_df["Quarter"] == q]["Date"].iloc[0]
        ax1.axvline(x=q_start, color="gray", linestyle="-", alpha=0.3)

    ax1.set_title("AAPL Closing Price: 2019 Actual vs. Predicted", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Price ($)", fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)

    # ── Error bar chart ──
    ax2 = axes[1]
    colors = ["g" if e >= 0 else "r" for e in comparison_df["Error"]]
    ax2.bar(comparison_df["Date"], comparison_df["Error"],
            color=colors, alpha=0.6, width=1.5)
    ax2.axhline(y=0, color="black", linewidth=0.5)
    ax2.axhline(y=comparison_df["Error"].mean(), color="blue", linestyle="--",
                linewidth=1.5, label=f"Mean Error: ${comparison_df['Error'].mean():.2f}")

    for q in range(2, 5):
        q_start = comparison_df[comparison_df["Quarter"] == q]["Date"].iloc[0]
        ax2.axvline(x=q_start, color="gray", linestyle="-", alpha=0.3)

    ax2.set_title("Prediction Error (Actual − Predicted)", fontsize=14, fontweight="bold")
    ax2.set_ylabel("Error ($)", fontsize=12)
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.legend(fontsize=11)

    # ── Format x-axis ──
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        ax.xaxis.set_major_locator(mdates.MonthLocator())

    plt.tight_layout()
    return fig
