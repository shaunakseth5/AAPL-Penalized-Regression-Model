"""Cumulative returns visualization: actual vs predicted."""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd


def plot_returns_comparison(
    comparison_df: pd.DataFrame,
) -> plt.Figure:
    """Plot cumulative returns (%) for actual vs predicted prices.

    Shows the compounding effect of prediction errors over the year.

    Args:
        comparison_df: DataFrame with Date, Actual_Close, Predicted_Close columns.

    Returns:
        matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(16, 8))

    cum_actual = (comparison_df["Actual_Close"] / comparison_df["Actual_Close"].iloc[0] - 1) * 100
    cum_pred = (comparison_df["Predicted_Close"] / comparison_df["Predicted_Close"].iloc[0] - 1) * 100

    ax.plot(comparison_df["Date"], cum_actual, "b-", linewidth=2, label="Actual")
    ax.plot(comparison_df["Date"], cum_pred, "r--", linewidth=1.5, label="Predicted")

    # Quarter separators
    for q in range(2, 5):
        q_start = comparison_df[comparison_df["Quarter"] == q]["Date"].iloc[0]
        ax.axvline(x=q_start, color="gray", linestyle="-", alpha=0.3)
        ax.text(q_start, ax.get_ylim()[1] * 0.95, f"Q{q}", ha="center", va="top",
                backgroundcolor="white", fontsize=9)

    # Final return annotations
    last_date = comparison_df["Date"].iloc[-1]
    ax.annotate(
        f"{cum_actual.iloc[-1]:.1f}%",
        xy=(last_date, cum_actual.iloc[-1]),
        xytext=(5, 5), textcoords="offset points",
        ha="left", va="bottom", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
    )
    ax.annotate(
        f"{cum_pred.iloc[-1]:.1f}%",
        xy=(last_date, cum_pred.iloc[-1]),
        xytext=(5, 5), textcoords="offset points",
        ha="left", va="bottom", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
    )

    ax.set_title("AAPL Cumulative Returns: 2019 Actual vs. Predicted (%)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Cumulative Return (%)", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())

    plt.tight_layout()
    return fig
