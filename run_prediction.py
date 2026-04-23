"""Entry point — run full AAPL prediction pipeline end-to-end.

Usage:
    python run_prediction.py

Or with custom parameters:
    python run_prediction.py \
        --data-path ./data/aapl_ohlcv.csv \
        --train-end 2018-12-31 \
        --test-start 2019-01-01 \
        --test-end 2019-12-31

This script runs:
    1. Load OHLCV data and engineer features
    2. Train ElasticNet with TimeSeriesSplit cross-validation
    3. Recursive forward prediction through test period
    4. Compute RMSE, MAE, MAPE, R² + monthly/quarterly breakdowns
    5. Generate and save 4 visualization plots
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is on path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import pandas as pd
import matplotlib.pyplot as plt

from src.data import DEFAULT_PARAMS, load_and_split_data, get_last_training_state
from src.model import train_model
from src.predictor import run_prediction_pipeline
from src.metrics import summarize_results
from plots.price_comparison import plot_price_comparison
from plots.returns_comparison import plot_returns_comparison
from plots.quarterly_analysis import plot_quarterly_analysis
from plots.error_heatmap import plot_error_heatmap


def save_all_plots(comparison_df, monthly_metrics, quarterly_metrics, output_dir: Path) -> None:
    """Generate and save all four visualization plots to the output directory."""
    output_dir.mkdir(exist_ok=True)

    # Plot 1: Price comparison + error bars
    fig1 = plot_price_comparison(comparison_df)
    fig1.savefig(output_dir / "aapl_2019_prediction.png", dpi=300, bbox_inches="tight")
    plt.close(fig1)
    print(f"    ✓ Saved → {output_dir / 'aapl_2019_prediction.png'}")

    # Plot 2: Cumulative returns
    fig2 = plot_returns_comparison(comparison_df)
    fig2.savefig(output_dir / "aapl_2019_returns.png", dpi=300, bbox_inches="tight")
    plt.close(fig2)
    print(f"    ✓ Saved → {output_dir / 'aapl_2019_returns.png'}")

    # Plot 3: Quarterly analysis
    fig3 = plot_quarterly_analysis(quarterly_metrics, monthly_metrics)
    fig3.savefig(output_dir / "aapl_2019_quarterly_analysis.png", dpi=300, bbox_inches="tight")
    plt.close(fig3)
    print(f"    ✓ Saved → {output_dir / 'aapl_2019_quarterly_analysis.png'}")

    # Plot 4: Error heatmap
    fig4 = plot_error_heatmap(comparison_df)
    fig4.savefig(output_dir / "aapl_2019_error_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig4)
    print(f"    ✓ Saved → {output_dir / 'aapl_2019_error_heatmap.png'}")


def run_pipeline(args: argparse.Namespace) -> None:
    """Execute the full prediction pipeline.

    Args:
        args: Parsed command-line arguments.
    """
    params = {
        "filepath": args.data_path,
        "train_end": args.train_end,
        "test_start": args.test_start,
        "test_end": args.test_end,
        "feature_cols": ["Open", "High", "Low", "Close", "Volume"],
    }

    # ── Step 1: Load Data ──
    print("=" * 60)
    print("Step 1/5 — Loading and preparing data")
    print("=" * 60)

    training_df, test_df, feature_cols = load_and_split_data(params)
    print(f"    Training set: {len(training_df)} days ({training_df['Date'].iloc[0].date()} → {training_df['Date'].iloc[-1].date()})")
    print(f"    Test set:     {len(test_df)} days ({test_df['Date'].iloc[0].date()} → {test_df['Date'].iloc[-1].date()})")

    # ── Step 2: Train Model ──
    print(f"\n{'=' * 60}")
    print("Step 2/5 — Training ElasticNet (TimeSeriesSplit CV)")
    print("=" * 60)

    best_model, grid_search, scaler = train_model(training_df, feature_cols, n_splits=args.n_splits)

    # ── Step 3: Forward Prediction ──
    print(f"\n{'=' * 60}")
    print("Step 3/5 — Running recursive forward prediction")
    print("=" * 60)

    comparison_df = run_prediction_pipeline(best_model, scaler, training_df, test_df, feature_cols)
    last_state = get_last_training_state(training_df, feature_cols)
    print(f"    Last training state: close=${last_state[0][3]:.2f}")
    print(f"    Generated {len(comparison_df)} daily predictions for 2019")

    # ── Step 4: Evaluation ──
    print(f"\n{'=' * 60}")
    print("Step 4/5 — Computing evaluation metrics")
    print("=" * 60)

    overall, monthly_metrics, quarterly_metrics = summarize_results(comparison_df)

    # Overall metrics table
    print(f"\n{'─' * 60}")
    print(f"{'Metric':<25s} {'Value':>10s}")
    print(f"{'─' * 60}")
    for metric, value in overall.items():
        print(f"{metric:<25s} {value:>10.4f}")

    # Monthly metrics
    if not monthly_metrics.empty:
        print(f"\n{'─' * 60}")
        print("Monthly Breakdown")
        print(f"{'─' * 60}")
        display_cols = [c for c in ["Month", "Days", "RMSE", "MAE", "MAPE", "Actual_Return", "Predicted_Return"] if c in monthly_metrics.columns]
        print(monthly_metrics[display_cols].to_string(index=False))

    # Quarterly metrics
    if not quarterly_metrics.empty:
        print(f"\n{'─' * 60}")
        print("Quarterly Breakdown")
        print(f"{'─' * 60}")
        display_q = [c for c in ["Quarter", "Days", "RMSE", "MAE", "MAPE", "Actual_Return", "Predicted_Return"] if c in quarterly_metrics.columns]
        print(quarterly_metrics[display_q].to_string(index=False))

    # ── Step 5: Generate Plots ──
    print(f"\n{'=' * 60}")
    print("Step 5/5 — Generating visualizations")
    print("=" * 60)

    output_dir = project_root / "output"
    save_all_plots(comparison_df, monthly_metrics, quarterly_metrics, output_dir)

    # ── Summary ──
    print(f"\n{'=' * 60}")
    print("Pipeline complete! Results saved to output/")
    print(f"{'=' * 60}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="AAPL Penalized Regression — Out-of-sample price prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--data-path", type=str, default=DEFAULT_PARAMS["filepath"],
                        help="Path to AAPL OHLCV CSV file (default: data/aapl_ohlcv.csv)")
    parser.add_argument("--train-end", type=str, default=DEFAULT_PARAMS["train_end"],
                        help="Last training date inclusive (default: 2018-12-31)")
    parser.add_argument("--test-start", type=str, default=DEFAULT_PARAMS["test_start"],
                        help="First test date inclusive (default: 2019-01-01)")
    parser.add_argument("--test-end", type=str, default=DEFAULT_PARAMS["test_end"],
                        help="Last test date inclusive (default: 2019-12-31)")
    parser.add_argument("--n-splits", type=int, default=5,
                        help="Number of TimeSeriesSplit CV folds (default: 5)")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args)
