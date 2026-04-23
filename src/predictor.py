"""Recursive one-step-ahead forward prediction simulation."""

import numpy as np
import pandas as pd


def recursive_forward_prediction(
    best_model,
    scaler,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    last_training_features: np.ndarray,
) -> list[float]:
    """Run recursive forward prediction through the entire test period.

    This simulates a live trading system: for each day in 2019, use the real
    OHLCV (open/high/low/volume) as known inputs, predict the close price, then
    use that *predicted* close (not actual) as input for the next day's prediction.

    This is a realistic forward test — in live trading you don't know tomorrow's
    close when making today's prediction, so your prediction feeds into tomorrow.

    Args:
        best_model: Trained ElasticNet model.
        scaler: Fitted StandardScaler.
        test_df: Test DataFrame with OHLCV features.
        feature_cols: Column names for the 5 features.
        last_training_features: OHLCV values from last training day (used as seed).

    Returns:
        List of predicted close prices, aligned with test_df index order.
    """
    # Initialize state with last training day's real values
    current_state = last_training_features.copy()

    predictions = []
    sorted_dates = sorted(test_df["Date"].unique())

    for date in sorted_dates:
        # Get real OHLCV for this date
        day_data = test_df[test_df["Date"] == date].iloc[0]
        real_open = day_data["Open"]
        real_high = day_data["High"]
        real_low = day_data["Low"]
        real_volume = day_data["Volume"]

        # Build feature vector: use real OHLCV for open/high/low,
        # but predicted close for the 'Close' column (simulating unknown close)
        features = np.array([[
            real_open,
            real_high,
            real_low,
            current_state[3],  # previous day's predicted close as today's "close" input
            real_volume,
        ]])

        # Scale and predict next-day close
        features_scaled = scaler.transform(features)
        pred_close = float(best_model.predict(features_scaled)[0])

        predictions.append(pred_close)

        # Update state for next day
        current_state[0] = real_open      # tomorrow's open (known today, but we update sequentially)
        current_state[1] = real_high      # tomorrow's high
        current_state[2] = real_low       # tomorrow's low
        current_state[3] = pred_close     # tomorrow's "close" = our prediction for today

    return predictions


def run_prediction_pipeline(
    best_model,
    scaler,
    training_df,
    test_df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Run full forward prediction and return comparison DataFrame.

    Args:
        best_model: Trained ElasticNet model.
        scaler: Fitted StandardScaler.
        training_df: Training DataFrame (for last state extraction).
        test_df: Test DataFrame with real prices.
        feature_cols: Feature column names.

    Returns:
        DataFrame with Date, Actual_Close, Predicted_Close, Error, Error_Pct.
    """
    from src.data import get_last_training_state

    last_state = get_last_training_state(training_df, feature_cols)
    predictions = recursive_forward_prediction(
        best_model, scaler, test_df, feature_cols, last_state,
    )

    # Align predictions with dates in chronological order
    sorted_dates = sorted(test_df["Date"].unique())
    comparison = pd.DataFrame({
        "Date": sorted_dates,
        "Actual_Close": [test_df[test_df["Date"] == d]["Close"].iloc[0] for d in sorted_dates],
        "Predicted_Close": predictions,
    })

    comparison["Error"] = comparison["Actual_Close"] - comparison["Predicted_Close"]
    comparison["Error_Pct"] = (comparison["Error"] / comparison["Actual_Close"]) * 100
    comparison["Abs_Error_Pct"] = np.abs(comparison["Error_Pct"])

    # Add temporal columns
    comparison["Year"] = comparison["Date"].dt.year
    comparison["Month"] = comparison["Date"].dt.month
    comparison["MonthName"] = comparison["Date"].dt.strftime("%b")
    comparison["Quarter"] = comparison["Date"].dt.quarter

    return comparison
