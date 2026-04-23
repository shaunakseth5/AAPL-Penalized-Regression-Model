"""Evaluation metrics and monthly/quarterly breakdown analysis."""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def compute_overall_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, float]:
    """Compute overall prediction accuracy metrics.

    Args:
        actual: Array of actual close prices.
        predicted: Array of predicted close prices.

    Returns:
        Dictionary with RMSE, MAE, MAPE, and R² scores.
    """
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    r2 = r2_score(actual, predicted)

    return {
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4),
        "MAPE": round(mape, 2),
        "R²": round(r2, 4),
    }


def compute_monthly_metrics(comparison_df: pd.DataFrame) -> pd.DataFrame:
    """Compute error metrics and return comparison for each month.

    For each month, calculates MAPE, RMSE, MAE, R², and compares actual vs
    predicted returns from start-to-end of the month.

    Args:
        comparison_df: DataFrame with Date, Actual_Close, Predicted_Close, Month.

    Returns:
        DataFrame with monthly metrics: Month, MonthName, Days, RMSE, MAE,
        MAPE, R², Start_Actual, End_Actual, Actual_Return(%), Predicted_Return(%).
    """
    results = []

    for month in sorted(comparison_df["Month"].unique()):
        month_data = comparison_df[comparison_df["Month"] == month].copy()

        actual = month_data["Actual_Close"].values
        predicted = month_data["Predicted_Close"].values

        if len(actual) < 5:
            continue

        rmse = np.sqrt(mean_squared_error(actual, predicted))
        mae = mean_absolute_error(actual, predicted)
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
        r2 = r2_score(actual, predicted) if len(actual) > 2 else None

        start_actual = actual[0]
        end_actual = actual[-1]
        start_pred = predicted[0]
        end_pred = predicted[-1]

        actual_return = (end_actual / start_actual - 1) * 100
        predicted_return = (end_pred / start_pred - 1) * 100 if start_pred > 0 else None

        results.append({
            "Month": month,
            "MonthName": month_data["MonthName"].iloc[0],
            "Days": len(month_data),
            "RMSE": round(rmse, 2),
            "MAE": round(mae, 2),
            "MAPE": round(mape, 2),
            "R²": round(r2, 4) if r2 is not None else None,
            "Start_Actual": round(start_actual, 2),
            "End_Actual": round(end_actual, 2),
            "Actual_Return": round(actual_return, 2),
            "Predicted_Return": round(predicted_return, 2) if predicted_return is not None else None,
        })

    return pd.DataFrame(results)


def compute_quarterly_metrics(comparison_df: pd.DataFrame) -> pd.DataFrame:
    """Compute error metrics and return comparison for each quarter.

    Similar to monthly metrics but aggregated by quarter (Q1–Q4).

    Args:
        comparison_df: DataFrame with Date, Actual_Close, Predicted_Close, Quarter.

    Returns:
        DataFrame with quarterly metrics.
    """
    results = []

    for quarter in sorted(comparison_df["Quarter"].unique()):
        q_data = comparison_df[comparison_df["Quarter"] == quarter].copy()

        actual = q_data["Actual_Close"].values
        predicted = q_data["Predicted_Close"].values

        if len(actual) < 5:
            continue

        rmse = np.sqrt(mean_squared_error(actual, predicted))
        mae = mean_absolute_error(actual, predicted)
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100

        start_actual = actual[0]
        end_actual = actual[-1]
        start_pred = predicted[0]
        end_pred = predicted[-1]

        actual_return = (end_actual / start_actual - 1) * 100
        predicted_return = (end_pred / start_pred - 1) * 100 if start_pred > 0 else None

        results.append({
            "Quarter": quarter,
            "Days": len(q_data),
            "RMSE": round(rmse, 2),
            "MAE": round(mae, 2),
            "MAPE": round(mape, 2),
            "Start_Actual": round(start_actual, 2),
            "End_Actual": round(end_actual, 2),
            "Actual_Return": round(actual_return, 2),
            "Predicted_Return": round(predicted_return, 2) if predicted_return is not None else None,
        })

    return pd.DataFrame(results)


def summarize_results(
    comparison_df: pd.DataFrame,
) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
    """Compute all metrics in one call.

    Args:
        comparison_df: DataFrame with Date, Actual_Close, Predicted_Close columns.

    Returns:
        Tuple of (overall_metrics dict, monthly_metrics DF, quarterly_metrics DF).
    """
    overall = compute_overall_metrics(
        comparison_df["Actual_Close"].values,
        comparison_df["Predicted_Close"].values,
    )
    monthly = compute_monthly_metrics(comparison_df)
    quarterly = compute_quarterly_metrics(comparison_df)
    return overall, monthly, quarterly
