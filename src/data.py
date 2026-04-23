"""Data loading, feature engineering, and train-test splitting."""

import pandas as pd
import numpy as np


def load_ohlcv(
    filepath: str,
    date_column: str = "Date",
) -> pd.DataFrame:
    """Load AAPL OHLCV data from CSV.

    Args:
        filepath: Path to CSV file with columns Date,Open,High,Low,Close,Volume.
        date_column: Name of the date column in the CSV.

    Returns:
        DataFrame sorted by date, indexed by Date, with clean OHLCV columns.
    """
    df = pd.read_csv(filepath, parse_dates=[date_column])
    df = df.sort_values(date_column).reset_index(drop=True)
    df[date_column] = df[date_column].dt.floor("D")
    return df


def create_features(
    df: pd.DataFrame,
    date_col: str = "Date",
    feature_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Create feature matrix (OHLCV) and target variable (next-day close).

    The target is the next day's close price (shifted -1), ensuring no look-
    ahead bias — you can only predict tomorrow's close after seeing today's.

    Args:
        df: OHLCV DataFrame.
        date_col: Name of the date column.
        feature_cols: Feature column names. Defaults to ['Open','High','Low','Close','Volume'].

    Returns:
        Tuple of (features_df, feature_names). Target is stored as 'Target' column.
    """
    if feature_cols is None:
        feature_cols = ["Open", "High", "Low", "Close", "Volume"]

    features_df = df[feature_cols].copy()
    features_df["Target"] = df["Close"].shift(-1)
    features_df.dropna(inplace=True)

    # Add temporal features
    features_df[date_col] = df.loc[features_df.index, date_col]
    features_df["Year"] = features_df[date_col].dt.year
    features_df["Month"] = features_df[date_col].dt.month
    features_df["Quarter"] = features_df[date_col].dt.quarter

    return features_df, feature_cols


def split_train_test(
    df: pd.DataFrame,
    train_end: str | None = None,
    test_start: str | None = None,
    test_end: str | None = None,
    date_col: str = "Date",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into training and test sets by date.

    Training set includes all data up to train_end (inclusive).
    Test set includes data from test_start through test_end.

    Args:
        df: DataFrame with a 'Date' column (output of create_features).
        train_end: Last date included in training (inclusive). Default: 2018-12-31.
        test_start: First date in test set. Default: 2019-01-01.
        test_end: Last date in test set. Default: 2019-12-31.
        date_col: Name of the date column.

    Returns:
        Tuple of (training_df, test_df).
    """
    if train_end is None:
        train_end = "2018-12-31"
    if test_start is None:
        test_start = "2019-01-01"
    if test_end is None:
        test_end = "2019-12-31"

    training_df = df[df[date_col] <= pd.Timestamp(train_end)]
    test_df = df[
        (df[date_col] >= pd.Timestamp(test_start))
        & (df[date_col] <= pd.Timestamp(test_end))
    ]

    return training_df, test_df


def get_last_training_state(
    training_df: pd.DataFrame, feature_cols: list[str]
) -> np.ndarray:
    """Get the OHLCV state from the last trading day of training.

    Used as the starting point for forward prediction simulation.

    Args:
        training_df: Training DataFrame.
        feature_cols: Feature column names.

    Returns:
        NumPy array of feature values for the last training day.
    """
    last_row = training_df.iloc[-1][feature_cols].values
    return last_row.reshape(1, -1)


DEFAULT_PARAMS = {
    "filepath": "data/aapl_ohlcv.csv",
    "train_end": "2018-12-31",
    "test_start": "2019-01-01",
    "test_end": "2019-12-31",
    "feature_cols": ["Open", "High", "Low", "Close", "Volume"],
}


def load_and_split_data(params: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """End-to-end data loading pipeline.

    Args:
        params: Configuration dict (overrides DEFAULT_PARAMS).

    Returns:
        Tuple of (training_df, test_df, feature_cols).
    """
    p = {**DEFAULT_PARAMS, **(params or {})}
    df = load_ohlcv(p["filepath"])
    features_df, feature_cols = create_features(df, feature_cols=p["feature_cols"])
    training_df, test_df = split_train_test(
        features_df,
        train_end=p["train_end"],
        test_start=p["test_start"],
        test_end=p["test_end"],
    )
    return training_df, test_df, feature_cols
