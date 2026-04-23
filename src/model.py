"""ElasticNet model training with TimeSeriesSplit cross-validation."""

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error


def standardize_features(
    X_train: np.ndarray,
    X_test: np.ndarray | None = None,
) -> tuple[np.ndarray, StandardScaler]:
    """Standardize features using training set statistics only.

    This is critical for preventing data leakage — the scaler must be fit
    exclusively on training data, then applied to both train and test sets.

    Args:
        X_train: Training feature matrix (n_samples, n_features).
        X_test: Optional test feature matrix. If None, only training set is standardized.

    Returns:
        Tuple of (standardized_X_train, scaler) or
               (standardized_X_train, standardized_X_test, scaler) if X_test provided.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    if X_test is None:
        return X_train_scaled, scaler

    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def train_elasticnet_with_cv(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_splits: int = 5,
    param_grid: dict | None = None,
) -> tuple[ElasticNet, GridSearchCV]:
    """Train ElasticNet model using TimeSeriesSplit cross-validation.

    Uses chronological splitting to prevent future data leakage — each fold's
    test set comes strictly after its training set.

    Args:
        X_train: Standardized training feature matrix.
        y_train: Training target values (next-day close prices).
        n_splits: Number of TimeSeriesSplit folds.
        param_grid: Hyperparameter grid for grid search. Defaults to standard grid.

    Returns:
        Tuple of (best_model, grid_search_result).
    """
    if param_grid is None:
        param_grid = {
            "alpha": [0.01, 0.1, 1.0, 10.0],
            "l1_ratio": [0.1, 0.5, 0.7, 0.9, 0.95],
        }

    cv = TimeSeriesSplit(n_splits=n_splits)

    model = ElasticNet(
        random_state=42,
        max_iter=10000,
    )

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
    )

    print("Training ElasticNet with TimeSeriesSplit cross-validation...")
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV RMSE: {np.sqrt(-grid_search.best_score_):.4f}")

    return best_model, grid_search


def train_model(
    training_df,
    feature_cols: list[str],
    n_splits: int = 5,
    param_grid: dict | None = None,
) -> tuple[ElasticNet, GridSearchCV, StandardScaler]:
    """End-to-end model training pipeline.

    Loads features/targets, standardizes, and trains with CV grid search.

    Args:
        training_df: Training DataFrame from data module.
        feature_cols: List of feature column names.
        n_splits: Number of TimeSeriesSplit folds.
        param_grid: Hyperparameter grid for GridSearchCV.

    Returns:
        Tuple of (best_model, grid_search_result, scaler).
    """
    import numpy as np

    X_train = training_df[feature_cols].values
    y_train = training_df["Target"].values

    # Standardize
    X_train_scaled, scaler = standardize_features(X_train)

    # Train
    best_model, grid_search = train_elasticnet_with_cv(
        X_train_scaled, y_train, n_splits=n_splits, param_grid=param_grid,
    )

    return best_model, grid_search, scaler
