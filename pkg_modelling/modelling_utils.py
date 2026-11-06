"""Reusable utilities for binary-classification model development."""

from collections.abc import Sequence
from numbers import Integral
from typing import Any, Protocol

import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler


class ProbabilityEstimator(Protocol):
    """Protocol for estimators that expose class probabilities."""

    def predict_proba(self, X: pd.DataFrame) -> Any: ...


def temporal_train_validation_split(
    dataframe: pd.DataFrame,
    date_column: str,
    validation_fraction: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Split data chronologically, keeping equal dates in the same partition."""
    if date_column not in dataframe.columns:
        raise ValueError(f"Date column {date_column!r} is not present.")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be strictly between 0 and 1.")

    sorted_dataframe = dataframe.copy()
    sorted_dataframe[date_column] = pd.to_datetime(
        sorted_dataframe[date_column],
        errors="coerce",
    )
    if sorted_dataframe[date_column].isna().any():
        raise ValueError(
            f"Date column {date_column!r} contains missing or invalid dates."
        )

    sorted_dataframe = sorted_dataframe.sort_values(date_column).reset_index(
        drop=True
    )
    split_position = int(len(sorted_dataframe) * (1 - validation_fraction))
    if split_position >= len(sorted_dataframe):
        raise ValueError("The dataframe is too small to create both partitions.")

    validation_start_date = sorted_dataframe.loc[split_position, date_column]
    development = sorted_dataframe.loc[
        sorted_dataframe[date_column] < validation_start_date
    ].reset_index(drop=True)
    validation = sorted_dataframe.loc[
        sorted_dataframe[date_column] >= validation_start_date
    ].reset_index(drop=True)

    if development.empty or validation.empty:
        raise ValueError(
            "The requested temporal split produces an empty partition; "
            "provide more rows or more distinct dates."
        )
    if development[date_column].max() >= validation[date_column].min():
        raise ValueError("Development and validation dates overlap.")

    return development, validation, validation_start_date


def build_preprocessor(
    continuous_features: Sequence[str],
    binary_features: Sequence[str],
    categorical_features: Sequence[str] = (),
    spline_features: Sequence[str] = (),
    spline_n_knots: int = 4,
    spline_degree: int = 2,
) -> ColumnTransformer:
    """Build an unfitted preprocessor for the supplied feature groups."""
    feature_groups = {
        "continuous": list(continuous_features),
        "binary": list(binary_features),
        "categorical": list(categorical_features),
        "spline": list(spline_features),
    }

    for group_name, features in feature_groups.items():
        seen = set()
        duplicates = set()
        for feature in features:
            if feature in seen:
                duplicates.add(feature)
            seen.add(feature)
        if duplicates:
            raise ValueError(
                f"Feature group {group_name!r} contains duplicate columns: "
                f"{sorted(duplicates)}"
            )

    feature_owners: dict[str, str] = {}
    overlaps: list[tuple[str, str, str]] = []
    for group_name, features in feature_groups.items():
        for feature in features:
            previous_group = feature_owners.get(feature)
            if previous_group is not None:
                overlaps.append((feature, previous_group, group_name))
            else:
                feature_owners[feature] = group_name
    if overlaps:
        raise ValueError(
            "Feature columns must not appear in more than one group; "
            f"overlaps: {overlaps}"
        )

    if (
        isinstance(spline_n_knots, bool)
        or not isinstance(spline_n_knots, Integral)
        or spline_n_knots < 2
    ):
        raise ValueError("spline_n_knots must be an integer of at least 2.")
    if (
        isinstance(spline_degree, bool)
        or not isinstance(spline_degree, Integral)
        or spline_degree < 0
    ):
        raise ValueError("spline_degree must be a non-negative integer.")

    if not any(feature_groups.values()):
        raise ValueError("At least one feature group must be non-empty.")

    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if feature_groups["continuous"]:
        transformers.append(
            (
                "continuous",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                feature_groups["continuous"],
            )
        )
    if feature_groups["binary"]:
        transformers.append(
            (
                "binary",
                Pipeline(
                    [("imputer", SimpleImputer(strategy="most_frequent"))]
                ),
                feature_groups["binary"],
            )
        )
    if feature_groups["categorical"]:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                feature_groups["categorical"],
            )
        )
    if feature_groups["spline"]:
        transformers.append(
            (
                "spline",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        (
                            "spline",
                            SplineTransformer(
                                n_knots=spline_n_knots,
                                degree=spline_degree,
                                knots="quantile",
                                include_bias=False,
                            ),
                        ),
                        ("scaler", StandardScaler()),
                    ]
                ),
                feature_groups["spline"],
            )
        )

    return ColumnTransformer(
        transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_logistic_pipeline(
    preprocessor: BaseEstimator,
    C: float = 1.0,
    penalty: str = "l2",
    solver: str = "lbfgs",
    class_weight: str | dict[int | str, float] | None = None,
    random_state: int | None = 42,
) -> Pipeline:
    """Create an unfitted pipeline with a cloned preprocessor and logistic model."""
    if not C > 0:
        raise ValueError("C must be positive.")

    return Pipeline(
        [
            ("preprocessing", clone(preprocessor)),
            (
                "classifier",
                LogisticRegression(
                    C=C,
                    penalty=penalty,
                    solver=solver,
                    class_weight=class_weight,
                    random_state=random_state,
                    max_iter=2000,
                ),
            ),
        ]
    )


def evaluate_probability_model(
    model_name: str,
    model: ProbabilityEstimator,
    X: pd.DataFrame,
    y: pd.Series | Sequence[int],
) -> dict[str, str | float]:
    """Evaluate predicted probabilities without thresholding them."""
    probabilities = model.predict_proba(X)[:, 1]

    return {
        "model": model_name,
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "pr_auc": float(average_precision_score(y, probabilities)),
        "log_loss": float(log_loss(y, probabilities)),
        "brier_score": float(brier_score_loss(y, probabilities)),
        "mean_predicted_default": float(probabilities.mean()),
        "actual_default_rate": float(pd.Series(y).mean()),
    }
