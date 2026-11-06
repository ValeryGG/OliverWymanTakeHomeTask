"""Visualisation utilities for binary-classification performance."""

from collections.abc import Mapping
from numbers import Real

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import ArrayLike, NDArray
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


_FALLBACK_COLOURS = (
    "#2463A3",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
)
_FALLBACK_LINESTYLES = ("-",)


def _validate_binary_target(y_true: ArrayLike) -> NDArray[np.int_]:
    """Return a validated one-dimensional target containing both classes."""
    target = np.asarray(y_true)
    if target.ndim != 1 or target.size == 0:
        raise ValueError("y_true must be a non-empty one-dimensional array.")
    if not np.isin(target, (0, 1)).all():
        raise ValueError("y_true must contain only 0 and 1.")
    if not (target == 0).any() or not (target == 1).any():
        raise ValueError("y_true must contain both classes, 0 and 1.")
    return target.astype(int, copy=False)


def _validate_predictions(
    predictions: Mapping[str, ArrayLike],
    expected_length: int,
) -> dict[str, NDArray[np.float64]]:
    """Return validated finite one-dimensional score arrays by model name."""
    if not isinstance(predictions, Mapping) or not predictions:
        raise ValueError("predictions must be a non-empty mapping.")

    validated: dict[str, NDArray[np.float64]] = {}
    for model_name, scores in predictions.items():
        if not isinstance(model_name, str):
            raise ValueError("Prediction mapping keys must be model-name strings.")
        try:
            score_array = np.asarray(scores, dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Predictions for {model_name!r} must be numeric."
            ) from error
        if score_array.ndim != 1 or score_array.size == 0:
            raise ValueError(
                f"Predictions for {model_name!r} must be a non-empty "
                "one-dimensional array."
            )
        if score_array.size != expected_length:
            raise ValueError(
                f"Predictions for {model_name!r} must have length "
                f"{expected_length}; got {score_array.size}."
            )
        if not np.isfinite(score_array).all():
            raise ValueError(
                f"Predictions for {model_name!r} must contain only finite values."
            )
        validated[model_name] = score_array
    return validated


def _validate_probabilities(
    predictions: Mapping[str, ArrayLike],
    expected_length: int,
) -> dict[str, NDArray[np.float64]]:
    """Validate prediction scores as probabilities between zero and one."""
    validated = _validate_predictions(predictions, expected_length)
    for model_name, probabilities in validated.items():
        if ((probabilities < 0) | (probabilities > 1)).any():
            raise ValueError(
                f"Predictions for {model_name!r} must be between 0 and 1."
            )
    return validated


def _resolve_model_style(
    model_name: str,
    model_index: int,
    model_styles: Mapping[str, Mapping[str, str]] | None,
) -> dict[str, str]:
    """Resolve supplied style values with deterministic fallbacks."""
    if model_styles is not None and not isinstance(model_styles, Mapping):
        raise ValueError("model_styles must be a mapping or None.")

    style = model_styles.get(model_name, {}) if model_styles is not None else {}
    if not isinstance(style, Mapping):
        raise ValueError(f"Style for {model_name!r} must be a mapping.")

    return {
        "color": style.get(
            "color",
            _FALLBACK_COLOURS[model_index % len(_FALLBACK_COLOURS)],
        ),
        "linestyle": style.get(
            "linestyle",
            _FALLBACK_LINESTYLES[model_index % len(_FALLBACK_LINESTYLES)],
        ),
    }


def plot_roc_curves(
    y_true: ArrayLike,
    predictions: Mapping[str, ArrayLike],
    model_styles: Mapping[str, Mapping[str, str]] | None = None,
    ax: Axes | None = None,
    title: str = "ROC curves — temporal validation",
) -> Axes:
    """Plot ROC curves for validated binary targets and model risk scores."""
    target = _validate_binary_target(y_true)
    score_arrays = _validate_predictions(predictions, target.size)

    if ax is None:
        _, ax = plt.subplots()

    for index, (model_name, scores) in enumerate(score_arrays.items()):
        style = _resolve_model_style(model_name, index, model_styles)
        false_positive_rate, true_positive_rate, _ = roc_curve(target, scores)
        area = roc_auc_score(target, scores)
        ax.plot(
            false_positive_rate,
            true_positive_rate,
            color=style["color"],
            linestyle=style["linestyle"],
            label=f"{model_name} (AUC = {area:.3f})",
        )

    ax.plot(
        [0, 1],
        [0, 1],
        color="grey",
        linestyle="--",
        linewidth=1,
        label="Random ranking",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False-positive rate")
    ax.set_ylabel("True-positive rate")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    return ax


def plot_precision_recall_curves(
    y_true: ArrayLike,
    predictions: Mapping[str, ArrayLike],
    model_styles: Mapping[str, Mapping[str, str]] | None = None,
    ax: Axes | None = None,
    title: str = "Precision–recall curves — temporal validation",
) -> Axes:
    """Plot precision–recall curves for validated binary targets and scores."""
    target = _validate_binary_target(y_true)
    score_arrays = _validate_predictions(predictions, target.size)

    if ax is None:
        _, ax = plt.subplots()

    for index, (model_name, scores) in enumerate(score_arrays.items()):
        style = _resolve_model_style(model_name, index, model_styles)
        precision, recall, _ = precision_recall_curve(target, scores)
        average_precision = average_precision_score(target, scores)
        ax.plot(
            recall,
            precision,
            color=style["color"],
            linestyle=style["linestyle"],
            label=f"{model_name} (AP = {average_precision:.3f})",
        )

    default_rate = float(target.mean())
    ax.axhline(
        default_rate,
        color="grey",
        linestyle="--",
        linewidth=1,
        label=f"Default rate = {default_rate:.3f}",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    return ax


def plot_calibration_curves(
    y_true: ArrayLike,
    predictions: Mapping[str, ArrayLike],
    model_styles: Mapping[str, Mapping[str, str]] | None = None,
    ax: Axes | None = None,
    title: str = "Calibration curves — temporal validation",
    n_bins: int = 10,
    strategy: str = "quantile",
) -> Axes:
    """Plot observed default rates against mean predicted probabilities."""
    target = _validate_binary_target(y_true)
    probability_arrays = _validate_probabilities(predictions, target.size)

    if isinstance(n_bins, bool) or not isinstance(n_bins, (int, np.integer)):
        raise ValueError("n_bins must be a positive integer.")
    if n_bins <= 0:
        raise ValueError("n_bins must be a positive integer.")
    if not isinstance(strategy, str) or strategy not in {"uniform", "quantile"}:
        raise ValueError("strategy must be either 'uniform' or 'quantile'.")

    if ax is None:
        _, ax = plt.subplots()

    for index, (model_name, probabilities) in enumerate(
        probability_arrays.items()
    ):
        style = _resolve_model_style(model_name, index, model_styles)
        observed_rate, mean_predicted_probability = calibration_curve(
            target,
            probabilities,
            n_bins=n_bins,
            strategy=strategy,
        )
        ax.plot(
            mean_predicted_probability,
            observed_rate,
            color=style["color"],
            linestyle=style["linestyle"],
            marker="o",
            label=model_name,
        )

    ax.plot(
        [0, 1],
        [0, 1],
        color="grey",
        linestyle="--",
        linewidth=1,
        label="Perfect calibration",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed default rate")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    return ax


def plot_validation_curves(
    y_true: ArrayLike,
    predictions: Mapping[str, ArrayLike],
    model_styles: Mapping[str, Mapping[str, str]] | None = None,
    figure_size: tuple[Real, Real] = (20, 6),
    title_suffix: str = "temporal validation",
    calibration_bins: int = 10,
    calibration_strategy: str = "quantile",
) -> tuple[Figure, np.ndarray]:
    """Plot ROC, precision–recall and calibration curves side by side."""
    target = _validate_binary_target(y_true)
    probability_arrays = _validate_probabilities(predictions, target.size)

    try:
        dimensions = tuple(figure_size)
    except TypeError as error:
        raise ValueError(
            "figure_size must contain exactly two positive numeric values."
        ) from error
    if len(dimensions) != 2 or any(
        isinstance(dimension, (bool, np.bool_))
        or not isinstance(dimension, Real)
        or not np.isfinite(dimension)
        or dimension <= 0
        for dimension in dimensions
    ):
        raise ValueError(
            "figure_size must contain exactly two positive numeric values."
        )

    figure, axes = plt.subplots(1, 3, figsize=dimensions)
    plot_roc_curves(
        target,
        probability_arrays,
        model_styles=model_styles,
        ax=axes[0],
        title=f"ROC curves — {title_suffix}",
    )
    plot_precision_recall_curves(
        target,
        probability_arrays,
        model_styles=model_styles,
        ax=axes[1],
        title=f"Precision–recall curves — {title_suffix}",
    )
    plot_calibration_curves(
        target,
        probability_arrays,
        model_styles=model_styles,
        ax=axes[2],
        title=f"Calibration curves — {title_suffix}",
        n_bins=calibration_bins,
        strategy=calibration_strategy,
    )
    figure.tight_layout()
    return figure, axes
