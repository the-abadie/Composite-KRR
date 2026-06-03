from pathlib import Path
import matplotlib

import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import RandomizedSearchCV
from utilities import configure_logging
from config import VERBOSITY
import logging
import matplotlib.pyplot as plt


configure_logging(VERBOSITY)
logger = logging.getLogger("post-processing")

def attach_random_search_history(search: RandomizedSearchCV, *, scoring) -> None:
    cv_results = search.cv_results_
    best_score = -np.inf
    best_validation_error = np.inf
    history = []
    improvements = []

    for index, params in enumerate(cv_results["params"], start=1):
        result_index = index - 1
        mean_test_score = float(cv_results["mean_test_score"][result_index])
        std_test_score = float(cv_results["std_test_score"][result_index])
        validation_error = validation_error_from_score(mean_test_score, scoring)
        improved = mean_test_score > best_score

        if improved:
            best_score = mean_test_score
            best_validation_error = validation_error

        record = {
            "iteration": index,
            "mean_test_score": mean_test_score,
            "std_test_score": std_test_score,
            "validation_error": validation_error,
            "best_mean_test_score": best_score,
            "best_validation_error": best_validation_error,
            "improved": improved,
            "params": params,
        }
        history.append(record)

        if improved:
            improvements.append(record)

    search.search_history_ = history
    search.improvement_history_ = improvements


def combined_random_search_history(stages) -> list[dict]:
    combined = []
    iteration_offset = 0

    for stage_index, search in enumerate(stages, start=1):
        stage_name = getattr(search, "stage_name", f"Stage {stage_index}")
        history = getattr(search, "search_history_", [])

        for record in history:
            stage_iteration = record["iteration"]
            combined_record = dict(record)
            combined_record["stage_improved"] = combined_record["improved"]
            combined_record["stage"] = stage_name
            combined_record["stage_iteration"] = stage_iteration
            combined_record["iteration"] = iteration_offset + stage_iteration
            combined.append(combined_record)

        iteration_offset += len(history)

    best_score = -np.inf
    best_validation_error = np.inf
    for record in combined:
        improved = record["mean_test_score"] > best_score
        if improved:
            best_score = record["mean_test_score"]
            best_validation_error = record["validation_error"]

        record["improved"] = improved
        record["best_mean_test_score"] = best_score
        record["best_validation_error"] = best_validation_error

    return combined


def offset_bayesian_search_history(
    history: list[dict],
    *,
    iteration_offset: int,
    initial_best_score: float = -np.inf,
    initial_best_validation_error: float = np.inf,
) -> list[dict]:
    offset_history = []
    best_score = initial_best_score
    best_validation_error = initial_best_validation_error

    for record in history:
        stage_iteration = record["iteration"]
        offset_record = dict(record)
        offset_record["stage_improved"] = offset_record["improved"]
        offset_record["stage"] = "Bayesian"
        offset_record["stage_iteration"] = stage_iteration
        offset_record["iteration"] = iteration_offset + stage_iteration
        improved = offset_record["mean_test_score"] > best_score
        if improved:
            best_score = offset_record["mean_test_score"]
            best_validation_error = offset_record["validation_error"]

        offset_record["improved"] = improved
        offset_record["best_mean_test_score"] = best_score
        offset_record["best_validation_error"] = best_validation_error
        offset_history.append(offset_record)

    return offset_history


def bayesian_search_history(study, *, scoring) -> list[dict]:
    best_score = -np.inf
    best_validation_error = np.inf
    history = []

    for trial in study.trials:
        if trial.value is None:
            continue

        mean_test_score = float(trial.value)
        split_scores = np.asarray(
            trial.user_attrs.get("split_scores", []),
            dtype=float,
        )
        std_test_score = (
            float(np.std(split_scores))
            if split_scores.size > 0 and np.all(np.isfinite(split_scores))
            else np.nan
        )
        validation_error = validation_error_from_score(mean_test_score, scoring)
        improved = mean_test_score > best_score

        if improved:
            best_score = mean_test_score
            best_validation_error = validation_error

        history.append(
            {
                "iteration": len(history) + 1,
                "mean_test_score": mean_test_score,
                "std_test_score": std_test_score,
                "validation_error": validation_error,
                "best_mean_test_score": best_score,
                "best_validation_error": best_validation_error,
                "improved": improved,
                "params": dict(trial.params),
            }
        )

    return history


def best_validation_mae_and_fold_std(
    search_result,
    scoring,
    ddof: int = 0,
) -> tuple[float, float]:
    """
    Return the mean validation error and fold standard deviation for the
    final selected hyperparameters.

    With sklearn's ``neg_mean_absolute_error`` scorer, the returned mean is
    the best validation MAE.
    """
    fold_errors = best_validation_fold_errors(search_result, scoring=scoring)
    finite_errors = fold_errors[np.isfinite(fold_errors)]

    if finite_errors.size == 0:
        raise ValueError("No finite validation fold errors are available.")

    mean_error = float(np.mean(finite_errors))
    if finite_errors.size <= ddof:
        fold_std = np.nan
    else:
        fold_std = float(np.std(finite_errors, ddof=ddof))

    return mean_error, fold_std


def best_validation_fold_errors(search_result, *, scoring) -> NDArray:
    """Return per-fold validation errors for the final selected hyperparameters."""
    final_stage = getattr(search_result, "final_stage", search_result)

    best_split_scores = getattr(final_stage, "best_split_scores_", None)
    if best_split_scores is not None:
        split_scores = np.asarray(best_split_scores, dtype=float)
        if split_scores.size > 0:
            return np.asarray(
                [
                    validation_error_from_score(float(score), scoring)
                    for score in split_scores
                ],
                dtype=float,
            )

    cv_results = getattr(final_stage, "cv_results_", None)
    if cv_results is None:
        raise ValueError(
            "Search result does not expose per-fold CV scores. Expected "
            "`best_split_scores_` or `cv_results_` with split test scores."
        )

    best_index = int(
        getattr(
            final_stage,
            "best_index_",
            np.argmax(np.asarray(cv_results["mean_test_score"], dtype=float)),
        )
    )
    split_scores = np.asarray(
        [
            cv_results[key][best_index]
            for key in _sorted_split_score_keys(cv_results)
        ],
        dtype=float,
    )

    return np.asarray(
        [
            validation_error_from_score(float(score), scoring)
            for score in split_scores
        ],
        dtype=float,
    )


def _sorted_split_score_keys(cv_results: dict) -> list[str]:
    split_keys = [
        key
        for key in cv_results
        if key.startswith("split") and key.endswith("_test_score")
    ]
    if not split_keys:
        raise ValueError("CV results do not include per-fold split test scores.")

    def split_index(key: str) -> int:
        index = key.removesuffix("_test_score").removeprefix("split")
        if not index.isdecimal():
            raise ValueError(f"Unexpected split score key: {key}")
        return int(index)

    return sorted(split_keys, key=split_index)


def log_random_search_improvements(search_result, *, scoring, logger) -> None:
    metric_label = validation_metric_label(scoring).lower()
    for record in search_result.random_search_improvements_:
        logger.debug(
            "Random search improved at iteration "
            f"{record['iteration']} ({record['stage']} iteration "
            f"{record['stage_iteration']}): best {metric_label} "
            f"{record['best_validation_error']:.6g} "
            f"(score {record['best_mean_test_score']:.6g})"
        )

def validation_error_from_score(score: float, scoring) -> float:
    if _is_negative_error_scorer(scoring):
        return -score
    return score


def validation_metric_label(scoring) -> str:
    if _is_negative_error_scorer(scoring):
        return "Validation error"
    return "Validation score"


def _annotate_stage_boundaries(ax, history: list[dict]) -> None:
    current_stage = history[0]["stage"]
    stage_start = history[0]["iteration"]

    for previous, current in zip(history, history[1:]):
        if current["stage"] == current_stage:
            continue

        stage_end = previous["iteration"]
        boundary = stage_end + 0.5
        ax.axvline(boundary, color="0.45", linestyle="--", linewidth=1, alpha=0.6)
        _add_stage_label(ax, current_stage, stage_start, stage_end)

        current_stage = current["stage"]
        stage_start = current["iteration"]

    _add_stage_label(ax, current_stage, stage_start, history[-1]["iteration"])


def _add_stage_label(ax, stage_name: str, start: int, end: int) -> None:
    ax.text(
        (start + end) / 2,
        0.98,
        stage_name,
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=9,
        color="0.25",
    )


def _is_negative_error_scorer(scoring) -> bool:
    return isinstance(scoring, str) and scoring.startswith("neg_")


def runtime_analysis(start_end_list:list[tuple[float, float, str]]) -> None:
    dts = []
    for segment in start_end_list:
        dts.append(segment[1] - segment[0])

    total_time:float = sum(dts)
    percents = [100*dt/total_time for dt in dts]

    logger.warning("RUNTIME BREAKDOWN")
    for i in range(len(start_end_list)):
        logger.warning(f"{percents[i]:05.2f}%: {start_end_list[i][2]}")


def plot_random_search_validation_error(
    search_result,
    OUTPUT_DIR:str,
    scoring: str | None = None,
):
    history = search_result.random_search_history_
    if not history:
        raise ValueError("No random search history is available to plot.")
    bayesian_history = search_result.bayesian_search_history_
    plot_history = history + bayesian_history

    matplotlib.use("Agg")

    improvements = [
        record
        for record in history
        if record["improved"] and np.isfinite(record["best_validation_error"])
    ]
    if not improvements:
        raise ValueError("No random search improvements are available to plot.")

    iterations = [record["iteration"] for record in improvements]
    best_validation_errors = [
        record["best_validation_error"] for record in improvements
    ]
    random_line_iterations = list(iterations)
    random_line_errors = list(best_validation_errors)
    if random_line_iterations[-1] != history[-1]["iteration"]:
        random_line_iterations.append(history[-1]["iteration"])
        random_line_errors.append(history[-1]["best_validation_error"])
    y_label = validation_metric_label(scoring)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    random_line = ax.plot(
        random_line_iterations,
        random_line_errors,
        linewidth=2,
        label=f"Best {y_label.lower()}",
    )[0]
    ax.scatter(
        iterations,
        best_validation_errors,
        s=42,
        color=random_line.get_color(),
        zorder=3,
    )

    if bayesian_history:
        bayesian_iterations = [history[-1]["iteration"]]
        bayesian_validation_errors = [history[-1]["best_validation_error"]]
        bayesian_iterations.extend(
            record["iteration"] for record in bayesian_history
        )
        bayesian_validation_errors.extend(
            record["best_validation_error"] for record in bayesian_history
        )
        ax.plot(
            bayesian_iterations,
            bayesian_validation_errors,
            marker="o",
            markersize=3,
            linewidth=1,
            alpha=0.75,
            label=f"Bayesian best {y_label.lower()}",
        )

    ax.set_xlim(0.5, plot_history[-1]["iteration"] + 0.5)
    _annotate_stage_boundaries(ax, plot_history)
    ax.set_title("Search Validation Error")
    ax.set_xlabel("Search iteration")
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/plot-loss.png", dpi=300)
    plt.close(fig)



def plot_yy(y_pred:NDArray, y_true:NDArray, OUTPUT_DIR:str) -> None:
    if len(y_pred) != len(y_true):
        raise ValueError(f"Length of y_pred ({len(y_pred)}) does not match length of y_true ({len(y_true)}).")

    plot_min = min(np.min(y_true), np.min(y_pred))
    plot_max = max(np.max(y_true), np.max(y_pred))
    plt.scatter(y_true, y_pred, s=2, color="goldenrod", label="Predictions")
    plt.plot([plot_min, plot_max], [plot_min, plot_max], color="black", linestyle=":", label="y = x")
    plt.legend()
    plt.xlabel("True Target Value")
    plt.ylabel("CKRR Prediction")
    plt.title("Target Values Versus CKRR Predictions")
    plt.savefig(f"{OUTPUT_DIR}/plot-yy.png", dpi=300, bbox_inches="tight")


def plot_error_histogram(y_pred: NDArray, y_true: NDArray, bins, OUTPUT_DIR:str):
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    y_true = np.asarray(y_true, dtype=float).reshape(-1)

    if len(y_pred) != len(y_true):
        raise ValueError(
            f"Length of y_pred ({len(y_pred)}) does not match length of "
            f"y_true ({len(y_true)})."
        )
    if len(y_pred) == 0:
        raise ValueError("Cannot generate an error histogram with zero samples.")
    if bins <= 0:
        raise ValueError(f"bins must be positive, got {bins}.")

    errors = y_pred - y_true
    error_mean = float(np.mean(errors))
    error_std = float(np.std(errors))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(
        errors,
        bins=bins,
        density=True,
        alpha=0.65,
        color="steelblue",
        edgecolor="white",
        label="Prediction errors",
    )

    x = np.linspace(np.min(errors), np.max(errors), 300)
    normal_pdf = (
        np.exp(-0.5 * ((x - error_mean) / error_std) ** 2)
        / (error_std * np.sqrt(2.0 * np.pi)))

    ax.plot(
        x,
        normal_pdf,
        color="black",
        linewidth=2,
        label=f"Normal fit (mu={error_mean:.3g}, sigma={error_std:.3g})")

    ax.axvline(0.0, color="red", linestyle=":", linewidth=1)
    ax.set_xlabel("Prediction Error (y_pred - y_true)")
    ax.set_ylabel("Density")
    ax.set_title("Prediction Error Distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/plot-error_histogram.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
