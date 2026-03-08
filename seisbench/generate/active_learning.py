from __future__ import annotations

import csv
import hashlib
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn

STATE_SCHEMA_VERSION = 1
logger = logging.getLogger(__name__)


def _as_1d_index_array(
    indices: Sequence[int] | np.ndarray, dataset_size: int, name: str
) -> np.ndarray:
    """
    Convert input indices to a sorted unique int array and validate bounds.
    """
    arr = np.asarray(indices)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional sequence of indices.")

    if arr.size == 0:
        return np.array([], dtype=np.int64)

    try:
        idx = arr.astype(np.int64)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must only contain integer values.")

    if not np.all(arr == idx):
        raise ValueError(f"{name} must only contain integer values.")
    if np.any(idx < 0) or np.any(idx >= dataset_size):
        raise ValueError(
            f"{name} contains out-of-range values. Valid range is [0, {dataset_size - 1}]."
        )
    if np.unique(idx).size != idx.size:
        raise ValueError(f"{name} contains duplicate values.")

    return np.sort(idx)


@dataclass(frozen=True)
class RoundResult:
    round: int
    selected_indices: np.ndarray
    query_size: int
    labeled_size: int
    unlabeled_size: int
    score_mean: float | None
    score_std: float | None
    score_min: float | None
    score_max: float | None
    strategy: str

    def to_metrics_row(self) -> dict[str, Any]:
        return {
            "round": self.round,
            "query_size": self.query_size,
            "labeled_size": self.labeled_size,
            "unlabeled_size": self.unlabeled_size,
            "score_mean": self.score_mean,
            "score_std": self.score_std,
            "score_min": self.score_min,
            "score_max": self.score_max,
            "strategy": self.strategy,
        }


class ActiveLearningPool:
    """
    Active learning pool state for trace-level selection.

    :param dataset_size: Total number of examples in the pool.
    :param labeled_indices: Indices that are already labeled.
    :param unlabeled_indices: Explicit unlabeled indices. If None, uses the complement.
    :param seed: Optional RNG seed.
    """

    def __init__(
        self,
        dataset_size: int,
        labeled_indices: Sequence[int] = (),
        unlabeled_indices: Sequence[int] | None = None,
        seed: int | None = None,
    ) -> None:
        if dataset_size < 0:
            raise ValueError("dataset_size must be >= 0.")

        self.dataset_size = int(dataset_size)
        self.seed = int(seed) if seed is not None else None
        self.round = 0
        self.strategy_config: dict[str, Any] = {}
        self.history_metadata: list[dict[str, Any]] = []

        self.labeled_indices = _as_1d_index_array(
            labeled_indices, self.dataset_size, "labeled_indices"
        )

        if unlabeled_indices is None:
            all_idx = np.arange(self.dataset_size, dtype=np.int64)
            self.unlabeled_indices = np.setdiff1d(
                all_idx, self.labeled_indices, assume_unique=True
            )
        else:
            self.unlabeled_indices = _as_1d_index_array(
                unlabeled_indices, self.dataset_size, "unlabeled_indices"
            )

        self._validate_index_state()
        self._rng = np.random.default_rng(self.seed)

    @property
    def rng(self) -> np.random.Generator:
        return self._rng

    def _validate_index_state(self) -> None:
        overlap = np.intersect1d(
            self.labeled_indices, self.unlabeled_indices, assume_unique=True
        )
        if overlap.size > 0:
            raise ValueError("labeled_indices and unlabeled_indices must be disjoint.")

        covered = np.union1d(self.labeled_indices, self.unlabeled_indices)
        if covered.size != self.dataset_size:
            raise ValueError(
                "labeled_indices and unlabeled_indices must form a full partition "
                "of [0, dataset_size)."
            )

    def mark_labeled(self, selected_indices: Sequence[int] | np.ndarray) -> None:
        selected = _as_1d_index_array(
            selected_indices, self.dataset_size, "selected_indices"
        )
        if selected.size == 0:
            return

        is_member = np.isin(selected, self.unlabeled_indices, assume_unique=True)
        if not np.all(is_member):
            raise ValueError("selected_indices must be a subset of unlabeled_indices.")

        keep_mask = ~np.isin(self.unlabeled_indices, selected, assume_unique=True)
        self.unlabeled_indices = self.unlabeled_indices[keep_mask]
        self.labeled_indices = np.sort(np.concatenate([self.labeled_indices, selected]))
        self._validate_index_state()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": STATE_SCHEMA_VERSION,
            "dataset_size": self.dataset_size,
            "seed": self.seed,
            "round": self.round,
            "labeled_indices": self.labeled_indices.tolist(),
            "unlabeled_indices": self.unlabeled_indices.tolist(),
            "strategy_config": dict(self.strategy_config),
            "history_metadata": [dict(row) for row in self.history_metadata],
        }

    def save(self, path_json: str) -> None:
        path = Path(path_json)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=True)

    @classmethod
    def load(cls, path_json: str) -> ActiveLearningPool:
        path = Path(path_json)
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        schema = payload.get("schema_version", None)
        if schema != STATE_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported schema version '{schema}'. "
                f"Expected {STATE_SCHEMA_VERSION}."
            )

        pool = cls(
            dataset_size=int(payload["dataset_size"]),
            labeled_indices=payload["labeled_indices"],
            unlabeled_indices=payload["unlabeled_indices"],
            seed=payload.get("seed"),
        )
        pool.round = int(payload.get("round", 0))

        strategy_config = payload.get("strategy_config", {})
        if not isinstance(strategy_config, dict):
            raise ValueError("strategy_config must be a dict in state file.")
        pool.strategy_config = strategy_config

        history_metadata = payload.get("history_metadata", [])
        if not isinstance(history_metadata, list):
            raise ValueError("history_metadata must be a list in state file.")
        pool.history_metadata = [
            dict(entry) for entry in history_metadata if isinstance(entry, dict)
        ]

        return pool


class QueryStrategy(ABC):
    @property
    def name(self) -> str:
        return self.__class__.__name__

    def get_config(self) -> dict[str, Any]:
        return {"name": self.name}

    @abstractmethod
    def select(
        self,
        candidate_indices: np.ndarray,
        budget: int,
        scores: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Select candidate indices for labeling.
        """


class RandomQueryStrategy(QueryStrategy):
    def select(
        self,
        candidate_indices: np.ndarray,
        budget: int,
        scores: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        candidate_indices = np.asarray(candidate_indices, dtype=np.int64)
        if candidate_indices.ndim != 1:
            raise ValueError("candidate_indices must be one-dimensional.")
        if budget < 0:
            raise ValueError("budget must be >= 0.")

        budget = min(int(budget), candidate_indices.size)
        if budget == 0:
            return np.array([], dtype=np.int64)

        selected = rng.choice(candidate_indices, size=budget, replace=False)
        return np.sort(selected.astype(np.int64))


class UncertaintyQueryStrategy(QueryStrategy):
    _VALID_METHODS = ("least_confidence", "margin", "entropy")

    def __init__(
        self,
        method: Literal["least_confidence", "margin", "entropy"] = "least_confidence",
    ) -> None:
        if method not in self._VALID_METHODS:
            raise ValueError(
                f"Unknown uncertainty method '{method}'. "
                f"Choose one of {self._VALID_METHODS}."
            )
        self.method = method

    @property
    def name(self) -> str:
        return f"{self.__class__.__name__}({self.method})"

    def get_config(self) -> dict[str, Any]:
        return {"name": self.name, "method": self.method}

    @staticmethod
    def score_probabilities(
        probabilities: np.ndarray,
        method: Literal["least_confidence", "margin", "entropy"] = "least_confidence",
    ) -> np.ndarray:
        if method not in UncertaintyQueryStrategy._VALID_METHODS:
            raise ValueError(
                f"Unknown uncertainty method '{method}'. "
                f"Choose one of {UncertaintyQueryStrategy._VALID_METHODS}."
            )

        probs = np.asarray(probabilities, dtype=np.float64)
        if probs.ndim == 1:
            probs = probs[None, :]
        if probs.ndim != 2:
            raise ValueError("probabilities must be a 1D or 2D array.")
        if probs.shape[1] == 0:
            raise ValueError("probabilities must include at least one class.")

        probs = np.clip(probs, 1e-12, None)
        norm = probs / np.sum(probs, axis=1, keepdims=True)

        if method == "least_confidence":
            return 1.0 - np.max(norm, axis=1)

        if method == "margin":
            if norm.shape[1] < 2:
                raise ValueError(
                    "margin uncertainty requires at least two output classes."
                )
            sorted_probs = np.sort(norm, axis=1)
            margin = sorted_probs[:, -1] - sorted_probs[:, -2]
            return 1.0 - margin

        # method == "entropy"
        return -np.sum(norm * np.log(norm), axis=1)

    def select(
        self,
        candidate_indices: np.ndarray,
        budget: int,
        scores: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        del rng  # Deterministic selection for uncertainty strategy.

        candidate_indices = np.asarray(candidate_indices, dtype=np.int64)
        scores = np.asarray(scores, dtype=np.float64)

        if candidate_indices.ndim != 1:
            raise ValueError("candidate_indices must be one-dimensional.")
        if scores.ndim != 1:
            raise ValueError("scores must be one-dimensional.")
        if candidate_indices.size != scores.size:
            raise ValueError("scores length must match candidate_indices length.")
        if budget < 0:
            raise ValueError("budget must be >= 0.")

        budget = min(int(budget), candidate_indices.size)
        if budget == 0:
            return np.array([], dtype=np.int64)

        # Primary key: descending score; tie-break key: ascending original index value.
        order = np.lexsort((candidate_indices, -scores))
        selected = candidate_indices[order[:budget]]
        return np.sort(selected.astype(np.int64))


class ActiveLearningLoop:
    """
    Orchestrates active learning rounds.

    Callback contracts:
      - score_fn(indices) -> np.ndarray (higher means more uncertain)
      - label_fn(selected_indices) -> Any
      - fit_fn(pool, round_result) -> None (optional)
    """

    def __init__(
        self,
        pool: ActiveLearningPool,
        score_fn: Callable[[np.ndarray], np.ndarray],
        label_fn: Callable[[np.ndarray], Any],
        fit_fn: Callable[[ActiveLearningPool, RoundResult], None] | None = None,
        strategy: QueryStrategy | None = None,
        model: Any | None = None,
        pool_dataset: Any | None = None,
        pool_metadata: Any | None = None,
    ) -> None:
        if not callable(score_fn):
            raise TypeError("score_fn must be callable.")
        if not callable(label_fn):
            raise TypeError("label_fn must be callable.")
        if fit_fn is not None and not callable(fit_fn):
            raise TypeError("fit_fn must be callable if provided.")

        self.pool = pool
        self.score_fn = score_fn
        self.label_fn = label_fn
        self.fit_fn = fit_fn
        self.strategy = strategy if strategy is not None else RandomQueryStrategy()
        self.model = model
        self.pool_dataset = pool_dataset
        self.pool_metadata = pool_metadata
        self._history: list[RoundResult] = []
        self.pool.strategy_config = self.strategy.get_config()

    @property
    def history(self) -> tuple[RoundResult, ...]:
        return tuple(self._history)

    def _validate_scores(self, scores: np.ndarray, n_candidates: int) -> None:
        if scores.ndim != 1:
            raise ValueError("score_fn must return a one-dimensional numpy array.")
        if scores.size != n_candidates:
            raise ValueError(
                "score_fn output length must match the number of candidate indices."
            )
        if not np.all(np.isfinite(scores)):
            raise ValueError("score_fn returned non-finite scores.")

    @staticmethod
    def _validate_selection(
        selected: np.ndarray, candidates: np.ndarray, expected_size: int
    ) -> None:
        if selected.ndim != 1:
            raise ValueError("strategy.select must return a one-dimensional array.")
        if selected.size != expected_size:
            raise ValueError(
                "strategy.select returned an unexpected number of indices."
            )
        if np.unique(selected).size != selected.size:
            raise ValueError("strategy.select returned duplicate indices.")

        is_member = np.isin(selected, candidates, assume_unique=True)
        if not np.all(is_member):
            raise ValueError(
                "strategy.select returned indices outside the candidate set."
            )

    def _build_result(
        self, selected: np.ndarray, selected_scores: np.ndarray
    ) -> RoundResult:
        if selected_scores.size == 0:
            score_mean = None
            score_std = None
            score_min = None
            score_max = None
        else:
            score_mean = float(np.mean(selected_scores))
            score_std = float(np.std(selected_scores))
            score_min = float(np.min(selected_scores))
            score_max = float(np.max(selected_scores))

        return RoundResult(
            round=self.pool.round,
            selected_indices=selected.copy(),
            query_size=int(selected.size),
            labeled_size=int(self.pool.labeled_indices.size),
            unlabeled_size=int(self.pool.unlabeled_indices.size),
            score_mean=score_mean,
            score_std=score_std,
            score_min=score_min,
            score_max=score_max,
            strategy=self.strategy.name,
        )

    def _record_result(self, result: RoundResult) -> None:
        self._history.append(result)
        self.pool.history_metadata.append(result.to_metrics_row())
        self.pool.strategy_config = self.strategy.get_config()

    def _selected_scores_from_strategy(self, selected: np.ndarray) -> np.ndarray:
        score_map = getattr(self.strategy, "last_uncertainty_scores", None)
        if not isinstance(score_map, Mapping):
            return np.array([], dtype=np.float64)

        values: list[float] = []
        for idx in selected.tolist():
            if int(idx) in score_map:
                values.append(float(score_map[int(idx)]))
        return np.asarray(values, dtype=np.float64)

    def run_round(self, budget: int) -> RoundResult:
        if budget < 0:
            raise ValueError("budget must be >= 0.")

        candidates = self.pool.unlabeled_indices.copy()
        budget = min(int(budget), int(candidates.size))

        if budget == 0:
            self.pool.round += 1
            result = self._build_result(
                np.array([], dtype=np.int64),
                np.array([], dtype=np.float64),
            )
            self._record_result(result)
            if self.fit_fn is not None:
                self.fit_fn(self.pool, result)
            return result

        use_pool_strategy = (
            hasattr(self.strategy, "select_batch")
            and self.model is not None
            and self.pool_dataset is not None
        )

        if use_pool_strategy:
            selected = np.asarray(
                self.strategy.select_batch(  # type: ignore[attr-defined]
                    model=self.model,
                    unlabeled_dataset=self.pool_dataset,
                    batch_size=budget,
                    metadata=self.pool_metadata,
                    unlabeled_indices=candidates,
                ),
                dtype=np.int64,
            )
            self._validate_selection(selected, candidates, expected_size=budget)
            selected_scores = self._selected_scores_from_strategy(selected)
        else:
            scores = np.asarray(self.score_fn(candidates), dtype=np.float64)
            self._validate_scores(scores, n_candidates=candidates.size)

            selected = np.asarray(
                self.strategy.select(candidates, budget, scores, self.pool.rng),
                dtype=np.int64,
            )
            self._validate_selection(selected, candidates, expected_size=budget)

            score_lookup = {
                int(idx): float(score)
                for idx, score in zip(candidates.tolist(), scores.tolist())
            }
            selected_scores = np.array([score_lookup[int(idx)] for idx in selected])

        # Atomic state update: labeling callback happens before mutating pool indices.
        self.label_fn(selected.copy())

        self.pool.mark_labeled(selected)
        self.pool.round += 1

        result = self._build_result(selected, selected_scores)
        self._record_result(result)

        if self.fit_fn is not None:
            self.fit_fn(self.pool, result)

        return result

    @staticmethod
    def _serialize_metric_value(value: Any) -> Any:
        if value is None:
            return ""
        if isinstance(value, np.generic):
            return value.item()
        return value

    def export_metrics(self, path_csv: str) -> None:
        columns = [
            "round",
            "query_size",
            "labeled_size",
            "unlabeled_size",
            "score_mean",
            "score_std",
            "score_min",
            "score_max",
            "strategy",
        ]

        path = Path(path_csv)
        path.parent.mkdir(parents=True, exist_ok=True)

        existing_rounds: set[int] = set()
        if path.exists():
            with path.open("r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("round", "") != "":
                        existing_rounds.add(int(row["round"]))

        mode = "a" if path.exists() else "w"
        with path.open(mode, newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            if mode == "w":
                writer.writeheader()

            for row in self.pool.history_metadata:
                round_id = int(row["round"])
                if round_id in existing_rounds:
                    continue

                writer.writerow(
                    {
                        key: self._serialize_metric_value(row.get(key, ""))
                        for key in columns
                    }
                )
                existing_rounds.add(round_id)


def robust_minmax(
    values: np.ndarray,
    lower_percentile: float = 5.0,
    upper_percentile: float = 95.0,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Robust min-max normalization that clips with percentiles before scaling.
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return arr
    if arr.ndim != 1:
        arr = arr.reshape(-1)

    lo = np.percentile(arr, lower_percentile)
    hi = np.percentile(arr, upper_percentile)
    if hi - lo < eps:
        return np.zeros_like(arr)

    clipped = np.clip(arr, lo, hi)
    return (clipped - lo) / (hi - lo + eps)


def _extract_batch_input(batch: Any) -> np.ndarray | torch.Tensor:
    """
    Best-effort extraction of waveform input tensor/array from common batch shapes.
    """
    if isinstance(batch, dict):
        if "X" in batch:
            return batch["X"]
        return next(iter(batch.values()))

    if isinstance(batch, (list, tuple)):
        if len(batch) == 0:
            raise ValueError("Encountered empty batch sequence.")
        first = batch[0]
        if (
            isinstance(first, (list, tuple))
            and len(first) == 2
            and not torch.is_tensor(first)
        ):
            return first[0]
        return first

    return batch


def _as_numpy(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    raise TypeError(f"Unsupported array type: {type(x)}")


def _extract_model_output_array(output: Any) -> np.ndarray:
    if isinstance(output, np.ndarray):
        return output
    if torch.is_tensor(output):
        return output.detach().cpu().numpy()
    if isinstance(output, dict):
        if "y_pred" in output:
            return _extract_model_output_array(output["y_pred"])
        return _extract_model_output_array(next(iter(output.values())))
    if isinstance(output, (list, tuple)):
        if len(output) == 0:
            raise ValueError("Model output is an empty sequence.")
        return _extract_model_output_array(output[0])
    raise TypeError(f"Unsupported model output type: {type(output)}")


def _resolve_model_device(
    model: Any, device: str | torch.device | None
) -> torch.device:
    if device is not None:
        return torch.device(device)

    if hasattr(model, "device"):
        try:
            return torch.device(model.device)
        except Exception:
            pass

    try:
        return next(model.parameters()).device
    except (StopIteration, AttributeError, TypeError):
        return torch.device("cpu")


def _stack_and_pad_arrays(arrays: Sequence[np.ndarray]) -> np.ndarray:
    if len(arrays) == 0:
        return np.zeros((0, 0), dtype=np.float32)

    np_arrays = [np.asarray(arr, dtype=np.float32) for arr in arrays]
    ndim = np_arrays[0].ndim
    if any(arr.ndim != ndim for arr in np_arrays):
        raise ValueError(
            "All waveform samples must share the same number of dimensions."
        )

    max_shape = [max(arr.shape[d] for arr in np_arrays) for d in range(ndim)]
    batch = np.zeros((len(np_arrays), *max_shape), dtype=np.float32)

    for i, arr in enumerate(np_arrays):
        slices = (i,) + tuple(slice(0, s) for s in arr.shape)
        batch[slices] = arr

    return batch


def _batched_dataset_inputs(
    dataset: Any, indices: np.ndarray, batch_size: int
) -> Iterable[tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]]:
    for p0 in range(0, len(indices), batch_size):
        batch_indices = indices[p0 : p0 + batch_size]
        waveforms = []
        metadata_rows: list[dict[str, Any]] = []
        for idx in batch_indices:
            waveform, metadata = dataset.get_sample(int(idx))
            waveforms.append(np.asarray(waveform))
            metadata_rows.append(dict(metadata))
        yield batch_indices, _stack_and_pad_arrays(waveforms), metadata_rows


def _resolve_indices(
    dataset: Any | None = None,
    indices: Sequence[int] | np.ndarray | None = None,
    waveforms: Sequence[np.ndarray] | np.ndarray | None = None,
) -> np.ndarray:
    if indices is not None:
        return np.asarray(indices, dtype=np.int64)

    if dataset is not None:
        return np.arange(len(dataset), dtype=np.int64)

    if waveforms is not None:
        return np.arange(len(waveforms), dtype=np.int64)

    raise ValueError("Unable to infer indices. Provide dataset, indices, or waveforms.")


def _model_predict_numpy(
    model: Any,
    batch_np: np.ndarray,
    device: str | torch.device | None = None,
    prediction_fn: Callable[[Any, torch.Tensor], Any] | None = None,
) -> np.ndarray:
    model_device = _resolve_model_device(model, device)
    batch_tensor = torch.as_tensor(batch_np, dtype=torch.float32, device=model_device)

    with torch.no_grad():
        if prediction_fn is None:
            output = model(batch_tensor)
        else:
            output = prediction_fn(model, batch_tensor)
    return np.asarray(_extract_model_output_array(output), dtype=np.float64)


def _dropout_modules() -> tuple[type[nn.Module], ...]:
    return (
        nn.Dropout,
        nn.Dropout1d,
        nn.Dropout2d,
        nn.Dropout3d,
        nn.AlphaDropout,
        nn.FeatureAlphaDropout,
    )


@dataclass(frozen=True)
class FilterResult:
    valid_indices: np.ndarray
    rejected_indices: np.ndarray
    reason_codes: dict[int, list[str]]
    valid_mask: np.ndarray


class PoolFilter:
    """
    Pre-query filter for low-value and problematic samples.
    """

    def __init__(
        self,
        enabled: bool = True,
        low_energy_enabled: bool = True,
        rms_threshold: float = 1e-4,
        abnormal_enabled: bool = True,
        max_abs_threshold: float = 1e3,
        flat_std_threshold: float = 1e-8,
        clipped_fraction_threshold: float = 0.2,
        duplicate_enabled: bool = True,
        duplicate_method: Literal[
            "exact_hash", "embedding_cosine", "metadata"
        ] = "exact_hash",
        similarity_threshold: float = 0.999,
        metadata_duplicate_keys: Sequence[str] = (
            "source_id",
            "event_id",
            "trace_name",
            "station_code",
        ),
    ) -> None:
        self.enabled = enabled
        self.low_energy_enabled = low_energy_enabled
        self.rms_threshold = float(rms_threshold)
        self.abnormal_enabled = abnormal_enabled
        self.max_abs_threshold = float(max_abs_threshold)
        self.flat_std_threshold = float(flat_std_threshold)
        self.clipped_fraction_threshold = float(clipped_fraction_threshold)
        self.duplicate_enabled = duplicate_enabled
        self.duplicate_method = duplicate_method
        self.similarity_threshold = float(similarity_threshold)
        self.metadata_duplicate_keys = tuple(metadata_duplicate_keys)

    def _append_reason(
        self, reasons: dict[int, list[str]], idx: int, reason: str
    ) -> None:
        reasons.setdefault(int(idx), []).append(reason)

    @staticmethod
    def _rms(x: np.ndarray) -> float:
        arr = np.asarray(x, dtype=np.float64)
        return float(np.sqrt(np.mean(arr**2)))

    def _is_abnormal(self, x: np.ndarray) -> list[str]:
        reasons = []
        arr = np.asarray(x, dtype=np.float64)
        finite_mask = np.isfinite(arr)

        if not np.all(finite_mask):
            reasons.append("abnormal_nan_or_inf")
            finite_arr = arr[finite_mask]
            if finite_arr.size == 0:
                return reasons
            arr = finite_arr

        if np.allclose(arr, 0.0):
            reasons.append("abnormal_all_zero")

        if float(np.std(arr)) < self.flat_std_threshold:
            reasons.append("abnormal_flat_signal")

        max_abs = float(np.max(np.abs(arr))) if arr.size > 0 else 0.0
        if max_abs > self.max_abs_threshold:
            reasons.append("abnormal_extreme_spike")

        if max_abs > 0.0:
            clipped_fraction = float(np.mean(np.isclose(np.abs(arr), max_abs)))
            if clipped_fraction >= self.clipped_fraction_threshold:
                reasons.append("abnormal_clipped")

        return reasons

    @staticmethod
    def _metadata_row_lookup(
        metadata: Any,
        local_pos: int,
        global_index: int,
    ) -> dict[str, Any]:
        if metadata is None:
            return {}

        if hasattr(metadata, "iloc"):
            try:
                if 0 <= global_index < len(metadata):
                    return dict(metadata.iloc[int(global_index)].to_dict())
            except Exception:
                pass
            return dict(metadata.iloc[int(local_pos)].to_dict())

        if isinstance(metadata, list):
            if len(metadata) == 0:
                return {}
            if 0 <= global_index < len(metadata):
                return dict(metadata[int(global_index)])
            return dict(metadata[int(local_pos)])

        if isinstance(metadata, dict):
            out = {}
            for key, value in metadata.items():
                if isinstance(value, (list, tuple, np.ndarray)):
                    if len(value) == 0:
                        out[key] = None
                    elif 0 <= global_index < len(value):
                        out[key] = value[int(global_index)]
                    else:
                        out[key] = value[int(local_pos)]
                else:
                    out[key] = value
            return out

        return {}

    @staticmethod
    def _exact_hash(x: np.ndarray) -> str:
        arr = np.ascontiguousarray(np.asarray(x))
        return hashlib.sha1(arr.view(np.uint8)).hexdigest()

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
        num = float(np.dot(a, b))
        den = float(np.linalg.norm(a) * np.linalg.norm(b)) + eps
        return num / den

    def apply(
        self,
        dataset: Any | None = None,
        metadata: Any | None = None,
        embeddings: np.ndarray | None = None,
        indices: Sequence[int] | np.ndarray | None = None,
        waveforms: Sequence[np.ndarray] | np.ndarray | None = None,
    ) -> FilterResult:
        global_indices = _resolve_indices(
            dataset=dataset, indices=indices, waveforms=waveforms
        )
        global_indices = np.asarray(global_indices, dtype=np.int64)

        if not self.enabled:
            return FilterResult(
                valid_indices=global_indices.copy(),
                rejected_indices=np.array([], dtype=np.int64),
                reason_codes={},
                valid_mask=np.ones(global_indices.size, dtype=bool),
            )

        local_waveforms: list[np.ndarray] = []
        local_metadata_rows: list[dict[str, Any]] = []

        if waveforms is not None:
            if isinstance(waveforms, np.ndarray):
                local_waveforms = [np.asarray(w) for w in waveforms]
            else:
                local_waveforms = [np.asarray(w) for w in waveforms]

            for p, gidx in enumerate(global_indices.tolist()):
                local_metadata_rows.append(
                    self._metadata_row_lookup(
                        metadata, local_pos=p, global_index=int(gidx)
                    )
                )

        elif dataset is not None:
            for p, gidx in enumerate(global_indices.tolist()):
                waveform, row = dataset.get_sample(int(gidx))
                local_waveforms.append(np.asarray(waveform))
                if metadata is None:
                    local_metadata_rows.append(dict(row))
                else:
                    local_metadata_rows.append(
                        self._metadata_row_lookup(
                            metadata, local_pos=p, global_index=int(gidx)
                        )
                    )
        else:
            raise ValueError("PoolFilter requires either dataset or waveforms.")

        if len(local_waveforms) != len(global_indices):
            raise ValueError("Number of waveforms must match number of indices.")

        valid_mask = np.ones(len(global_indices), dtype=bool)
        reason_codes: dict[int, list[str]] = {}

        for p, gidx in enumerate(global_indices.tolist()):
            x = local_waveforms[p]

            if self.low_energy_enabled and self._rms(x) < self.rms_threshold:
                valid_mask[p] = False
                self._append_reason(reason_codes, int(gidx), "low_energy")

            if self.abnormal_enabled:
                for reason in self._is_abnormal(x):
                    valid_mask[p] = False
                    self._append_reason(reason_codes, int(gidx), reason)

        if self.duplicate_enabled:
            if self.duplicate_method == "exact_hash":
                seen_hashes: dict[str, int] = {}
                for p, gidx in enumerate(global_indices.tolist()):
                    if not valid_mask[p]:
                        continue
                    sig = self._exact_hash(local_waveforms[p])
                    if sig in seen_hashes:
                        valid_mask[p] = False
                        self._append_reason(reason_codes, int(gidx), "duplicate_exact")
                    else:
                        seen_hashes[sig] = int(gidx)

            elif self.duplicate_method == "embedding_cosine":
                if embeddings is None:
                    raise ValueError(
                        "duplicate_method='embedding_cosine' requires embeddings."
                    )
                emb = np.asarray(embeddings, dtype=np.float64)
                if emb.shape[0] != len(global_indices):
                    raise ValueError(
                        "Embeddings must align with provided indices in the same order."
                    )
                kept_positions: list[int] = []
                for p, gidx in enumerate(global_indices.tolist()):
                    if not valid_mask[p]:
                        continue
                    v = emb[p]
                    is_duplicate = False
                    for kp in kept_positions:
                        sim = self._cosine_similarity(v, emb[kp])
                        if sim >= self.similarity_threshold:
                            is_duplicate = True
                            break
                    if is_duplicate:
                        valid_mask[p] = False
                        self._append_reason(reason_codes, int(gidx), "duplicate_near")
                    else:
                        kept_positions.append(p)

            elif self.duplicate_method == "metadata":
                signatures: dict[tuple[Any, ...], int] = {}
                for p, gidx in enumerate(global_indices.tolist()):
                    if not valid_mask[p]:
                        continue
                    row = local_metadata_rows[p]
                    available_keys = [
                        k for k in self.metadata_duplicate_keys if k in row
                    ]
                    if len(available_keys) == 0:
                        continue
                    sig = tuple(row[k] for k in available_keys)
                    if sig in signatures:
                        valid_mask[p] = False
                        self._append_reason(
                            reason_codes, int(gidx), "duplicate_metadata"
                        )
                    else:
                        signatures[sig] = int(gidx)
            else:
                raise ValueError(f"Unknown duplicate method '{self.duplicate_method}'.")

        valid_indices = global_indices[valid_mask]
        rejected_indices = global_indices[~valid_mask]

        return FilterResult(
            valid_indices=np.asarray(valid_indices, dtype=np.int64),
            rejected_indices=np.asarray(rejected_indices, dtype=np.int64),
            reason_codes=reason_codes,
            valid_mask=valid_mask,
        )


def _normalize_probabilities(
    values: np.ndarray, axis: int = -1, eps: float = 1e-12
) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return arr

    arr = np.clip(arr, a_min=0.0, a_max=None)
    denom = np.sum(arr, axis=axis, keepdims=True)
    needs_softmax = np.any(denom <= eps) or np.any(np.abs(denom - 1.0) > 1e-3)

    if needs_softmax:
        shifted = arr - np.max(arr, axis=axis, keepdims=True)
        exp = np.exp(shifted)
        return exp / (np.sum(exp, axis=axis, keepdims=True) + eps)

    return arr / (denom + eps)


def _safe_log(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return np.log(np.clip(values, eps, 1.0))


def _predict_model_on_dataset(
    model: Any,
    dataset: Any | None = None,
    indices: Sequence[int] | np.ndarray | None = None,
    dataloader: Any | None = None,
    batch_size: int = 64,
    device: str | torch.device | None = None,
    prediction_fn: Callable[[Any, torch.Tensor], Any] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    outputs: list[np.ndarray] = []
    output_indices: list[np.ndarray] = []

    if dataloader is not None:
        cursor = 0
        for batch in dataloader:
            x = _extract_batch_input(batch)
            batch_np = _as_numpy(x).astype(np.float32, copy=False)
            pred = _model_predict_numpy(
                model=model,
                batch_np=batch_np,
                device=device,
                prediction_fn=prediction_fn,
            )
            outputs.append(np.asarray(pred, dtype=np.float64))
            n_batch = int(pred.shape[0])
            output_indices.append(np.arange(cursor, cursor + n_batch, dtype=np.int64))
            cursor += n_batch
    else:
        if dataset is None:
            raise ValueError("Provide either dataloader or dataset.")
        resolved_indices = _resolve_indices(dataset=dataset, indices=indices)
        for batch_indices, batch_np, _ in _batched_dataset_inputs(
            dataset=dataset,
            indices=resolved_indices,
            batch_size=batch_size,
        ):
            pred = _model_predict_numpy(
                model=model,
                batch_np=batch_np,
                device=device,
                prediction_fn=prediction_fn,
            )
            outputs.append(np.asarray(pred, dtype=np.float64))
            output_indices.append(np.asarray(batch_indices, dtype=np.int64))

    if len(outputs) == 0:
        return np.zeros((0, 0), dtype=np.float64), np.array([], dtype=np.int64)

    return np.concatenate(outputs, axis=0), np.concatenate(output_indices, axis=0)


class EmbeddingExtractor:
    """
    Reusable embedding extractor with optional disk caching.
    """

    CACHE_SCHEMA_VERSION = 1

    def __init__(
        self,
        batch_size: int = 64,
        device: str | torch.device | None = None,
    ) -> None:
        self.batch_size = int(batch_size)
        self.device = torch.device(device) if device is not None else None

    @staticmethod
    def _model_signature(model: Any) -> str:
        name = model.__class__.__name__
        shapes = []
        try:
            for p in model.parameters():
                shapes.append(tuple(p.shape))
        except Exception:
            pass
        payload = {"name": name, "shapes": shapes}
        return hashlib.sha1(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()

    @staticmethod
    def _indices_signature(indices: np.ndarray) -> str:
        payload = np.asarray(indices, dtype=np.int64).tobytes()
        return hashlib.sha1(payload).hexdigest()

    def _cache_meta(
        self,
        layer: str,
        model: Any,
        indices: np.ndarray,
    ) -> dict[str, Any]:
        return {
            "schema_version": self.CACHE_SCHEMA_VERSION,
            "layer": layer,
            "model_signature": self._model_signature(model),
            "indices_signature": self._indices_signature(indices),
            "n_samples": int(indices.size),
        }

    @staticmethod
    def _cache_meta_path(cache_path: Path) -> Path:
        return cache_path.with_suffix(cache_path.suffix + ".meta.json")

    def _load_cache(
        self,
        cache_path: Path,
        expected_meta: Mapping[str, Any],
    ) -> np.ndarray | None:
        meta_path = self._cache_meta_path(cache_path)
        if not cache_path.exists() or not meta_path.exists():
            return None

        try:
            with meta_path.open("r", encoding="utf-8") as f:
                observed_meta = json.load(f)
        except Exception:
            return None

        for key, expected_value in expected_meta.items():
            if observed_meta.get(key) != expected_value:
                return None

        try:
            embeddings = np.load(cache_path)
        except Exception:
            return None
        return np.asarray(embeddings, dtype=np.float64)

    def _save_cache(
        self,
        cache_path: Path,
        embeddings: np.ndarray,
        meta: Mapping[str, Any],
    ) -> None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, np.asarray(embeddings, dtype=np.float64))
        with self._cache_meta_path(cache_path).open("w", encoding="utf-8") as f:
            json.dump(dict(meta), f, indent=2)

    @staticmethod
    def _reshape_embedding(embedding: np.ndarray) -> np.ndarray:
        arr = np.asarray(embedding, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr[:, None]
        if arr.ndim > 2:
            arr = arr.reshape(arr.shape[0], -1)
        return arr

    @staticmethod
    def _find_named_module(model: Any, layer: str) -> nn.Module | None:
        try:
            for name, module in model.named_modules():
                if name == layer:
                    return module
        except Exception:
            return None
        return None

    def _extract_batch_embeddings(
        self,
        model: Any,
        batch_np: np.ndarray,
        layer: str,
        device: str | torch.device | None,
        prediction_fn: Callable[[Any, torch.Tensor], Any] | None = None,
    ) -> np.ndarray:
        model_device = _resolve_model_device(model, device or self.device)
        batch_tensor = torch.as_tensor(
            batch_np, dtype=torch.float32, device=model_device
        )

        if hasattr(model, "extract_embedding") and callable(model.extract_embedding):
            with torch.no_grad():
                output = model.extract_embedding(batch_tensor)  # type: ignore[attr-defined]
            return self._reshape_embedding(_extract_model_output_array(output))

        module = self._find_named_module(model, layer)
        if module is not None:
            captured: dict[str, np.ndarray] = {}

            def _hook(_module: nn.Module, _inputs: Any, output: Any) -> None:
                captured["embedding"] = _extract_model_output_array(output)

            handle = module.register_forward_hook(_hook)
            try:
                with torch.no_grad():
                    if prediction_fn is None:
                        model(batch_tensor)
                    else:
                        prediction_fn(model, batch_tensor)
            finally:
                handle.remove()

            if "embedding" not in captured:
                raise RuntimeError(
                    f"Failed to capture embeddings from layer '{layer}'."
                )
            return self._reshape_embedding(captured["embedding"])

        with torch.no_grad():
            if prediction_fn is None:
                output = model(batch_tensor)
            else:
                output = prediction_fn(model, batch_tensor)
        return self._reshape_embedding(_extract_model_output_array(output))

    def extract(
        self,
        model: Any,
        dataloader: Any | None = None,
        dataset: Any | None = None,
        indices: Sequence[int] | np.ndarray | None = None,
        layer: str = "encoder_global_pool",
        cache_path: str | None = None,
        use_cache: bool = True,
        batch_size: int | None = None,
        device: str | torch.device | None = None,
        prediction_fn: Callable[[Any, torch.Tensor], Any] | None = None,
    ) -> np.ndarray:
        resolved_batch_size = int(batch_size or self.batch_size)

        if dataloader is None and dataset is None:
            raise ValueError("Provide either dataloader or dataset.")

        if dataloader is not None:
            outputs: list[np.ndarray] = []
            for batch in dataloader:
                x = _extract_batch_input(batch)
                batch_np = _as_numpy(x).astype(np.float32, copy=False)
                emb = self._extract_batch_embeddings(
                    model=model,
                    batch_np=batch_np,
                    layer=layer,
                    device=device,
                    prediction_fn=prediction_fn,
                )
                outputs.append(np.asarray(emb, dtype=np.float64))
            if len(outputs) == 0:
                return np.zeros((0, 0), dtype=np.float64)
            return np.concatenate(outputs, axis=0)

        resolved_indices = np.asarray(
            _resolve_indices(dataset=dataset, indices=indices), dtype=np.int64
        )
        expected_meta = self._cache_meta(
            layer=layer, model=model, indices=resolved_indices
        )

        cache_file: Path | None = None
        if cache_path is not None:
            cache_file = Path(cache_path)
            if cache_file.suffix == "":
                cache_file = cache_file.with_suffix(".npy")

        if use_cache and cache_file is not None:
            cached = self._load_cache(cache_file, expected_meta=expected_meta)
            if cached is not None:
                if cached.shape[0] != resolved_indices.size:
                    raise ValueError(
                        "Cached embeddings do not match requested indices."
                    )
                return cached

        was_training = bool(getattr(model, "training", False))
        try:
            if hasattr(model, "eval"):
                model.eval()
            outputs = []
            for _, batch_np, _ in _batched_dataset_inputs(
                dataset=dataset,
                indices=resolved_indices,
                batch_size=resolved_batch_size,
            ):
                emb = self._extract_batch_embeddings(
                    model=model,
                    batch_np=batch_np,
                    layer=layer,
                    device=device,
                    prediction_fn=prediction_fn,
                )
                outputs.append(np.asarray(emb, dtype=np.float64))
        finally:
            if was_training and hasattr(model, "train"):
                model.train()

        embeddings = (
            np.zeros((0, 0), dtype=np.float64)
            if len(outputs) == 0
            else np.concatenate(outputs, axis=0)
        )

        if embeddings.shape[0] != resolved_indices.size:
            raise ValueError(
                "Extracted embeddings are not aligned with requested indices."
            )

        if cache_file is not None:
            self._save_cache(cache_file, embeddings=embeddings, meta=expected_meta)

        return embeddings


class SeismicUncertaintyMetrics:
    """
    Uncertainty metrics for phase-picking style probability curves.
    """

    @staticmethod
    def predictive_entropy(probabilities: np.ndarray, axis: int = -1) -> np.ndarray:
        probs = _normalize_probabilities(probabilities, axis=axis)
        return -np.sum(probs * _safe_log(probs), axis=axis)

    @staticmethod
    def pick_time_variance(predictions: np.ndarray) -> np.ndarray:
        arr = np.asarray(predictions, dtype=np.float64)
        if arr.ndim < 2:
            raise ValueError("pick_time_variance requires [n_passes, n_samples, ...].")
        peak_idx = np.argmax(arr, axis=-1).astype(np.float64)
        return np.var(peak_idx, axis=0)

    @staticmethod
    def peak_probability_variance(predictions: np.ndarray) -> np.ndarray:
        arr = np.asarray(predictions, dtype=np.float64)
        if arr.ndim < 2:
            raise ValueError(
                "peak_probability_variance requires [n_passes, n_samples, ...]."
            )
        peak_prob = np.max(arr, axis=-1)
        return np.var(peak_prob, axis=0)

    @staticmethod
    def per_timestep_variance(predictions: np.ndarray) -> np.ndarray:
        arr = np.asarray(predictions, dtype=np.float64)
        if arr.ndim < 2:
            raise ValueError(
                "per_timestep_variance requires [n_passes, n_samples, ...]."
            )
        var_curve = np.var(arr, axis=0)
        if var_curve.ndim == 1:
            return var_curve
        return var_curve.reshape(var_curve.shape[0], -1).mean(axis=1)

    @staticmethod
    def multi_peak_ambiguity(probability_curves: np.ndarray) -> np.ndarray:
        arr = np.asarray(probability_curves, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.shape[-1] < 2:
            return np.zeros(arr.shape[0], dtype=np.float64)

        sorted_vals = np.sort(arr, axis=-1)
        top1 = sorted_vals[..., -1]
        top2 = sorted_vals[..., -2]
        margin = (top1 - top2) / (np.abs(top1) + 1e-12)
        ambiguity = 1.0 - np.clip(margin, 0.0, 1.0)
        if ambiguity.ndim > 1:
            ambiguity = ambiguity.reshape(ambiguity.shape[0], -1).mean(axis=1)
        return ambiguity

    @staticmethod
    def peak_sharpness_uncertainty(probability_curves: np.ndarray) -> np.ndarray:
        curves = np.asarray(probability_curves, dtype=np.float64)
        if curves.ndim == 1:
            curves = curves[None, :]
        if curves.ndim > 2:
            curves = curves.reshape(curves.shape[0], -1)

        scores = np.zeros(curves.shape[0], dtype=np.float64)
        for i, c in enumerate(curves):
            peak = int(np.argmax(c))
            peak_value = float(c[peak])
            if peak_value <= 0:
                scores[i] = 1.0
                continue
            half = 0.5 * peak_value
            left = peak
            while left > 0 and c[left] >= half:
                left -= 1
            right = peak
            last = len(c) - 1
            while right < last and c[right] >= half:
                right += 1
            width = right - left
            scores[i] = float(width) / float(max(len(c), 1))
        return np.clip(scores, 0.0, 1.0)

    @staticmethod
    def mutual_information(prob_predictions: np.ndarray) -> np.ndarray:
        arr = np.asarray(prob_predictions, dtype=np.float64)
        if arr.ndim < 3:
            raise ValueError(
                "mutual_information requires [n_passes, n_samples, n_classes/time]."
            )
        probs = _normalize_probabilities(arr, axis=-1)
        mean_prob = np.mean(probs, axis=0)
        entropy_mean = SeismicUncertaintyMetrics.predictive_entropy(mean_prob, axis=-1)
        entropy_each = SeismicUncertaintyMetrics.predictive_entropy(probs, axis=-1)
        if entropy_each.ndim > 2:
            entropy_each = entropy_each.reshape(
                entropy_each.shape[0], entropy_each.shape[1], -1
            ).mean(axis=-1)
        expected_entropy = np.mean(entropy_each, axis=0)
        return entropy_mean - expected_entropy

    @staticmethod
    def js_divergence(prob_predictions: np.ndarray) -> np.ndarray:
        arr = np.asarray(prob_predictions, dtype=np.float64)
        if arr.ndim < 3:
            raise ValueError(
                "js_divergence requires [n_members, n_samples, n_classes/time]."
            )
        probs = _normalize_probabilities(arr, axis=-1)
        mean_prob = np.mean(probs, axis=0)
        kl_terms = probs * (_safe_log(probs) - _safe_log(mean_prob[None, ...]))
        kl = np.sum(kl_terms, axis=-1)
        if kl.ndim > 2:
            kl = kl.reshape(kl.shape[0], kl.shape[1], -1).mean(axis=-1)
        return np.mean(kl, axis=0)

    @staticmethod
    def mean_pairwise_kl(prob_predictions: np.ndarray) -> np.ndarray:
        arr = np.asarray(prob_predictions, dtype=np.float64)
        if arr.ndim < 3:
            raise ValueError(
                "mean_pairwise_kl requires [n_members, n_samples, n_classes/time]."
            )
        probs = _normalize_probabilities(arr, axis=-1)
        n_members = probs.shape[0]
        if n_members < 2:
            return np.zeros(probs.shape[1], dtype=np.float64)

        acc = np.zeros(probs.shape[1], dtype=np.float64)
        n_pairs = 0
        for i in range(n_members):
            for j in range(i + 1, n_members):
                p = probs[i]
                q = probs[j]
                kl = np.sum(p * (_safe_log(p) - _safe_log(q)), axis=-1)
                if kl.ndim > 1:
                    kl = kl.reshape(kl.shape[0], -1).mean(axis=1)
                acc += kl
                n_pairs += 1
        return acc / float(max(n_pairs, 1))

    @staticmethod
    def vote_entropy(prob_predictions: np.ndarray) -> np.ndarray:
        arr = np.asarray(prob_predictions, dtype=np.float64)
        if arr.ndim < 3:
            raise ValueError(
                "vote_entropy requires [n_members, n_samples, n_classes/time]."
            )
        votes = np.argmax(arr, axis=-1)
        if votes.ndim > 2:
            votes = votes.reshape(votes.shape[0], votes.shape[1], -1)[:, :, 0]

        n_members = votes.shape[0]
        n_samples = votes.shape[1]
        max_vote = int(np.max(votes)) if votes.size else 0
        out = np.zeros(n_samples, dtype=np.float64)
        for i in range(n_samples):
            counts = np.bincount(votes[:, i], minlength=max_vote + 1).astype(np.float64)
            probs = counts / float(max(n_members, 1))
            nz = probs > 0
            out[i] = -np.sum(probs[nz] * np.log(probs[nz]))
        return out

    @staticmethod
    def compute(metric: str, predictions: np.ndarray) -> np.ndarray:
        metric_name = metric.lower()
        if metric_name == "predictive_entropy":
            mean_prediction = (
                np.mean(predictions, axis=0) if predictions.ndim >= 3 else predictions
            )
            values = SeismicUncertaintyMetrics.predictive_entropy(
                mean_prediction, axis=-1
            )
            if values.ndim > 1:
                values = values.reshape(values.shape[0], -1).mean(axis=1)
            return np.asarray(values, dtype=np.float64)

        if metric_name == "mutual_information":
            return np.asarray(
                SeismicUncertaintyMetrics.mutual_information(predictions),
                dtype=np.float64,
            )
        if metric_name == "pick_time_variance":
            return np.asarray(
                SeismicUncertaintyMetrics.pick_time_variance(predictions),
                dtype=np.float64,
            )
        if metric_name == "peak_probability_variance":
            return np.asarray(
                SeismicUncertaintyMetrics.peak_probability_variance(predictions),
                dtype=np.float64,
            )
        if metric_name == "per_timestep_variance":
            return np.asarray(
                SeismicUncertaintyMetrics.per_timestep_variance(predictions),
                dtype=np.float64,
            )
        if metric_name == "multi_peak_ambiguity":
            curves = (
                np.mean(predictions, axis=0) if predictions.ndim >= 3 else predictions
            )
            return np.asarray(
                SeismicUncertaintyMetrics.multi_peak_ambiguity(curves),
                dtype=np.float64,
            )
        if metric_name == "peak_sharpness_uncertainty":
            curves = (
                np.mean(predictions, axis=0) if predictions.ndim >= 3 else predictions
            )
            return np.asarray(
                SeismicUncertaintyMetrics.peak_sharpness_uncertainty(curves),
                dtype=np.float64,
            )
        if metric_name == "js_divergence":
            return np.asarray(
                SeismicUncertaintyMetrics.js_divergence(predictions),
                dtype=np.float64,
            )
        if metric_name == "mean_pairwise_kl":
            return np.asarray(
                SeismicUncertaintyMetrics.mean_pairwise_kl(predictions),
                dtype=np.float64,
            )
        if metric_name == "vote_entropy":
            return np.asarray(
                SeismicUncertaintyMetrics.vote_entropy(predictions),
                dtype=np.float64,
            )
        raise ValueError(f"Unknown uncertainty metric '{metric}'.")


class BaseUncertaintyEstimator(ABC):
    """
    Base uncertainty estimator contract.
    """

    name = "base_uncertainty"

    @abstractmethod
    def score(
        self,
        model: Any,
        dataloader: Any | None = None,
        dataset: Any | None = None,
        indices: Sequence[int] | np.ndarray | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        raise NotImplementedError

    def get_config(self) -> dict[str, Any]:
        return {"name": self.name}


class SoftmaxUncertainty(BaseUncertaintyEstimator):
    """
    Single-model uncertainty from deterministic predictions.
    """

    name = "softmax_uncertainty"

    def __init__(
        self,
        metric: Literal[
            "predictive_entropy", "least_confidence", "margin", "multi_peak_ambiguity"
        ] = "predictive_entropy",
        batch_size: int = 64,
        device: str | torch.device | None = None,
        prediction_fn: Callable[[Any, torch.Tensor], Any] | None = None,
    ) -> None:
        self.metric = metric
        self.batch_size = int(batch_size)
        self.device = device
        self.prediction_fn = prediction_fn

    def score(
        self,
        model: Any,
        dataloader: Any | None = None,
        dataset: Any | None = None,
        indices: Sequence[int] | np.ndarray | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        predictions, used_indices = _predict_model_on_dataset(
            model=model,
            dataset=dataset,
            indices=indices,
            dataloader=dataloader,
            batch_size=self.batch_size,
            device=self.device,
            prediction_fn=self.prediction_fn,
        )
        probs = _normalize_probabilities(predictions, axis=-1)

        if self.metric == "least_confidence":
            peak = np.max(probs, axis=-1)
            if np.ndim(peak) > 1:
                peak = peak.reshape(peak.shape[0], -1).mean(axis=1)
            scores = 1.0 - peak
        elif self.metric == "margin":
            flat = probs if probs.ndim <= 2 else probs.reshape(probs.shape[0], -1)
            if flat.shape[1] < 2:
                scores = np.zeros(flat.shape[0], dtype=np.float64)
            else:
                sorted_probs = np.sort(flat, axis=-1)
                scores = 1.0 - (sorted_probs[:, -1] - sorted_probs[:, -2])
        elif self.metric == "multi_peak_ambiguity":
            scores = SeismicUncertaintyMetrics.multi_peak_ambiguity(probs)
        else:
            entropy = SeismicUncertaintyMetrics.predictive_entropy(probs, axis=-1)
            if np.ndim(entropy) > 1:
                entropy = entropy.reshape(entropy.shape[0], -1).mean(axis=1)
            scores = entropy

        diagnostics = {
            "indices": used_indices,
            "mean_prediction": probs,
        }
        return np.asarray(scores, dtype=np.float64), diagnostics

    def get_config(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "metric": self.metric,
            "batch_size": self.batch_size,
        }


class MCDropoutUncertainty(BaseUncertaintyEstimator):
    """
    MC Dropout uncertainty estimator.
    """

    name = "mc_dropout_uncertainty"

    def __init__(
        self,
        n_passes: int = 20,
        score_metric: str = "pick_time_variance",
        batch_size: int = 64,
        device: str | torch.device | None = None,
        prediction_fn: Callable[[Any, torch.Tensor], Any] | None = None,
    ) -> None:
        if n_passes <= 0:
            raise ValueError("n_passes must be > 0.")
        self.n_passes = int(n_passes)
        self.score_metric = score_metric
        self.batch_size = int(batch_size)
        self.device = device
        self.prediction_fn = prediction_fn

    @staticmethod
    def _enable_dropout_only(model: Any) -> tuple[bool, dict[nn.Module, bool]]:
        was_training = bool(getattr(model, "training", False))
        module_states: dict[nn.Module, bool] = {}
        for module in model.modules():
            module_states[module] = bool(module.training)
        model.eval()
        for module in model.modules():
            if isinstance(module, _dropout_modules()):
                module.train()
        return was_training, module_states

    @staticmethod
    def _restore_module_training_states(
        model: Any, module_states: Mapping[nn.Module, bool], was_training: bool
    ) -> None:
        for module, state in module_states.items():
            module.train(state)
        if was_training:
            model.train()
        else:
            model.eval()

    def score(
        self,
        model: Any,
        dataloader: Any | None = None,
        dataset: Any | None = None,
        indices: Sequence[int] | np.ndarray | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if dataloader is None and dataset is None:
            raise ValueError("Provide either dataloader or dataset.")

        was_training, module_states = self._enable_dropout_only(model)
        all_passes: list[np.ndarray] = []
        used_indices: np.ndarray | None = None
        try:
            for _ in range(self.n_passes):
                preds, idx = _predict_model_on_dataset(
                    model=model,
                    dataloader=dataloader,
                    dataset=dataset,
                    indices=indices,
                    batch_size=self.batch_size,
                    device=self.device,
                    prediction_fn=self.prediction_fn,
                )
                all_passes.append(np.asarray(preds, dtype=np.float64))
                if used_indices is None:
                    used_indices = np.asarray(idx, dtype=np.int64)
                elif not np.array_equal(used_indices, idx):
                    raise ValueError(
                        "Inconsistent sample ordering across MC dropout passes."
                    )
        finally:
            self._restore_module_training_states(
                model=model,
                module_states=module_states,
                was_training=was_training,
            )

        stacked = np.stack(all_passes, axis=0)
        scores = SeismicUncertaintyMetrics.compute(self.score_metric, stacked)
        diagnostics = {
            "indices": np.array([], dtype=np.int64)
            if used_indices is None
            else used_indices,
            "mean_prediction": np.mean(stacked, axis=0),
            "variance_curve": np.var(stacked, axis=0),
            "predicted_peak_indices": np.argmax(stacked, axis=-1),
            "entropy": SeismicUncertaintyMetrics.compute("predictive_entropy", stacked),
            "raw_pass_predictions": stacked,
        }
        return np.asarray(scores, dtype=np.float64), diagnostics

    def get_config(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "n_passes": self.n_passes,
            "score_metric": self.score_metric,
            "batch_size": self.batch_size,
        }


class EnsembleUncertainty(BaseUncertaintyEstimator):
    """
    Deep-ensemble uncertainty estimator.
    """

    name = "ensemble_uncertainty"

    def __init__(
        self,
        models: Sequence[Any] | None = None,
        checkpoints: Sequence[str] | None = None,
        model_loader: Callable[[str], Any] | None = None,
        score_metric: str = "js_divergence",
        batch_size: int = 64,
        device: str | torch.device | None = None,
        prediction_fn: Callable[[Any, torch.Tensor], Any] | None = None,
    ) -> None:
        self.models = list(models) if models is not None else []
        self.checkpoints = list(checkpoints) if checkpoints is not None else []
        self.model_loader = model_loader
        self.score_metric = score_metric
        self.batch_size = int(batch_size)
        self.device = device
        self.prediction_fn = prediction_fn

    def _resolve_models(self) -> list[Any]:
        if self.models:
            return list(self.models)
        if len(self.checkpoints) == 0:
            raise ValueError("EnsembleUncertainty requires models or checkpoints.")
        if self.model_loader is None:
            raise ValueError(
                "model_loader callable is required when using checkpoint paths."
            )
        return [self.model_loader(path) for path in self.checkpoints]

    def score(
        self,
        model: Any,
        dataloader: Any | None = None,
        dataset: Any | None = None,
        indices: Sequence[int] | np.ndarray | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        members = self._resolve_models()
        if len(members) == 0:
            raise ValueError("EnsembleUncertainty received an empty model list.")

        outputs: list[np.ndarray] = []
        used_indices: np.ndarray | None = None
        for member in members:
            preds, idx = _predict_model_on_dataset(
                model=member,
                dataloader=dataloader,
                dataset=dataset,
                indices=indices,
                batch_size=self.batch_size,
                device=self.device,
                prediction_fn=self.prediction_fn,
            )
            outputs.append(np.asarray(preds, dtype=np.float64))
            if used_indices is None:
                used_indices = np.asarray(idx, dtype=np.int64)
            elif not np.array_equal(used_indices, idx):
                raise ValueError(
                    "Inconsistent sample ordering across ensemble members."
                )

        stacked = np.stack(outputs, axis=0)
        scores = SeismicUncertaintyMetrics.compute(self.score_metric, stacked)
        diagnostics = {
            "indices": np.array([], dtype=np.int64)
            if used_indices is None
            else used_indices,
            "mean_prediction": np.mean(stacked, axis=0),
            "variance_curve": np.var(stacked, axis=0),
            "predicted_peak_indices": np.argmax(stacked, axis=-1),
            "raw_member_predictions": stacked,
        }
        return np.asarray(scores, dtype=np.float64), diagnostics

    def get_config(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "score_metric": self.score_metric,
            "batch_size": self.batch_size,
            "n_models": len(self.models) if self.models else len(self.checkpoints),
        }


def _validate_diversity_inputs(
    candidate_indices: np.ndarray,
    embeddings: np.ndarray,
    n_select: int,
) -> int:
    candidates = np.asarray(candidate_indices, dtype=np.int64)
    emb = np.asarray(embeddings, dtype=np.float64)

    if candidates.ndim != 1:
        raise ValueError("candidate_indices must be one-dimensional.")
    if emb.ndim != 2:
        raise ValueError("embeddings must be a 2D array.")
    if emb.shape[0] != candidates.size:
        raise ValueError("embeddings must align with candidate_indices.")
    if n_select < 0:
        raise ValueError("n_select must be >= 0.")
    return min(int(n_select), int(candidates.size))


def _pairwise_sqeuclidean(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a2 = np.sum(a * a, axis=1, keepdims=True)
    b2 = np.sum(b * b, axis=1, keepdims=True).T
    cross = a @ b.T
    return np.maximum(a2 + b2 - 2.0 * cross, 0.0)


def _run_kmeans(
    embeddings: np.ndarray,
    n_clusters: int,
    rng: np.random.Generator,
    max_iter: int = 25,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(embeddings, dtype=np.float64)
    n_samples = x.shape[0]
    n_clusters = min(max(int(n_clusters), 1), n_samples)

    init_indices = rng.choice(n_samples, size=n_clusters, replace=False)
    centers = x[init_indices].copy()
    labels = np.zeros(n_samples, dtype=np.int64)

    for _ in range(max_iter):
        dist = _pairwise_sqeuclidean(x, centers)
        new_labels = np.argmin(dist, axis=1).astype(np.int64)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for c in range(n_clusters):
            members = x[labels == c]
            if len(members) == 0:
                centers[c] = x[int(rng.integers(0, n_samples))]
            else:
                centers[c] = np.mean(members, axis=0)

    return labels, centers


def representativeness_score(
    embeddings: np.ndarray, n_neighbors: int = 10
) -> np.ndarray:
    x = np.asarray(embeddings, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("embeddings must be 2D.")
    if x.shape[0] == 0:
        return np.array([], dtype=np.float64)
    if x.shape[0] == 1:
        return np.array([1.0], dtype=np.float64)

    dist = np.sqrt(_pairwise_sqeuclidean(x, x))
    np.fill_diagonal(dist, np.inf)
    k = min(max(int(n_neighbors), 1), x.shape[0] - 1)
    nearest = np.partition(dist, kth=k, axis=1)[:, :k]
    density = 1.0 / (np.mean(nearest, axis=1) + 1e-12)
    return robust_minmax(density)


class BaseDiversitySelector(ABC):
    name = "base_diversity"

    @abstractmethod
    def select(
        self,
        candidate_indices: np.ndarray,
        embeddings: np.ndarray,
        n_select: int,
        uncertainty_scores: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        raise NotImplementedError

    def get_config(self) -> dict[str, Any]:
        return {"name": self.name}


class FarthestFirstSelector(BaseDiversitySelector):
    name = "farthest_first"

    def select(
        self,
        candidate_indices: np.ndarray,
        embeddings: np.ndarray,
        n_select: int,
        uncertainty_scores: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        n = _validate_diversity_inputs(candidate_indices, embeddings, n_select)
        if n == 0:
            return np.array([], dtype=np.int64)

        candidates = np.asarray(candidate_indices, dtype=np.int64)
        x = np.asarray(embeddings, dtype=np.float64)
        rng = rng if rng is not None else np.random.default_rng(0)

        if uncertainty_scores is not None:
            unc = np.asarray(uncertainty_scores, dtype=np.float64)
            if unc.shape[0] != candidates.shape[0]:
                raise ValueError(
                    "uncertainty_scores must align with candidate_indices."
                )
            start = int(np.argmax(unc))
        else:
            start = int(rng.integers(0, candidates.size))

        selected_pos = [start]
        min_dist = _pairwise_sqeuclidean(x, x[[start], :]).reshape(-1)

        while len(selected_pos) < n:
            min_dist[selected_pos] = -np.inf
            next_pos = int(np.argmax(min_dist))
            selected_pos.append(next_pos)
            dist_to_new = _pairwise_sqeuclidean(x, x[[next_pos], :]).reshape(-1)
            min_dist = np.minimum(min_dist, dist_to_new)

        return np.sort(candidates[np.asarray(selected_pos, dtype=np.int64)])


class KCenterGreedySelector(BaseDiversitySelector):
    name = "kcenter_greedy"

    def __init__(
        self,
        uncertainty_weight: float = 0.0,
    ) -> None:
        self.uncertainty_weight = float(uncertainty_weight)

    def select(
        self,
        candidate_indices: np.ndarray,
        embeddings: np.ndarray,
        n_select: int,
        uncertainty_scores: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        n = _validate_diversity_inputs(candidate_indices, embeddings, n_select)
        if n == 0:
            return np.array([], dtype=np.int64)

        candidates = np.asarray(candidate_indices, dtype=np.int64)
        x = np.asarray(embeddings, dtype=np.float64)

        if uncertainty_scores is not None:
            unc = np.asarray(uncertainty_scores, dtype=np.float64)
            if unc.shape[0] != candidates.shape[0]:
                raise ValueError(
                    "uncertainty_scores must align with candidate_indices."
                )
            start = int(np.argmax(unc))
            unc_norm = robust_minmax(unc)
        else:
            start = int(np.argmin(candidates))
            unc_norm = np.zeros(candidates.shape[0], dtype=np.float64)

        selected_pos = [start]
        min_dist = _pairwise_sqeuclidean(x, x[[start], :]).reshape(-1)

        while len(selected_pos) < n:
            min_dist[selected_pos] = -np.inf
            objective = min_dist + self.uncertainty_weight * unc_norm
            next_pos = int(np.argmax(objective))
            selected_pos.append(next_pos)
            dist_to_new = _pairwise_sqeuclidean(x, x[[next_pos], :]).reshape(-1)
            min_dist = np.minimum(min_dist, dist_to_new)

        return np.sort(candidates[np.asarray(selected_pos, dtype=np.int64)])

    def get_config(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "uncertainty_weight": self.uncertainty_weight,
        }


class KMeansRepresentativeSelector(BaseDiversitySelector):
    name = "kmeans_representative"

    def __init__(
        self,
        method: Literal["centroid", "medoid"] = "medoid",
        max_iter: int = 25,
    ) -> None:
        self.method = method
        self.max_iter = int(max_iter)

    def _centroid_nearest(
        self,
        members: np.ndarray,
        centers: np.ndarray,
        labels: np.ndarray,
    ) -> list[int]:
        selected: list[int] = []
        n_clusters = centers.shape[0]
        for cluster_id in range(n_clusters):
            member_pos = np.where(labels == cluster_id)[0]
            if member_pos.size == 0:
                continue
            x = members[member_pos]
            d2 = np.sum((x - centers[cluster_id]) ** 2, axis=1)
            selected.append(int(member_pos[int(np.argmin(d2))]))
        return selected

    def _medoids(
        self, members: np.ndarray, labels: np.ndarray, n_clusters: int
    ) -> list[int]:
        selected: list[int] = []
        for cluster_id in range(n_clusters):
            member_pos = np.where(labels == cluster_id)[0]
            if member_pos.size == 0:
                continue
            x = members[member_pos]
            d2 = _pairwise_sqeuclidean(x, x)
            selected.append(int(member_pos[int(np.argmin(np.sum(d2, axis=1)))]))
        return selected

    def select(
        self,
        candidate_indices: np.ndarray,
        embeddings: np.ndarray,
        n_select: int,
        uncertainty_scores: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        n = _validate_diversity_inputs(candidate_indices, embeddings, n_select)
        if n == 0:
            return np.array([], dtype=np.int64)

        candidates = np.asarray(candidate_indices, dtype=np.int64)
        x = np.asarray(embeddings, dtype=np.float64)
        rng = rng if rng is not None else np.random.default_rng(0)

        labels, centers = _run_kmeans(
            embeddings=x,
            n_clusters=n,
            rng=rng,
            max_iter=self.max_iter,
        )

        if self.method == "centroid":
            selected_pos = self._centroid_nearest(x, centers, labels)
        elif self.method == "medoid":
            selected_pos = self._medoids(x, labels, n_clusters=centers.shape[0])
        else:
            raise ValueError(f"Unknown kmeans representative method '{self.method}'.")

        selected_pos = sorted(set(int(p) for p in selected_pos))
        if len(selected_pos) < n:
            remaining = np.setdiff1d(
                np.arange(candidates.size, dtype=np.int64),
                np.asarray(selected_pos, dtype=np.int64),
                assume_unique=True,
            )
            filler = FarthestFirstSelector().select(
                candidate_indices=remaining,
                embeddings=x[remaining],
                n_select=n - len(selected_pos),
                uncertainty_scores=None,
                rng=rng,
            )
            selected_pos.extend(filler.tolist())

        selected_pos = np.asarray(sorted(set(selected_pos))[:n], dtype=np.int64)
        return np.sort(candidates[selected_pos])

    def get_config(self) -> dict[str, Any]:
        return {"name": self.name, "method": self.method, "max_iter": self.max_iter}


def _stable_topk_indices(
    indices: np.ndarray,
    scores: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    idx = np.asarray(indices, dtype=np.int64)
    sc = np.asarray(scores, dtype=np.float64)
    if idx.ndim != 1 or sc.ndim != 1 or idx.size != sc.size:
        raise ValueError("indices and scores must be aligned one-dimensional arrays.")

    k = min(max(int(k), 0), int(idx.size))
    if k == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float64)

    order = np.lexsort((idx, -sc))
    chosen = order[:k]
    return idx[chosen], sc[chosen]


def _score_stats(scores: np.ndarray) -> dict[str, float]:
    arr = np.asarray(scores, dtype=np.float64)
    if arr.size == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
        }
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _write_csv_rows(
    path: Path, rows: Sequence[dict[str, Any]], columns: Sequence[str]
) -> None:
    if len(rows) == 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


class UncertaintyEstimatorQueryStrategy(QueryStrategy):
    """
    Query strategy wrapper around an uncertainty estimator.
    """

    def __init__(
        self,
        uncertainty_estimator: BaseUncertaintyEstimator,
        pool_filter: PoolFilter | None = None,
    ) -> None:
        self.uncertainty_estimator = uncertainty_estimator
        self.pool_filter = (
            pool_filter if pool_filter is not None else PoolFilter(enabled=False)
        )
        self.last_uncertainty_scores: dict[int, float] = {}
        self.last_diagnostics: dict[str, Any] = {}
        self.name = uncertainty_estimator.name

    def select(
        self,
        candidate_indices: np.ndarray,
        budget: int,
        scores: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        selected, _ = _stable_topk_indices(candidate_indices, scores, budget)
        return np.sort(selected.astype(np.int64))

    def select_batch(
        self,
        model: Any,
        unlabeled_dataset: Any,
        batch_size: int,
        metadata: Any | None = None,
        unlabeled_indices: Sequence[int] | np.ndarray | None = None,
    ) -> np.ndarray:
        pool_indices = np.asarray(
            _resolve_indices(dataset=unlabeled_dataset, indices=unlabeled_indices),
            dtype=np.int64,
        )
        filter_result = self.pool_filter.apply(
            dataset=unlabeled_dataset,
            metadata=metadata,
            indices=pool_indices,
        )
        valid_indices = np.asarray(filter_result.valid_indices, dtype=np.int64)
        k = min(int(batch_size), int(valid_indices.size))
        if k == 0:
            self.last_uncertainty_scores = {}
            self.last_diagnostics = {"filter_result": filter_result}
            return np.array([], dtype=np.int64)

        scores, diagnostics = self.uncertainty_estimator.score(
            model=model,
            dataset=unlabeled_dataset,
            indices=valid_indices,
        )
        if scores.shape[0] != valid_indices.shape[0]:
            raise ValueError("Uncertainty scores are not aligned with valid indices.")

        selected, selected_scores = _stable_topk_indices(valid_indices, scores, k)
        self.last_uncertainty_scores = {
            int(i): float(s)
            for i, s in zip(selected.tolist(), selected_scores.tolist())
        }
        self.last_diagnostics = {
            "filter_result": filter_result,
            "uncertainty_stats": _score_stats(scores),
            "uncertainty_diagnostics": diagnostics,
        }
        return np.sort(selected.astype(np.int64))

    def get_config(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "uncertainty": self.uncertainty_estimator.get_config(),
            "filtering_enabled": bool(self.pool_filter.enabled),
        }


class RepresentativeInitStrategy(QueryStrategy):
    """
    Representative cold-start selection in embedding space.
    """

    name = "representative_init"

    def __init__(
        self,
        selection_method: Literal[
            "kmeans_centroid", "kmeans_medoid", "kcenter_greedy"
        ] = "kmeans_medoid",
        embedding_layer: str = "encoder_global_pool",
        embedding_extractor: EmbeddingExtractor | None = None,
        pool_filter: PoolFilter | None = None,
        embedding_cache_path: str | None = None,
        batch_size: int = 64,
        seed: int | None = None,
    ) -> None:
        self.selection_method = selection_method
        self.embedding_layer = embedding_layer
        self.embedding_extractor = (
            embedding_extractor
            if embedding_extractor is not None
            else EmbeddingExtractor()
        )
        self.pool_filter = (
            pool_filter if pool_filter is not None else PoolFilter(enabled=False)
        )
        self.embedding_cache_path = embedding_cache_path
        self.batch_size = int(batch_size)
        self.rng = np.random.default_rng(seed)
        self.last_uncertainty_scores: dict[int, float] = {}
        self.last_diagnostics: dict[str, Any] = {}

    def select(
        self,
        candidate_indices: np.ndarray,
        budget: int,
        scores: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        del rng
        selected, _ = _stable_topk_indices(candidate_indices, scores, budget)
        return np.sort(selected.astype(np.int64))

    def _selector(self) -> BaseDiversitySelector:
        if self.selection_method == "kmeans_centroid":
            return KMeansRepresentativeSelector(method="centroid")
        if self.selection_method == "kmeans_medoid":
            return KMeansRepresentativeSelector(method="medoid")
        if self.selection_method == "kcenter_greedy":
            return KCenterGreedySelector()
        raise ValueError(f"Unknown representative method '{self.selection_method}'.")

    def select_batch(
        self,
        model: Any,
        unlabeled_dataset: Any,
        batch_size: int,
        metadata: Any | None = None,
        unlabeled_indices: Sequence[int] | np.ndarray | None = None,
    ) -> np.ndarray:
        pool_indices = np.asarray(
            _resolve_indices(dataset=unlabeled_dataset, indices=unlabeled_indices),
            dtype=np.int64,
        )
        k = min(int(batch_size), int(pool_indices.size))
        if k == 0:
            self.last_diagnostics = {}
            return np.array([], dtype=np.int64)

        embeddings = self.embedding_extractor.extract(
            model=model,
            dataset=unlabeled_dataset,
            indices=pool_indices,
            layer=self.embedding_layer,
            cache_path=self.embedding_cache_path,
            batch_size=self.batch_size,
        )
        if embeddings.shape[0] != pool_indices.shape[0]:
            raise ValueError(
                "EmbeddingExtractor output is not aligned with pool indices."
            )

        filter_result = self.pool_filter.apply(
            dataset=unlabeled_dataset,
            metadata=metadata,
            embeddings=embeddings,
            indices=pool_indices,
        )
        valid_indices = np.asarray(filter_result.valid_indices, dtype=np.int64)
        valid_embeddings = np.asarray(
            embeddings[filter_result.valid_mask], dtype=np.float64
        )
        k = min(k, int(valid_indices.size))
        if k == 0:
            self.last_diagnostics = {"filter_result": filter_result}
            return np.array([], dtype=np.int64)

        selected = self._selector().select(
            candidate_indices=valid_indices,
            embeddings=valid_embeddings,
            n_select=k,
            uncertainty_scores=None,
            rng=self.rng,
        )
        self.last_diagnostics = {
            "filter_result": filter_result,
            "selection_method": self.selection_method,
            "pool_size": int(pool_indices.size),
            "valid_size": int(valid_indices.size),
            "selected_size": int(selected.size),
        }
        return np.sort(np.asarray(selected, dtype=np.int64))

    def get_config(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "selection_method": self.selection_method,
            "embedding_layer": self.embedding_layer,
            "batch_size": self.batch_size,
            "filtering_enabled": bool(self.pool_filter.enabled),
        }


class HybridUncertaintyDiversityStrategy(QueryStrategy):
    """
    Hybrid strategy: filtering -> uncertainty -> top-K -> diversity.
    """

    name = "hybrid_uncertainty_diversity"

    def __init__(
        self,
        uncertainty_estimator: BaseUncertaintyEstimator | None = None,
        diversity_selector: BaseDiversitySelector | None = None,
        embedding_extractor: EmbeddingExtractor | None = None,
        pool_filter: PoolFilter | None = None,
        mode: Literal[
            "topk_uncertainty_then_kcenter",
            "topk_uncertainty_then_cluster",
            "weighted_uncertainty_plus_diversity",
            "weighted_sum",
        ] = "topk_uncertainty_then_kcenter",
        candidate_pool_size: int = 1000,
        alpha: float = 0.7,
        beta: float = 0.3,
        gamma: float = 0.0,
        embedding_layer: str = "encoder_global_pool",
        embedding_cache_path: str | None = None,
        artifact_dir: str | None = None,
        seed: int | None = None,
    ) -> None:
        self.uncertainty_estimator = (
            uncertainty_estimator
            if uncertainty_estimator is not None
            else SoftmaxUncertainty(metric="predictive_entropy")
        )
        self.diversity_selector = (
            diversity_selector
            if diversity_selector is not None
            else KCenterGreedySelector()
        )
        self.embedding_extractor = (
            embedding_extractor
            if embedding_extractor is not None
            else EmbeddingExtractor()
        )
        self.pool_filter = (
            pool_filter if pool_filter is not None else PoolFilter(enabled=False)
        )
        self.mode = mode
        self.candidate_pool_size = int(candidate_pool_size)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.embedding_layer = embedding_layer
        self.embedding_cache_path = embedding_cache_path
        self.artifact_dir = artifact_dir
        self.rng = np.random.default_rng(seed)
        self.last_uncertainty_scores: dict[int, float] = {}
        self.last_diagnostics: dict[str, Any] = {}

    def select(
        self,
        candidate_indices: np.ndarray,
        budget: int,
        scores: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        del rng
        selected, _ = _stable_topk_indices(candidate_indices, scores, budget)
        return np.sort(selected.astype(np.int64))

    def _requires_embeddings(self) -> bool:
        if self.pool_filter.enabled and self.pool_filter.duplicate_enabled:
            if self.pool_filter.duplicate_method == "embedding_cosine":
                return True
        if self.diversity_selector is not None:
            return True
        return self.mode in {"weighted_uncertainty_plus_diversity", "weighted_sum"}

    def _save_artifacts(
        self,
        selected_indices: np.ndarray,
        uncertainty_rows: list[dict[str, Any]],
        filter_result: FilterResult,
        diagnostics: Mapping[str, Any],
    ) -> None:
        if self.artifact_dir is None:
            return

        out_dir = Path(self.artifact_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        np.save(
            out_dir / "selected_indices.npy",
            np.asarray(selected_indices, dtype=np.int64),
        )
        _write_csv_rows(
            out_dir / "uncertainty_scores.csv",
            rows=uncertainty_rows,
            columns=["index", "score"],
        )
        reject_rows = []
        for idx, reasons in filter_result.reason_codes.items():
            reject_rows.append({"index": int(idx), "reasons": "|".join(reasons)})
        _write_csv_rows(
            out_dir / "filter_rejections.csv",
            rows=reject_rows,
            columns=["index", "reasons"],
        )
        with (out_dir / "selection_diagnostics.json").open("w", encoding="utf-8") as f:
            json.dump(dict(diagnostics), f, indent=2)

    def select_batch(
        self,
        model: Any,
        unlabeled_dataset: Any,
        batch_size: int,
        metadata: Any | None = None,
        unlabeled_indices: Sequence[int] | np.ndarray | None = None,
    ) -> np.ndarray:
        pool_indices = np.asarray(
            _resolve_indices(dataset=unlabeled_dataset, indices=unlabeled_indices),
            dtype=np.int64,
        )
        n_query = min(int(batch_size), int(pool_indices.size))
        if n_query == 0:
            self.last_uncertainty_scores = {}
            self.last_diagnostics = {"pool_size": int(pool_indices.size)}
            return np.array([], dtype=np.int64)

        embeddings: np.ndarray | None = None
        if self._requires_embeddings():
            embeddings = self.embedding_extractor.extract(
                model=model,
                dataset=unlabeled_dataset,
                indices=pool_indices,
                layer=self.embedding_layer,
                cache_path=self.embedding_cache_path,
            )

        filter_result = self.pool_filter.apply(
            dataset=unlabeled_dataset,
            metadata=metadata,
            embeddings=embeddings,
            indices=pool_indices,
        )
        valid_indices = np.asarray(filter_result.valid_indices, dtype=np.int64)
        valid_embeddings = (
            None
            if embeddings is None
            else np.asarray(embeddings[filter_result.valid_mask], dtype=np.float64)
        )
        n_query = min(n_query, int(valid_indices.size))
        if n_query == 0:
            self.last_uncertainty_scores = {}
            self.last_diagnostics = {
                "pool_size": int(pool_indices.size),
                "valid_size": int(valid_indices.size),
                "filter_result": filter_result,
            }
            return np.array([], dtype=np.int64)

        uncertainty_scores, uncertainty_diag = self.uncertainty_estimator.score(
            model=model,
            dataset=unlabeled_dataset,
            indices=valid_indices,
        )
        uncertainty_scores = np.asarray(uncertainty_scores, dtype=np.float64)
        if uncertainty_scores.shape[0] != valid_indices.shape[0]:
            raise ValueError("Uncertainty scores are not aligned with valid indices.")

        top_k = min(max(self.candidate_pool_size, n_query), int(valid_indices.size))
        candidate_indices, candidate_scores = _stable_topk_indices(
            valid_indices, uncertainty_scores, top_k
        )

        uncertainty_rows = [
            {"index": int(i), "score": float(s)}
            for i, s in zip(valid_indices.tolist(), uncertainty_scores.tolist())
        ]

        if self.mode in {"weighted_uncertainty_plus_diversity", "weighted_sum"}:
            if valid_embeddings is None:
                raise ValueError("Weighted hybrid mode requires embeddings.")
            position_lookup = {
                int(idx): pos for pos, idx in enumerate(valid_indices.tolist())
            }
            candidate_pos = np.array(
                [position_lookup[int(i)] for i in candidate_indices], dtype=np.int64
            )
            candidate_emb = valid_embeddings[candidate_pos]

            unc_norm = robust_minmax(candidate_scores)
            density = representativeness_score(candidate_emb)
            centroid = np.mean(candidate_emb, axis=0, keepdims=True)
            dist = np.sqrt(_pairwise_sqeuclidean(candidate_emb, centroid)).reshape(-1)
            rep = 1.0 - robust_minmax(dist)
            combined = self.alpha * unc_norm + self.beta * density + self.gamma * rep
            selected, selected_scores = _stable_topk_indices(
                candidate_indices, combined, n_query
            )
        else:
            if self.diversity_selector is None or valid_embeddings is None:
                selected, selected_scores = _stable_topk_indices(
                    candidate_indices, candidate_scores, n_query
                )
            else:
                position_lookup = {
                    int(idx): pos for pos, idx in enumerate(valid_indices.tolist())
                }
                candidate_pos = np.array(
                    [position_lookup[int(i)] for i in candidate_indices], dtype=np.int64
                )
                candidate_emb = valid_embeddings[candidate_pos]
                selected = self.diversity_selector.select(
                    candidate_indices=candidate_indices,
                    embeddings=candidate_emb,
                    n_select=n_query,
                    uncertainty_scores=candidate_scores,
                    rng=self.rng,
                )
                selected = np.asarray(selected, dtype=np.int64)
                score_lookup = {
                    int(i): float(s)
                    for i, s in zip(
                        candidate_indices.tolist(), candidate_scores.tolist()
                    )
                }
                selected_scores = np.asarray(
                    [score_lookup[int(i)] for i in selected], dtype=np.float64
                )

        selected = np.sort(np.asarray(selected, dtype=np.int64))
        if np.unique(selected).size != selected.size:
            raise ValueError("Hybrid selection produced duplicate indices.")
        if not np.all(np.isin(selected, valid_indices)):
            raise ValueError("Hybrid selection produced indices outside valid pool.")

        self.last_uncertainty_scores = {
            int(i): float(s)
            for i, s in zip(selected.tolist(), selected_scores.tolist())
        }
        self.last_diagnostics = {
            "pool_size": int(pool_indices.size),
            "filtered_out": int(filter_result.rejected_indices.size),
            "valid_size": int(valid_indices.size),
            "top_k": int(top_k),
            "final_batch_size": int(selected.size),
            "uncertainty_stats": _score_stats(uncertainty_scores),
            "selection_stats": _score_stats(selected_scores),
            "mode": self.mode,
            "uncertainty_method": self.uncertainty_estimator.name,
            "diversity_method": None
            if self.diversity_selector is None
            else self.diversity_selector.name,
            "filter_result": {
                "rejected_count": int(filter_result.rejected_indices.size),
                "reason_counts": {
                    key: len(value) for key, value in filter_result.reason_codes.items()
                },
            },
        }

        self._save_artifacts(
            selected_indices=selected,
            uncertainty_rows=uncertainty_rows,
            filter_result=filter_result,
            diagnostics=self.last_diagnostics,
        )

        logger.info(
            "AL hybrid select: pool=%d filtered=%d valid=%d topk=%d batch=%d",
            int(pool_indices.size),
            int(filter_result.rejected_indices.size),
            int(valid_indices.size),
            int(top_k),
            int(selected.size),
        )
        self.last_diagnostics["uncertainty_diagnostics"] = uncertainty_diag
        return selected

    def get_config(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "mode": self.mode,
            "candidate_pool_size": self.candidate_pool_size,
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
            "embedding_layer": self.embedding_layer,
            "uncertainty": self.uncertainty_estimator.get_config(),
            "diversity": None
            if self.diversity_selector is None
            else self.diversity_selector.get_config(),
            "filtering_enabled": bool(self.pool_filter.enabled),
        }


def build_pool_filter(config: Mapping[str, Any] | None = None) -> PoolFilter:
    if config is None:
        return PoolFilter(enabled=False)

    cfg = dict(config)
    low_energy = dict(cfg.get("low_energy", {}))
    abnormal = dict(cfg.get("abnormal", {}))
    duplicate = dict(cfg.get("duplicate", {}))

    return PoolFilter(
        enabled=bool(cfg.get("enabled", True)),
        low_energy_enabled=bool(low_energy.get("enabled", True)),
        rms_threshold=float(low_energy.get("rms_threshold", 1e-4)),
        abnormal_enabled=bool(abnormal.get("enabled", True)),
        max_abs_threshold=float(abnormal.get("max_abs_threshold", 1e3)),
        flat_std_threshold=float(abnormal.get("flat_std_threshold", 1e-8)),
        clipped_fraction_threshold=float(
            abnormal.get("clipped_fraction_threshold", 0.2)
        ),
        duplicate_enabled=bool(duplicate.get("enabled", True)),
        duplicate_method=duplicate.get("method", "exact_hash"),
        similarity_threshold=float(duplicate.get("similarity_threshold", 0.999)),
        metadata_duplicate_keys=tuple(
            duplicate.get(
                "metadata_keys",
                ("source_id", "event_id", "trace_name", "station_code"),
            )
        ),
    )


def build_uncertainty_estimator(
    config: Mapping[str, Any] | None = None,
    *,
    models: Sequence[Any] | None = None,
    model_loader: Callable[[str], Any] | None = None,
) -> BaseUncertaintyEstimator:
    if config is None:
        return SoftmaxUncertainty(metric="predictive_entropy")

    cfg = dict(config)
    method = str(cfg.get("method", "softmax")).lower()

    if method in {"softmax", "entropy", "softmax_uncertainty"}:
        metric = str(cfg.get("score_metric", cfg.get("metric", "predictive_entropy")))
        return SoftmaxUncertainty(
            metric=metric, batch_size=int(cfg.get("batch_size", 64))
        )

    if method in {"mc_dropout", "mc_dropout_uncertainty"}:
        metric = str(cfg.get("score_metric", "pick_time_variance"))
        return MCDropoutUncertainty(
            n_passes=int(cfg.get("n_passes", 20)),
            score_metric=metric,
            batch_size=int(cfg.get("batch_size", 64)),
        )

    if method in {"ensemble", "ensemble_uncertainty"}:
        metric = str(cfg.get("score_metric", "js_divergence"))
        checkpoints = cfg.get("checkpoints")
        if checkpoints is not None and not isinstance(checkpoints, list):
            raise ValueError("uncertainty.checkpoints must be a list if provided.")
        return EnsembleUncertainty(
            models=models,
            checkpoints=checkpoints,
            model_loader=model_loader,
            score_metric=metric,
            batch_size=int(cfg.get("batch_size", 64)),
        )

    raise ValueError(f"Unsupported uncertainty method '{method}'.")


def build_diversity_selector(
    config: Mapping[str, Any] | None = None,
) -> BaseDiversitySelector | None:
    if config is None:
        return None
    cfg = dict(config)
    if not bool(cfg.get("enabled", True)):
        return None

    method = str(cfg.get("method", "kcenter_greedy")).lower()
    if method == "kcenter_greedy":
        return KCenterGreedySelector(
            uncertainty_weight=float(cfg.get("uncertainty_weight", 0.0))
        )
    if method in {"kmeans_representative", "cluster", "cluster_balanced"}:
        rep_method = str(cfg.get("representative_method", "medoid")).lower()
        method_map = {"centroid": "centroid", "medoid": "medoid"}
        return KMeansRepresentativeSelector(method=method_map.get(rep_method, "medoid"))
    if method == "farthest_first":
        return FarthestFirstSelector()

    raise ValueError(f"Unsupported diversity selector method '{method}'.")


def build_query_strategy(
    config: Mapping[str, Any] | None = None,
    *,
    uncertainty_config: Mapping[str, Any] | None = None,
    diversity_config: Mapping[str, Any] | None = None,
    filtering_config: Mapping[str, Any] | None = None,
    models: Sequence[Any] | None = None,
    model_loader: Callable[[str], Any] | None = None,
) -> QueryStrategy:
    if config is None:
        return RandomQueryStrategy()

    strategy_cfg = dict(config)
    name = str(strategy_cfg.get("name", "random")).lower()

    pool_filter = build_pool_filter(filtering_config)
    resolved_uncertainty_cfg: Mapping[str, Any] | None = uncertainty_config
    if resolved_uncertainty_cfg is None and name in {
        "mc_dropout_uncertainty",
        "ensemble_uncertainty",
    }:
        resolved_uncertainty_cfg = {"method": name}

    uncertainty_estimator = build_uncertainty_estimator(
        resolved_uncertainty_cfg, models=models, model_loader=model_loader
    )
    diversity_selector = build_diversity_selector(diversity_config)

    if name == "random":
        return RandomQueryStrategy()
    if name == "uncertainty":
        method = str(strategy_cfg.get("method", "least_confidence"))
        return UncertaintyQueryStrategy(method=method)
    if name in {"representative_init", "representative"}:
        return RepresentativeInitStrategy(
            selection_method=str(strategy_cfg.get("selection_method", "kmeans_medoid")),
            embedding_layer=str(
                strategy_cfg.get("embedding_layer", "encoder_global_pool")
            ),
            embedding_cache_path=strategy_cfg.get("embedding_cache_path"),
            batch_size=int(strategy_cfg.get("batch_size", 64)),
            pool_filter=pool_filter,
            seed=strategy_cfg.get("seed"),
        )
    if name in {
        "hybrid_uncertainty_diversity",
        "topk_uncertainty_then_kcenter",
        "topk_uncertainty_then_cluster",
    }:
        mode = str(strategy_cfg.get("mode", name))
        return HybridUncertaintyDiversityStrategy(
            uncertainty_estimator=uncertainty_estimator,
            diversity_selector=diversity_selector,
            pool_filter=pool_filter,
            mode=mode,
            candidate_pool_size=int(
                strategy_cfg.get("top_k", strategy_cfg.get("candidate_pool_size", 1000))
            ),
            alpha=float(strategy_cfg.get("alpha", 0.7)),
            beta=float(strategy_cfg.get("beta", 0.3)),
            gamma=float(strategy_cfg.get("gamma", 0.0)),
            embedding_layer=str(
                strategy_cfg.get("embedding_layer", "encoder_global_pool")
            ),
            embedding_cache_path=strategy_cfg.get("embedding_cache_path"),
            artifact_dir=strategy_cfg.get("artifact_dir"),
            seed=strategy_cfg.get("seed"),
        )
    if name in {"mc_dropout_uncertainty", "ensemble_uncertainty"}:
        return UncertaintyEstimatorQueryStrategy(
            uncertainty_estimator=uncertainty_estimator,
            pool_filter=pool_filter,
        )

    raise ValueError(f"Unsupported query strategy '{name}'.")
