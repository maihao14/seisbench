from __future__ import annotations

import csv
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Sequence

import numpy as np

STATE_SCHEMA_VERSION = 1


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

        scores = np.asarray(self.score_fn(candidates), dtype=np.float64)
        self._validate_scores(scores, n_candidates=candidates.size)

        selected = np.asarray(
            self.strategy.select(candidates, budget, scores, self.pool.rng),
            dtype=np.int64,
        )
        self._validate_selection(selected, candidates, expected_size=budget)

        # Atomic state update: labeling callback happens before mutating pool indices.
        self.label_fn(selected.copy())

        score_lookup = {
            int(idx): float(score)
            for idx, score in zip(candidates.tolist(), scores.tolist())
        }
        selected_scores = np.array([score_lookup[int(idx)] for idx in selected])

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
