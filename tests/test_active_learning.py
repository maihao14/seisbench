import csv

import numpy as np
import pytest

import seisbench.generate as sbg


def test_pool_initialization_and_defaults():
    pool = sbg.ActiveLearningPool(dataset_size=6, labeled_indices=[0, 2], seed=7)

    assert pool.round == 0
    assert pool.seed == 7
    assert np.array_equal(pool.labeled_indices, np.array([0, 2]))
    assert np.array_equal(pool.unlabeled_indices, np.array([1, 3, 4, 5]))


def test_pool_validation_errors():
    with pytest.raises(ValueError, match="disjoint"):
        sbg.ActiveLearningPool(
            dataset_size=5, labeled_indices=[0, 1], unlabeled_indices=[1, 2, 3, 4]
        )

    with pytest.raises(ValueError, match="duplicate"):
        sbg.ActiveLearningPool(dataset_size=5, labeled_indices=[0, 0])

    with pytest.raises(ValueError, match="out-of-range"):
        sbg.ActiveLearningPool(dataset_size=5, labeled_indices=[5])

    with pytest.raises(ValueError, match="full partition"):
        sbg.ActiveLearningPool(
            dataset_size=5, labeled_indices=[0], unlabeled_indices=[1, 2, 3]
        )


def test_random_query_strategy_reproducible():
    strategy = sbg.RandomQueryStrategy()
    candidates = np.arange(10)
    scores = np.zeros(10)

    selected_a = strategy.select(candidates, 4, scores, np.random.default_rng(42))
    selected_b = strategy.select(candidates, 4, scores, np.random.default_rng(42))

    assert np.array_equal(selected_a, selected_b)
    assert selected_a.size == 4


def test_uncertainty_score_formulas():
    probs = np.array(
        [
            [0.9, 0.1],
            [0.5, 0.5],
            [0.6, 0.4],
        ]
    )

    least_conf = sbg.UncertaintyQueryStrategy.score_probabilities(
        probs, method="least_confidence"
    )
    margin = sbg.UncertaintyQueryStrategy.score_probabilities(probs, method="margin")
    entropy = sbg.UncertaintyQueryStrategy.score_probabilities(probs, method="entropy")

    assert np.allclose(least_conf, np.array([0.1, 0.5, 0.4]), atol=1e-8)
    assert np.allclose(margin, np.array([0.2, 1.0, 0.8]), atol=1e-8)

    expected_entropy = -np.sum(probs * np.log(probs), axis=1)
    assert np.allclose(entropy, expected_entropy, atol=1e-8)


def test_uncertainty_strategy_tie_break_stable():
    strategy = sbg.UncertaintyQueryStrategy(method="least_confidence")

    candidates = np.array([10, 2, 5, 1], dtype=np.int64)
    scores = np.array([0.8, 0.8, 0.8, 0.9], dtype=np.float64)

    selected = strategy.select(
        candidates, budget=3, scores=scores, rng=np.random.default_rng(0)
    )

    assert np.array_equal(selected, np.array([1, 2, 5]))


def test_loop_budget_clamp():
    pool = sbg.ActiveLearningPool(dataset_size=4, labeled_indices=[0], seed=1)

    selected_calls = []

    def score_fn(indices):
        return indices.astype(np.float64)

    def label_fn(selected_indices):
        selected_calls.append(selected_indices.copy())

    loop = sbg.ActiveLearningLoop(
        pool,
        score_fn=score_fn,
        label_fn=label_fn,
        strategy=sbg.UncertaintyQueryStrategy(method="least_confidence"),
    )
    result = loop.run_round(budget=99)

    assert result.query_size == 3
    assert result.labeled_size == 4
    assert result.unlabeled_size == 0
    assert pool.round == 1
    assert len(selected_calls) == 1


def test_atomic_rollback_when_label_callback_fails():
    pool = sbg.ActiveLearningPool(dataset_size=5, labeled_indices=[0], seed=3)

    old_labeled = pool.labeled_indices.copy()
    old_unlabeled = pool.unlabeled_indices.copy()

    def score_fn(indices):
        return indices.astype(np.float64)

    def label_fn(_selected_indices):
        raise RuntimeError("label failed")

    loop = sbg.ActiveLearningLoop(
        pool,
        score_fn=score_fn,
        label_fn=label_fn,
        strategy=sbg.UncertaintyQueryStrategy(method="least_confidence"),
    )

    with pytest.raises(RuntimeError, match="label failed"):
        loop.run_round(budget=2)

    assert pool.round == 0
    assert np.array_equal(pool.labeled_indices, old_labeled)
    assert np.array_equal(pool.unlabeled_indices, old_unlabeled)


def test_fit_callback_order_and_payload():
    pool = sbg.ActiveLearningPool(dataset_size=5, labeled_indices=[0], seed=4)
    calls = []

    def score_fn(indices):
        calls.append("score")
        return indices.astype(np.float64)

    def label_fn(selected_indices):
        calls.append("label")
        assert selected_indices.size == 2

    def fit_fn(fit_pool, round_result):
        calls.append("fit")
        assert fit_pool is pool
        assert round_result.round == pool.round
        assert round_result.query_size == 2

    loop = sbg.ActiveLearningLoop(
        pool,
        score_fn=score_fn,
        label_fn=label_fn,
        fit_fn=fit_fn,
        strategy=sbg.UncertaintyQueryStrategy(method="least_confidence"),
    )
    loop.run_round(budget=2)

    assert calls == ["score", "label", "fit"]


def test_pool_save_load_round_trip(tmp_path):
    pool = sbg.ActiveLearningPool(dataset_size=6, labeled_indices=[0], seed=13)

    def score_fn(indices):
        return indices.astype(np.float64)

    def label_fn(_selected_indices):
        return None

    strategy = sbg.UncertaintyQueryStrategy(method="entropy")
    loop = sbg.ActiveLearningLoop(
        pool, score_fn=score_fn, label_fn=label_fn, strategy=strategy
    )
    loop.run_round(budget=2)

    state_path = tmp_path / "al_state.json"
    pool.save(str(state_path))
    loaded = sbg.ActiveLearningPool.load(str(state_path))

    assert loaded.dataset_size == pool.dataset_size
    assert loaded.seed == pool.seed
    assert loaded.round == pool.round
    assert np.array_equal(loaded.labeled_indices, pool.labeled_indices)
    assert np.array_equal(loaded.unlabeled_indices, pool.unlabeled_indices)
    assert loaded.strategy_config == pool.strategy_config
    assert len(loaded.history_metadata) == len(pool.history_metadata)


def test_metrics_export_append_on_resume(tmp_path):
    def score_fn(indices):
        return indices.astype(np.float64)

    def label_fn(_selected_indices):
        return None

    strategy = sbg.UncertaintyQueryStrategy(method="least_confidence")
    metrics_path = tmp_path / "metrics.csv"
    state_path = tmp_path / "state.json"

    pool = sbg.ActiveLearningPool(dataset_size=6, labeled_indices=[0], seed=1)
    loop = sbg.ActiveLearningLoop(
        pool, score_fn=score_fn, label_fn=label_fn, strategy=strategy
    )
    loop.run_round(budget=2)
    loop.export_metrics(str(metrics_path))
    pool.save(str(state_path))

    resumed_pool = sbg.ActiveLearningPool.load(str(state_path))
    resumed_loop = sbg.ActiveLearningLoop(
        resumed_pool, score_fn=score_fn, label_fn=label_fn, strategy=strategy
    )
    resumed_loop.run_round(budget=2)
    resumed_loop.export_metrics(str(metrics_path))
    resumed_loop.export_metrics(str(metrics_path))

    with metrics_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 2
    assert [int(row["round"]) for row in rows] == [1, 2]


def test_integration_multi_round_mocked_callbacks():
    pool = sbg.ActiveLearningPool(dataset_size=5, labeled_indices=[0], seed=9)

    scores = {1: 0.2, 2: 0.9, 3: 0.8, 4: 0.1}
    labeled_log = []

    def score_fn(indices):
        return np.array([scores[int(idx)] for idx in indices], dtype=np.float64)

    def label_fn(selected_indices):
        labeled_log.append(selected_indices.copy())

    loop = sbg.ActiveLearningLoop(
        pool,
        score_fn=score_fn,
        label_fn=label_fn,
        strategy=sbg.UncertaintyQueryStrategy(method="least_confidence"),
    )

    round1 = loop.run_round(budget=2)
    round2 = loop.run_round(budget=2)

    assert round1.query_size == 2
    assert round2.query_size == 2
    assert pool.unlabeled_indices.size == 0
    assert len(labeled_log) == 2
    assert np.array_equal(np.sort(np.concatenate(labeled_log)), np.array([1, 2, 3, 4]))
