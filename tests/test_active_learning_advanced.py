import numpy as np
import torch

import seisbench.generate as sbg


class DummyPoolDataset:
    def __init__(self, waveforms, metadata=None):
        self.waveforms = [np.asarray(w, dtype=np.float32) for w in waveforms]
        if metadata is None:
            metadata = [{} for _ in self.waveforms]
        self.metadata = metadata

    def __len__(self):
        return len(self.waveforms)

    def get_sample(self, idx):
        return self.waveforms[int(idx)], self.metadata[int(idx)]


class TinyPhaseModel(torch.nn.Module):
    def __init__(self, n_channels=1, n_samples=32, n_outputs=16, p_dropout=0.2):
        super().__init__()
        self.flatten = torch.nn.Flatten()
        self.encoder_linear = torch.nn.Linear(n_channels * n_samples, 24)
        self.encoder_act = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=p_dropout)
        self.head = torch.nn.Linear(24, n_outputs)

    def extract_embedding(self, x):
        x = self.flatten(x)
        x = self.encoder_linear(x)
        return self.encoder_act(x)

    def forward(self, x):
        z = self.extract_embedding(x)
        z = self.dropout(z)
        logits = self.head(z)
        return torch.softmax(logits, dim=-1)


def _make_dataset(n=8, seed=7):
    rng = np.random.default_rng(seed)
    waves = [rng.normal(size=(1, 32)).astype(np.float32) for _ in range(n)]
    metadata = [{"event_id": i // 2, "station_code": f"S{i % 3}"} for i in range(n)]
    return DummyPoolDataset(waves, metadata)


def test_embedding_extractor_alignment_and_cache(tmp_path):
    dataset = _make_dataset(n=6)
    model = TinyPhaseModel()
    extractor = sbg.EmbeddingExtractor(batch_size=2)

    indices = np.array([5, 1, 3], dtype=np.int64)
    cache_path = tmp_path / "embeddings.npy"

    emb_a = extractor.extract(
        model=model,
        dataset=dataset,
        indices=indices,
        layer="encoder_global_pool",
        cache_path=str(cache_path),
    )
    emb_b = extractor.extract(
        model=model,
        dataset=dataset,
        indices=indices,
        layer="encoder_global_pool",
        cache_path=str(cache_path),
    )

    assert emb_a.shape[0] == indices.size
    assert np.allclose(emb_a, emb_b)


def test_pool_filter_low_energy_rejects_flat_traces():
    waves = np.array(
        [
            np.zeros((1, 32), dtype=np.float32),
            np.ones((1, 32), dtype=np.float32) * 1e-6,
            np.ones((1, 32), dtype=np.float32) * 0.1,
        ]
    )
    pool_filter = sbg.PoolFilter(
        enabled=True,
        low_energy_enabled=True,
        rms_threshold=1e-4,
        abnormal_enabled=False,
        duplicate_enabled=False,
    )

    result = pool_filter.apply(waveforms=waves)
    assert set(result.rejected_indices.tolist()) == {0, 1}
    assert set(result.valid_indices.tolist()) == {2}


def test_pool_filter_abnormal_rejects_nan_inf():
    waves = np.array(
        [
            np.full((1, 32), np.nan, dtype=np.float32),
            np.full((1, 32), np.inf, dtype=np.float32),
            np.random.default_rng(0).normal(size=(1, 32)).astype(np.float32),
        ]
    )
    pool_filter = sbg.PoolFilter(
        enabled=True,
        low_energy_enabled=False,
        abnormal_enabled=True,
        duplicate_enabled=False,
    )

    result = pool_filter.apply(waveforms=waves)
    assert set(result.rejected_indices.tolist()) == {0, 1}
    assert set(result.valid_indices.tolist()) == {2}


def test_pool_filter_duplicate_exact_hash():
    base = np.random.default_rng(1).normal(size=(1, 32)).astype(np.float32)
    waves = np.array([base, base.copy(), base + 0.1], dtype=np.float32)
    pool_filter = sbg.PoolFilter(
        enabled=True,
        low_energy_enabled=False,
        abnormal_enabled=False,
        duplicate_enabled=True,
        duplicate_method="exact_hash",
    )

    result = pool_filter.apply(waveforms=waves)
    assert set(result.rejected_indices.tolist()) == {1}
    assert set(result.valid_indices.tolist()) == {0, 2}


def test_mc_dropout_uncertainty_score_length_matches_pool():
    dataset = _make_dataset(n=7)
    model = TinyPhaseModel(p_dropout=0.4)
    estimator = sbg.MCDropoutUncertainty(
        n_passes=5,
        score_metric="pick_time_variance",
        batch_size=3,
    )

    scores, diagnostics = estimator.score(model=model, dataset=dataset)
    assert scores.shape == (len(dataset),)
    assert np.all(np.isfinite(scores))
    assert diagnostics["mean_prediction"].shape[0] == len(dataset)


def test_ensemble_uncertainty_with_dummy_models():
    dataset = _make_dataset(n=6)
    models = [TinyPhaseModel() for _ in range(3)]
    estimator = sbg.EnsembleUncertainty(
        models=models,
        score_metric="js_divergence",
        batch_size=2,
    )

    scores, diagnostics = estimator.score(model=models[0], dataset=dataset)
    assert scores.shape == (len(dataset),)
    assert np.all(np.isfinite(scores))
    assert diagnostics["mean_prediction"].shape[0] == len(dataset)


def test_kcenter_selector_returns_unique_indices():
    rng = np.random.default_rng(0)
    candidates = np.arange(15, dtype=np.int64)
    embeddings = rng.normal(size=(15, 4))
    scores = rng.uniform(size=15)

    selector = sbg.KCenterGreedySelector(uncertainty_weight=0.1)
    selected = selector.select(
        candidate_indices=candidates,
        embeddings=embeddings,
        n_select=6,
        uncertainty_scores=scores,
        rng=np.random.default_rng(3),
    )

    assert selected.size == 6
    assert np.unique(selected).size == 6
    assert np.all(np.isin(selected, candidates))


def test_representative_strategy_returns_seed_count():
    dataset = _make_dataset(n=12)
    model = TinyPhaseModel()
    strategy = sbg.RepresentativeInitStrategy(
        selection_method="kmeans_medoid",
        pool_filter=sbg.PoolFilter(enabled=False),
        seed=11,
    )

    selected = strategy.select_batch(
        model=model,
        unlabeled_dataset=dataset,
        batch_size=5,
    )

    assert selected.size == 5
    assert np.unique(selected).size == 5
    assert np.all(np.isin(selected, np.arange(len(dataset))))


def test_hybrid_strategy_excludes_rejected_samples():
    rng = np.random.default_rng(4)
    waves = [np.zeros((1, 32), dtype=np.float32) for _ in range(2)]
    waves += [rng.normal(size=(1, 32)).astype(np.float32) for _ in range(8)]
    dataset = DummyPoolDataset(waves)

    strategy = sbg.HybridUncertaintyDiversityStrategy(
        uncertainty_estimator=sbg.SoftmaxUncertainty(metric="predictive_entropy"),
        diversity_selector=sbg.KCenterGreedySelector(),
        pool_filter=sbg.PoolFilter(
            enabled=True,
            low_energy_enabled=True,
            rms_threshold=1e-4,
            abnormal_enabled=False,
            duplicate_enabled=False,
        ),
        candidate_pool_size=6,
        mode="topk_uncertainty_then_kcenter",
        seed=5,
    )

    model = TinyPhaseModel()
    selected = strategy.select_batch(
        model=model,
        unlabeled_dataset=dataset,
        batch_size=4,
    )

    assert selected.size == 4
    assert np.unique(selected).size == 4
    assert 0 not in selected and 1 not in selected


def test_smoke_active_learning_loop_with_hybrid_strategy():
    dataset = _make_dataset(n=10)
    model = TinyPhaseModel()
    pool = sbg.ActiveLearningPool(
        dataset_size=len(dataset), labeled_indices=[0], seed=9
    )

    labeled_calls = []

    def score_fn(indices):
        return np.zeros(indices.shape[0], dtype=np.float64)

    def label_fn(selected_indices):
        labeled_calls.append(selected_indices.copy())

    strategy = sbg.HybridUncertaintyDiversityStrategy(
        uncertainty_estimator=sbg.SoftmaxUncertainty(metric="predictive_entropy"),
        diversity_selector=sbg.KCenterGreedySelector(),
        pool_filter=sbg.PoolFilter(enabled=False),
        candidate_pool_size=8,
        mode="topk_uncertainty_then_kcenter",
        seed=13,
    )

    loop = sbg.ActiveLearningLoop(
        pool=pool,
        score_fn=score_fn,
        label_fn=label_fn,
        strategy=strategy,
        model=model,
        pool_dataset=dataset,
    )

    result = loop.run_round(budget=3)

    assert result.query_size == 3
    assert np.unique(result.selected_indices).size == 3
    assert np.all(np.isin(result.selected_indices, np.arange(1, len(dataset))))
    assert pool.labeled_indices.size == 4
    assert len(labeled_calls) == 1
