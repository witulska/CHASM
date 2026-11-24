import numpy as np

from chasm import (
    ARTICLE_METRICS,
    bootstraping_one_dim_stats,
    chasm_bounds_main,
    chasm_main,
)

EXPECTED_KEYS = {"decision", "upper_bound", "lower_bound", "stat_value", "imf"}


def test_chasm_returns_brainlab_schema_for_all_metrics():
    rng = np.random.default_rng(42)
    segment1 = rng.normal(size=(30, 2))
    segment2 = rng.normal(loc=0.5, size=(30, 2))
    results = chasm_main(segment1, segment2, n_resamples=3)

    assert {"sig_l_mean", "sig_robust_mean", "sig_spearman_corr", "sig_kl_divergence"} <= set(results)
    for value in results.values():
        assert set(value) == EXPECTED_KEYS
        assert value["decision"] in (0, 1)
        assert isinstance(value["imf"], float)


def test_one_dimensional_data_keeps_correlation_result_well_formed():
    rng = np.random.default_rng(7)
    segment1 = rng.normal(size=(20, 1))
    segment2 = rng.normal(size=(20, 1))
    result = chasm_main(segment1, segment2, n_resamples=2)["sig_spearman_corr"]
    assert set(result) == EXPECTED_KEYS
    assert result["decision"] == 0


def test_custom_mean_statistic_and_article_metric_selection():
    rng = np.random.default_rng(11)
    segment1 = rng.normal(size=(24, 2))
    segment2 = rng.normal(loc=0.5, size=(24, 2))

    def classical_mean_score(
        first: np.ndarray,
        second: np.ndarray,
        axis: int | None = None,
    ) -> np.ndarray:
        difference = np.abs(np.mean(first, axis=axis) - np.mean(second, axis=axis))
        return np.array([-np.log(difference + np.finfo(float).eps)])

    mean_result = bootstraping_one_dim_stats(
        segment1,
        segment2,
        statistic=classical_mean_score,
        n_resamples=2,
    )
    assert set(mean_result) == EXPECTED_KEYS

    article_results = chasm_bounds_main(segment1, segment2, n_resamples=3)
    assert tuple(article_results) == ARTICLE_METRICS
