from __future__ import annotations

import numpy as np
import pytest

from cnlp_llm.utils.annotation_tools import (
    ObservedEvalPerformance,
    SampledConfusionRates,
)


class TestConfusionRatesValidation:
    def test_valid_rates_construct_successfully(self) -> None:
        rates = SampledConfusionRates(p_tp=0.4, p_tn=0.3, p_fp=0.1, p_fn=0.2)
        assert rates.p_tp == 0.4

    def test_rates_must_sum_to_one(self) -> None:
        with pytest.raises(ValueError, match="sum to 1"):
            SampledConfusionRates(p_tp=0.4, p_tn=0.4, p_fp=0.4, p_fn=0.4)

    def test_rates_just_below_one_rejected(self) -> None:
        with pytest.raises(ValueError, match="sum to 1"):
            SampledConfusionRates(p_tp=0.25, p_tn=0.25, p_fp=0.25, p_fn=0.24)

    def test_small_floating_point_drift_accepted(self) -> None:
        SampledConfusionRates(p_tp=0.1 + 0.2, p_tn=0.3, p_fp=0.2, p_fn=0.2)

    def test_negative_rate_rejected(self) -> None:
        with pytest.raises(ValueError, match="negative"):
            SampledConfusionRates(p_tp=-0.1, p_tn=0.5, p_fp=0.3, p_fn=0.3)


class TestConfusionRatesDerivedQuantities:
    @pytest.fixture
    def rates(self) -> SampledConfusionRates:
        # Cells chosen so derived quantities have clean values:
        # p_gold(True) = 0.5, p_pred(True) = 0.5, tpr = 0.8
        return SampledConfusionRates(p_tp=0.4, p_tn=0.4, p_fp=0.1, p_fn=0.1)

    def test_tpr(self, rates: SampledConfusionRates) -> None:
        assert rates.tpr == pytest.approx(0.8)

    def test_fnr_complements_tpr(self, rates: SampledConfusionRates) -> None:
        assert rates.fnr == pytest.approx(1 - rates.tpr)

    def test_p_gold_marginals_sum_to_one(self, rates: SampledConfusionRates) -> None:
        assert rates.p_gold(True) + rates.p_gold(False) == pytest.approx(1.0)

    def test_p_pred_marginals_sum_to_one(self, rates: SampledConfusionRates) -> None:
        assert rates.p_pred(True) + rates.p_pred(False) == pytest.approx(1.0)

    def test_p_gold_true(self, rates: SampledConfusionRates) -> None:
        assert rates.p_gold(True) == pytest.approx(0.5)

    def test_p_pred_true(self, rates: SampledConfusionRates) -> None:
        assert rates.p_pred(True) == pytest.approx(0.5)


class TestPGoldGivenPred:
    @pytest.fixture
    def rates(self) -> SampledConfusionRates:
        return SampledConfusionRates(p_tp=0.4, p_tn=0.4, p_fp=0.1, p_fn=0.1)

    def test_matches_direct_calculation_when_pred_true(
        self, rates: SampledConfusionRates
    ) -> None:
        # P(gold=1 | pred=1) reduces algebraically to p_tp / (p_tp + p_fp)
        expected = rates.p_tp / (rates.p_tp + rates.p_fp)
        assert rates.p_gold_given_pred(gold=True, pred=True) == pytest.approx(expected)

    def test_matches_direct_calculation_when_pred_false(
        self, rates: SampledConfusionRates
    ) -> None:
        # P(gold=1 | pred=0) reduces algebraically to p_fn / (p_fn + p_tn)
        expected = rates.p_fn / (rates.p_fn + rates.p_tn)
        assert rates.p_gold_given_pred(gold=True, pred=False) == pytest.approx(expected)

    @pytest.mark.parametrize("pred", [True, False])
    def test_gold_branches_complement(
        self, rates: SampledConfusionRates, pred: bool
    ) -> None:
        p_true = rates.p_gold_given_pred(gold=True, pred=pred)
        p_false = rates.p_gold_given_pred(gold=False, pred=pred)
        assert p_true + p_false == pytest.approx(1.0)


class TestSampleConfusionRates:
    @pytest.fixture
    def observed(self) -> ObservedEvalPerformance:
        return ObservedEvalPerformance(tp=80, tn=85, fp=15, fn=20)

    def test_returns_requested_number_of_samples(
        self, observed: ObservedEvalPerformance
    ) -> None:
        samples = observed.sample_confusion_rates(n=37)
        assert len(samples) == 37

    def test_default_n_returns_one_sample(
        self, observed: ObservedEvalPerformance
    ) -> None:
        samples = observed.sample_confusion_rates()
        assert len(samples) == 1

    def test_each_sample_is_valid_confusion_rates(
        self, observed: ObservedEvalPerformance
    ) -> None:
        samples = observed.sample_confusion_rates(n=10)
        assert all(isinstance(s, SampledConfusionRates) for s in samples)

    def test_seeded_sampling_is_reproducible(
        self, observed: ObservedEvalPerformance
    ) -> None:
        s1 = observed.sample_confusion_rates(n=5, rng=np.random.default_rng(42))
        s2 = observed.sample_confusion_rates(n=5, rng=np.random.default_rng(42))
        for a, b in zip(s1, s2, strict=True):
            assert a == b

    def test_unseeded_calls_produce_different_samples(
        self, observed: ObservedEvalPerformance
    ) -> None:
        s1 = observed.sample_confusion_rates(n=5)
        s2 = observed.sample_confusion_rates(n=5)
        assert s1 != s2

    def test_posterior_mean_matches_closed_form(
        self, observed: ObservedEvalPerformance
    ) -> None:
        # For Dirichlet(alpha), E[p_i] = alpha_i / sum(alpha). With uniform prior (1,1,1,1)
        # and counts (80, 85, 15, 20), posterior is Dirichlet(81, 86, 16, 21).
        rng = np.random.default_rng(0)
        samples = observed.sample_confusion_rates(n=20_000, rng=rng)

        alpha = np.array([81, 86, 16, 21])
        expected_means = alpha / alpha.sum()

        sample_means = np.array(
            [
                np.mean([s.p_tp for s in samples]),
                np.mean([s.p_tn for s in samples]),
                np.mean([s.p_fp for s in samples]),
                np.mean([s.p_fn for s in samples]),
            ]
        )
        np.testing.assert_allclose(sample_means, expected_means, atol=0.005)

    def test_strong_prior_pulls_toward_uniform(self) -> None:
        # Tiny eval set + strong uniform prior → posterior mean near 0.25 for each cell.
        observed = ObservedEvalPerformance(tp=2, tn=2, fp=1, fn=1)
        rng = np.random.default_rng(0)
        samples = observed.sample_confusion_rates(
            n=10_000, prior_alpha=(100.0, 100.0, 100.0, 100.0), rng=rng
        )
        mean_p_tp = np.mean([s.p_tp for s in samples])
        assert mean_p_tp == pytest.approx(0.25, abs=0.01)

    def test_weak_prior_lets_data_dominate(self) -> None:
        # Same data, weak prior → posterior mean near MLE.
        observed = ObservedEvalPerformance(tp=80, tn=85, fp=15, fn=20)
        rng = np.random.default_rng(0)
        samples = observed.sample_confusion_rates(n=10_000, rng=rng)
        mle_p_tp = 80 / 200
        mean_p_tp = np.mean([s.p_tp for s in samples])
        assert mean_p_tp == pytest.approx(mle_p_tp, abs=0.005)


class TestEstimatePrevalenceDistribution:
    @pytest.fixture
    def observed(self) -> ObservedEvalPerformance:
        return ObservedEvalPerformance(tp=80, tn=85, fp=15, fn=20)

    def test_returns_array_of_requested_length(
        self, observed: ObservedEvalPerformance
    ) -> None:
        result = observed.estimate_prevalence_from_predictions(
            predictions=[True, False, True], n_samples=50
        )
        assert result.shape == (50,)

    def test_all_prevalences_in_unit_interval(
        self, observed: ObservedEvalPerformance
    ) -> None:
        result = observed.estimate_prevalence_from_predictions(
            predictions=[True, False] * 100, n_samples=200
        )
        assert ((result >= 0) & (result <= 1)).all()

    def test_seeded_runs_are_reproducible(
        self, observed: ObservedEvalPerformance
    ) -> None:
        preds = [True, False, True, True, False] * 20
        a = observed.estimate_prevalence_from_predictions(
            predictions=preds,
            n_samples=100,
            rng=np.random.default_rng(7),
        )
        b = observed.estimate_prevalence_from_predictions(
            predictions=preds,
            n_samples=100,
            rng=np.random.default_rng(7),
        )
        np.testing.assert_array_equal(a, b)

    def test_recovers_true_prevalence_with_perfect_classifier(self) -> None:
        # A classifier with no errors should yield prevalence ≈ observed pred rate.
        # 1000 eval samples, all correct: tp=300, tn=700, fp=0, fn=0.
        observed = ObservedEvalPerformance(tp=300, tn=700, fp=0, fn=0)
        # Deployment: 40% predicted positive. With a perfect classifier, true
        # prevalence should be ≈ 40%.
        preds = [True] * 400 + [False] * 600
        result = observed.estimate_prevalence_from_predictions(
            predictions=preds,
            n_samples=2000,
            rng=np.random.default_rng(0),
        )
        assert result.mean() == pytest.approx(0.40, abs=0.01)

    def test_corrects_for_known_error_rates(self) -> None:
        # Construct a scenario where the raw prediction rate differs meaningfully
        # from the corrected prevalence. Classifier with TPR=0.8, specificity=0.9,
        # evaluated on a balanced set.
        # Eval: 100 positives (80 tp, 20 fn), 100 negatives (90 tn, 10 fp).
        observed = ObservedEvalPerformance(tp=80, tn=90, fp=10, fn=20)
        # Deployment: assume true prevalence is 0.5. Expected predicted-positive rate:
        # P(pred=1) = TPR * π + FPR * (1 - π) = 0.8 * 0.5 + 0.1 * 0.5 = 0.45
        # So generate predictions with that rate.
        rng = np.random.default_rng(0)
        n = 5000
        true_labels = rng.random(n) < 0.5
        preds = np.where(
            true_labels,
            rng.random(n) < 0.8,  # TPR
            rng.random(n) < 0.1,  # FPR
        )
        result = observed.estimate_prevalence_from_predictions(
            predictions=preds.tolist(),
            n_samples=2000,
            rng=np.random.default_rng(1),
        )
        # Raw predicted-positive rate is ~0.45, but corrected prevalence
        # should be near the true 0.5.
        raw_rate = preds.mean()
        assert raw_rate == pytest.approx(0.45, abs=0.02)
        assert result.mean() == pytest.approx(0.50, abs=0.02)

    def test_more_eval_data_narrows_distribution(self) -> None:
        # Same model error profile, but 10x more eval samples → tighter posterior.
        small_eval = ObservedEvalPerformance(tp=8, tn=9, fp=1, fn=2)
        large_eval = ObservedEvalPerformance(tp=80, tn=90, fp=10, fn=20)
        preds = [True] * 500 + [False] * 500

        small = small_eval.estimate_prevalence_from_predictions(
            predictions=preds,
            n_samples=2000,
            rng=np.random.default_rng(0),
        )
        large = large_eval.estimate_prevalence_from_predictions(
            predictions=preds,
            n_samples=2000,
            rng=np.random.default_rng(0),
        )
        assert small.std() > large.std()

    def test_empty_predictions_raises_or_returns_empty_safely(
        self, observed: ObservedEvalPerformance
    ) -> None:
        # Edge case: no predictions to estimate from. The mean of zero samples
        # is NaN; we should at least not crash, and the result should reflect
        # that nothing meaningful can be computed.
        with pytest.warns():
            result = observed.estimate_prevalence_from_predictions(
                predictions=[], n_samples=10
            )
        assert result.shape == (10,)
        assert np.isnan(result).all()
