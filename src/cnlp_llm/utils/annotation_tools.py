from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, kw_only=True)
class ObservedEvalPerformance:
    tp: int
    tn: int
    fp: int
    fn: int

    def sample_confusion_rates(
        self,
        n: int = 1,
        prior_alpha: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
        rng: np.random.Generator | None = None,
    ) -> list["SampledConfusionRates"]:
        if rng is None:
            rng = np.random.default_rng()

        concentration = np.array(prior_alpha) + np.array(
            [self.tp, self.tn, self.fp, self.fn]
        )
        samples = rng.dirichlet(concentration, size=n)

        return [
            SampledConfusionRates(p_tp=p_tp, p_tn=p_tn, p_fp=p_fp, p_fn=p_fn)
            for p_tp, p_tn, p_fp, p_fn in samples
        ]

    def estimate_prevalence_from_predictions(
        self,
        predictions: Sequence[bool],
        n_samples: int = 1000,
        prior_alpha: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()

        preds = np.asarray(predictions, dtype=bool)
        rate_samples = self.sample_confusion_rates(
            n=n_samples, prior_alpha=prior_alpha, rng=rng
        )

        prevalences = np.empty(n_samples)
        for i, rates in enumerate(rate_samples):
            p_gold_if_pred_true = rates.p_gold_given_pred(gold=True, pred=True)
            p_gold_if_pred_false = rates.p_gold_given_pred(gold=True, pred=False)

            per_instance_p = np.where(preds, p_gold_if_pred_true, p_gold_if_pred_false)
            sampled_labels = rng.random(len(preds)) < per_instance_p
            prevalences[i] = sampled_labels.mean()

        return prevalences


@dataclass(frozen=True, kw_only=True)
class SampledConfusionRates:
    p_tp: float
    p_tn: float
    p_fp: float
    p_fn: float

    def __post_init__(self):
        if abs((self.p_tp + self.p_tn + self.p_fp + self.p_fn) - 1) > 1e-6:
            raise ValueError("confusion rates must sum to 1")
        if any(x < 0 for x in (self.p_tp, self.p_tn, self.p_fp, self.p_fn)):
            raise ValueError("confusion rates must not be negative")

    @property
    def tpr(self) -> float:
        return self.p_tp / (self.p_tp + self.p_fn)

    @property
    def fnr(self) -> float:
        return self.p_fn / (self.p_tp + self.p_fn)

    def p_gold(self, gold: bool) -> float:
        if gold:
            return self.p_tp + self.p_fn
        else:
            return self.p_fp + self.p_tn

    def p_pred(self, pred: bool) -> float:
        if pred:
            return self.p_tp + self.p_fp
        else:
            return self.p_tn + self.p_fn

    def p_gold_given_pred(self, gold: bool, pred: bool) -> float:
        if pred:
            # P(gold=1|pred=1) = P(pred=1|gold=1) * P(gold=1) / P(pred=1)
            p = self.tpr * self.p_gold(True) / self.p_pred(True)
        else:
            # P(gold=1|pred=0) = P(pred=0|gold=1) * P(gold=1) / P(pred=0)
            p = self.fnr * self.p_gold(True) / self.p_pred(False)

        if not gold:
            p = 1 - p

        return p
