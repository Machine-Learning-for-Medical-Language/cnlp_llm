import click
import numpy as np

from ..utils.annotation_tools import ObservedEvalPerformance


def _ci_half_width(
    n_annotations: int,
    assumed_p_gold: float,
    assumed_tpr: float,
    assumed_fpr: float,
    test_set_size: int,
    confidence: float,
    n_prevalence_samples: int,
    rng: np.random.Generator,
) -> float:
    tp = round(n_annotations * assumed_p_gold * assumed_tpr)
    fn = round(n_annotations * assumed_p_gold * (1 - assumed_tpr))
    fp = round(n_annotations * (1 - assumed_p_gold) * assumed_fpr)
    tn = round(n_annotations * (1 - assumed_p_gold) * (1 - assumed_fpr))
    observed = ObservedEvalPerformance(tp=tp, tn=tn, fp=fp, fn=fn)

    p_pred_true = assumed_tpr * assumed_p_gold + assumed_fpr * (1 - assumed_p_gold)
    n_pred_true = round(test_set_size * p_pred_true)
    predictions = [True] * n_pred_true + [False] * (test_set_size - n_pred_true)

    prevalences = observed.estimate_prevalence_from_predictions(
        predictions=predictions,
        n_samples=n_prevalence_samples,
        rng=rng,
    )

    alpha = 1 - confidence
    lower, upper = np.quantile(prevalences, [alpha / 2, 1 - alpha / 2])
    return (upper - lower) / 2


@click.command("annotation_calculator")
@click.option(
    "--target_ci_half_width",
    type=float,
    help="The target 'plus or minus' value for your CI.",
    required=True,
)
@click.option(
    "--assumed_p_gold",
    type=float,
    help="An estimate of how often the gold label is positive in the dataset.",
    required=True,
)
@click.option(
    "--assumed_tpr",
    type=float,
    help="An estimate of the model's true positive rate.",
    required=True,
)
@click.option(
    "--assumed_fpr",
    type=float,
    help="An estimate of the model's false positive rate.",
    required=True,
)
@click.option(
    "--test_set_size",
    type=int,
    help="The size of the dataset that the model will eventually be run on to predict prevalence.",
    required=True,
)
@click.option(
    "--confidence",
    type=float,
    help="The confidence value of your CI.",
    default=0.95,
    show_default=True,
)
@click.option(
    "--min_annotations",
    type=int,
    help="Lower bound for how many annotations to do.",
    default=10,
    show_default=True,
)
@click.option(
    "--max_annotations",
    type=int,
    help="Upper bound for how many annotations to do.",
    default=1000,
    show_default=True,
)
@click.option(
    "--n_prevalence_samples",
    type=int,
    help="Number of sampling iterations per sampled prevalence values.",
    default=1000,
    show_default=True,
)
@click.option(
    "--seed",
    type=int,
    help="Seed for RNG.",
    default=0,
    show_default=True,
)
def annotation_calculator(
    target_ci_half_width: float,
    assumed_p_gold: float,
    assumed_tpr: float,
    assumed_fpr: float,
    test_set_size: int,
    confidence: float,
    min_annotations: int,
    max_annotations: int,
    n_prevalence_samples: int,
    seed: int,
) -> int:
    """
    'How many annotations should I do' calculator for models that will be
    used to estimate prevalence of a phenomenon in a larger dataset.
    """
    if not 0 < target_ci_half_width < 1:
        raise ValueError("target_ci_half_width must be in (0, 1)")
    if not 0 < assumed_p_gold < 1:
        raise ValueError("assumed_p_gold must be in (0, 1)")
    if not 0 < assumed_tpr < 1:
        raise ValueError("assumed_tpr must be in (0, 1)")
    if not 0 < assumed_fpr < 1:
        raise ValueError("assumed_fpr must be in (0, 1)")
    if min_annotations >= max_annotations:
        raise ValueError("min_eval_size must be less than max_eval_size")

    def hw(n: int) -> float:
        result = _ci_half_width(
            n_annotations=n,
            assumed_p_gold=assumed_p_gold,
            assumed_tpr=assumed_tpr,
            assumed_fpr=assumed_fpr,
            test_set_size=test_set_size,
            confidence=confidence,
            n_prevalence_samples=n_prevalence_samples,
            rng=np.random.default_rng(seed),
        )
        print(f"{n} annotations gives {confidence:.2%} CI of ±{result:.4%}")
        return result

    if hw(max_annotations) > target_ci_half_width:
        raise ValueError(
            f"target half-width {target_ci_half_width} not achievable within "
            f"max_eval_size={max_annotations} (got {hw(max_annotations):.4f}). "
            "Increase max_eval_size or relax the target."
        )

    if hw(min_annotations) <= target_ci_half_width:
        return min_annotations

    lo, hi = min_annotations, max_annotations
    while lo < hi:
        mid = (lo + hi) // 2
        if hw(mid) <= target_ci_half_width:
            hi = mid
        else:
            lo = mid + 1
    return lo
