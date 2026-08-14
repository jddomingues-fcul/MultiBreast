import logging
import warnings
from multiprocessing import Pool, cpu_count

import evaluate
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from eval.coverage_evaluator import CoverageEvaluator

# Scorers
bleu_scorer = evaluate.load("bleu")
meteor_scorer = evaluate.load("meteor")


def get_func_from_name(func_name: str):
    """Retrieve the evaluation function based on its name.
    Args:
        func_name (str): Name of the evaluation function.
    Raises:
        ValueError: If the function name is not recognized.
    """
    funcs = {
        "cls": compute_cls_metrics,
        "rg": compute_rg_metrics,
        "paired_cls": compute_paired_cls,
        "paired_rg": compute_paired_rg,
    }

    if func_name in funcs:
        return funcs[func_name]
    else:
        raise ValueError(
            f"Function '{func_name}' not found. Available functions: {list(funcs.keys())}"
        )


def compute_paired_cls(preds: list, targets: list):
    return [1 if p == t else 0 for p, t in zip(preds, targets)]


def compute_paired_rg(preds: list, targets: list):
    result = []
    for p, t in zip(preds, targets):
        if len(p) == 0 or len(t) == 0:
            result.append(0)
            continue
        res = bleu_scorer.compute(
            predictions=[p], references=[t], max_order=4
        )  # fixed max_order to 4 for consistency
        if res is not None:
            result.append(res["bleu"])
        else:
            result.append(0)
    return result


def compute_cls_metrics(pred, target, features_approach="macro", sample_weight=None):
    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore"
        )  # ignore warnings from sklearn involved with y_pred not being in y_true

        # log the predictions that do not appear in target
        if len(set(pred) - set(target)) > 0:
            logging.warning(f"preds not in target {[list(set(pred) - set(target))]}")

        acc = balanced_accuracy_score(target, pred, sample_weight=sample_weight)
        precision = precision_score(
            target,
            pred,
            average=features_approach,
            zero_division=0,
            sample_weight=sample_weight,
        )
        recall = recall_score(
            target,
            pred,
            average=features_approach,
            zero_division=0,
            sample_weight=sample_weight,
        )
        f1 = f1_score(
            target,
            pred,
            average=features_approach,
            zero_division=0,
            sample_weight=sample_weight,
        )

        return {
            "balanced accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1 score": f1,
        }


def compute_single_report_metrics(pred_target: tuple) -> dict:
    predicted_report, target_report = pred_target

    try:
        cov, cov_eq = CoverageEvaluator.construct_findings_coverage(
            [predicted_report], [target_report]
        )

        metrics = {
            "bleu4": bleu_scorer.compute(
                predictions=[predicted_report], references=[target_report], max_order=4
            )["bleu"],
            "meteor": meteor_scorer.compute(
                predictions=[predicted_report], references=[target_report]
            )["meteor"],
            "findings_coverage": cov[0],
            "findings_effective_equality": cov_eq[0],
        }
    except ZeroDivisionError as e:
        logging.error(f"Error in computing metrics: {e}")
        metrics = {}
    return metrics


def compute_rg_metrics(predicted_reports: list, target_reports: list):
    n_samples = len(predicted_reports)
    metrics_sum = {
        "bleu4": 0,
        "meteor": 0,
        "findings_coverage": 0,
        "findings_effective_equality": 0,
    }

    if n_samples == 0:
        logging.warning("No samples to compute metrics.")
        return metrics_sum

    n = cpu_count() - 1
    reports = list(zip(predicted_reports, target_reports))
    with Pool(processes=n) as p:
        results = p.map(compute_single_report_metrics, reports)

    for result in results:
        for k, v in result.items():
            metrics_sum[k] += v

    # Average over all samples.
    averaged_metrics = {k: v / n_samples for k, v in metrics_sum.items()}
    return averaged_metrics
