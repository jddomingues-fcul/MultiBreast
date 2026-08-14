import logging
from dataclasses import dataclass, field
from typing import Any

import numpy
from scipy.stats import wilcoxon
from torch import nn
from torchvision import transforms
from tqdm import tqdm

from components.plotting.plotting import plot_basic_performance_table
from data_preprocessing.medical_mappings import (
    birads_assessment_reverse,
    modality_mapping,
)
from eval.coverage_evaluator import CoverageEvaluator
from eval.report_generation_metrics import (
    compute_cls_metrics,
    compute_rg_metrics,
    compute_single_report_metrics,
)


@dataclass
class MetricsHolder:
    modality_f1: float = 0.0
    birads_f1: float = 0.0
    findings_bleu4: float = 0.0
    findings_meteor: float = 0.0
    findings_coverage: float = 0.0
    findings_coverage_equality: float = 0.0

    def compute_diff_with(self, other: "MetricsHolder") -> "MetricsHolder":
        return MetricsHolder(
            modality_f1=self.modality_f1 - other.modality_f1,
            birads_f1=self.birads_f1 - other.birads_f1,
            findings_bleu4=self.findings_bleu4 - other.findings_bleu4,
            findings_meteor=self.findings_meteor - other.findings_meteor,
            findings_coverage=self.findings_coverage - other.findings_coverage,
            findings_coverage_equality=self.findings_coverage_equality
            - other.findings_coverage_equality,
        )


@dataclass
class MetricsListHolder:
    modality_f1: list = field(default_factory=list)
    birads_f1: list = field(default_factory=list)
    findings_bleu4: list = field(default_factory=list)
    findings_meteor: list = field(default_factory=list)
    findings_coverage: list = field(default_factory=list)
    findings_coverage_equality: list = field(default_factory=list)

    def extend_with_metrics(self, diffs: MetricsHolder):
        self.modality_f1.append(diffs.modality_f1)
        self.birads_f1.append(diffs.birads_f1)
        self.findings_bleu4.append(diffs.findings_bleu4)
        self.findings_meteor.append(diffs.findings_meteor)
        self.findings_coverage.append(diffs.findings_coverage)
        self.findings_coverage_equality.append(diffs.findings_coverage_equality)

    def extend_with_average(self, holder: "MetricsListHolder"):
        self.modality_f1.append(sum(holder.modality_f1) / len(holder.modality_f1))
        self.birads_f1.append(sum(holder.birads_f1) / len(holder.birads_f1))
        self.findings_bleu4.append(
            sum(holder.findings_bleu4) / len(holder.findings_bleu4)
        )
        self.findings_meteor.append(
            sum(holder.findings_meteor) / len(holder.findings_meteor)
        )
        self.findings_coverage.append(
            sum(holder.findings_coverage) / len(holder.findings_coverage)
        )
        self.findings_coverage_equality.append(
            sum(holder.findings_coverage_equality)
            / len(holder.findings_coverage_equality)
        )

    @staticmethod
    def compute_95_ci(metrics: "MetricsListHolder") -> dict:
        percentiles = [2.5, 97.5]
        return {
            "modality_f1": [
                numpy.percentile(sorted(metrics.modality_f1), percentiles[0]),
                numpy.mean(metrics.modality_f1),
                numpy.percentile(sorted(metrics.modality_f1), percentiles[1]),
            ],
            "birads_f1": [
                numpy.percentile(sorted(metrics.birads_f1), percentiles[0]),
                numpy.mean(metrics.birads_f1),
                numpy.percentile(sorted(metrics.birads_f1), percentiles[1]),
            ],
            "findings_bleu4": [
                numpy.percentile(sorted(metrics.findings_bleu4), percentiles[0]),
                numpy.mean(metrics.findings_bleu4),
                numpy.percentile(sorted(metrics.findings_bleu4), percentiles[1]),
            ],
            "findings_meteor": [
                numpy.percentile(sorted(metrics.findings_meteor), percentiles[0]),
                numpy.mean(metrics.findings_meteor),
                numpy.percentile(sorted(metrics.findings_meteor), percentiles[1]),
            ],
            "findings_coverage": [
                numpy.percentile(sorted(metrics.findings_coverage), percentiles[0]),
                numpy.mean(metrics.findings_coverage),
                numpy.percentile(sorted(metrics.findings_coverage), percentiles[1]),
            ],
            "findings_coverage_equality": [
                numpy.percentile(
                    sorted(metrics.findings_coverage_equality), percentiles[0]
                ),
                numpy.mean(metrics.findings_coverage_equality),
                numpy.percentile(
                    sorted(metrics.findings_coverage_equality), percentiles[1]
                ),
            ],
        }

    @staticmethod
    def compute_99_ci(metrics: "MetricsListHolder") -> dict:
        percentiles = [0.5, 99.5]
        return {
            "modality_f1": [
                numpy.percentile(sorted(metrics.modality_f1), percentiles[0]),
                numpy.mean(metrics.modality_f1),
                numpy.percentile(sorted(metrics.modality_f1), percentiles[1]),
            ],
            "birads_f1": [
                numpy.percentile(sorted(metrics.birads_f1), percentiles[0]),
                numpy.mean(metrics.birads_f1),
                numpy.percentile(sorted(metrics.birads_f1), percentiles[1]),
            ],
            "findings_bleu4": [
                numpy.percentile(sorted(metrics.findings_bleu4), percentiles[0]),
                numpy.mean(metrics.findings_bleu4),
                numpy.percentile(sorted(metrics.findings_bleu4), percentiles[1]),
            ],
            "findings_meteor": [
                numpy.percentile(sorted(metrics.findings_meteor), percentiles[0]),
                numpy.mean(metrics.findings_meteor),
                numpy.percentile(sorted(metrics.findings_meteor), percentiles[1]),
            ],
            "findings_coverage": [
                numpy.percentile(sorted(metrics.findings_coverage), percentiles[0]),
                numpy.mean(metrics.findings_coverage),
                numpy.percentile(sorted(metrics.findings_coverage), percentiles[1]),
            ],
            "findings_coverage_equality": [
                numpy.percentile(
                    sorted(metrics.findings_coverage_equality), percentiles[0]
                ),
                numpy.mean(metrics.findings_coverage_equality),
                numpy.percentile(
                    sorted(metrics.findings_coverage_equality), percentiles[1]
                ),
            ],
        }

    @staticmethod
    def compute_p_value(metrics: "MetricsListHolder") -> dict:
        return {
            "modality_f1": metrics.p_val(metrics.modality_f1),
            "birads_f1": metrics.p_val(metrics.birads_f1),
            "findings_bleu4": metrics.p_val(metrics.findings_bleu4),
            "findings_meteor": metrics.p_val(metrics.findings_meteor),
            "findings_coverage": metrics.p_val(metrics.findings_coverage),
            "findings_coverage_equality": metrics.p_val(
                metrics.findings_coverage_equality
            ),
        }

    @staticmethod
    def p_val(deltas: list[float]) -> float:
        # Compute deltas counts
        count_greater_zero = sum(1 for delta in deltas if delta > 0)
        count_smaller_zero = sum(1 for delta in deltas if delta < 0)

        # Rationalize
        count_greater_zero /= len(deltas)
        count_smaller_zero /= len(deltas)

        return 2 * min(count_greater_zero, count_smaller_zero)  # Two-tailed p-value


@dataclass(frozen=True)
class SeedModel:
    run_str: str
    model: nn.Module


class SeedModelResults:
    def __init__(self):
        self.target_modality: list = []
        self.target_birads: list = []
        self.target_findings: list = []

        self.pred_modality: list = []
        self.pred_birads: list = []
        self.pred_findings: list = []

        self.patient_ids: list = []
        self.pred_findings_lexical_metrics: list = []

    def extend_with_patient_ids(self, patient_ids: list):
        self.patient_ids.extend(patient_ids)

    def extend_with_targets(self, modality_true, birads_true, findings_true):
        self.target_modality.extend(modality_true)
        self.target_birads.extend(birads_true)
        self.target_findings.extend(findings_true)

    def extend_with_predictions(self, modality_pred, birads_pred, findings_pred):
        self.pred_modality.extend(modality_pred)
        self.pred_birads.extend(birads_pred)
        self.pred_findings.extend(findings_pred)

    def compute_lexical_metrics(self):
        """
        Compute lexical metrics for the current predictions and targets. This will make evals run much faster since they will only be computed once
        """
        self.pred_findings_lexical_metrics = [
            compute_single_report_metrics((pred, tgt))
            for pred, tgt in zip(self.pred_findings, self.target_findings)
        ]


def setup_seed_models(
    runs: list[str], device: str, eval_mode: bool, model_cls
) -> list[SeedModel]:
    seed_models = []
    for run_str in runs:
        # model_path = download_model_checkpoint(run_str)
        model = model_cls.from_pretrained(
            chkpt_path=run_str, device=device, eval_mode=eval_mode
        )
        seed_models.append(SeedModel(run_str=run_str, model=model))
    return seed_models


def predict_with_test_loader(
    loader_iterator, model: SeedModel, trainer_iters: int, norm_mean, norm_std, device
) -> MetricsHolder:
    full_modalities_pred = []
    full_birads_pred = []
    full_findings_pred = []

    full_modalties_true = []
    full_birads_true = []
    full_findings_true = []

    for _ in tqdm(range(trainer_iters), desc="Evaluating models", total=trainer_iters):
        try:
            imgs, report, birads, modalities, _ = next(loader_iterator)
        except StopIteration:
            logging.warning(
                "DataLoader iterator exhausted before reaching the specified number of iterations."
            )
            break

        imgs = imgs.to(device)
        if imgs.dim() == 5:
            imgs = imgs.squeeze(1)  # Remove the singleton dimension if present
        birads_np = birads.detach().numpy().tolist()
        modalities_mapped = [modality_mapping[modality] for modality in modalities]
        report = [
            rep.lower() for rep in report
        ]  # convert to lower case to ensure non case sensitive comparison

        baseline_predictions = model.tokenizer.predict_with_model(
            input_images=imgs,
            model=model.model,
            pre_transform=transforms.Normalize(mean=norm_mean, std=norm_std),
            plot_attention=False,
            short_report=False,
        )

        (
            _,
            predictions_findings,
            predictions_birads,
            prediction_modality,
            _,
        ) = get_content_from_predictions(baseline_predictions)

        full_modalities_pred.extend(prediction_modality)
        full_birads_pred.extend(predictions_birads)
        full_findings_pred.extend(predictions_findings)

        full_modalties_true.extend(modalities_mapped)
        full_birads_true.extend(birads_np)
        full_findings_true.extend(report)

    modalities_scores = compute_cls_metrics(
        pred=full_modalities_pred, target=full_modalties_true
    )
    birads_scores = compute_cls_metrics(pred=full_birads_pred, target=full_birads_true)
    findings_scores = compute_rg_metrics(
        predicted_reports=full_findings_pred, target_reports=full_findings_true
    )
    findings_coverage, findings_coverage_equality = (
        CoverageEvaluator.construct_findings_coverage(
            predicted_reports=full_findings_pred, gt_reports=full_findings_true
        )
    )

    return MetricsHolder(
        modality_f1=modalities_scores["f1 score"],
        birads_f1=birads_scores["f1 score"],
        findings_bleu4=findings_scores["bleu4"],
        findings_meteor=findings_scores["meteor"],
        findings_coverage=sum(findings_coverage) / len(findings_coverage)
        if findings_coverage
        else 0.0,
        findings_coverage_equality=sum(findings_coverage_equality)
        / len(findings_coverage_equality)
        if findings_coverage_equality
        else 0.0,
    )


def optional_strip(value):
    return value.strip() if value is not None else ""


def try_get_map(value, mapping):
    return mapping[value] if value in mapping else -1


def get_content_from_predictions(predictions):
    predictions_pretty_report = [
        optional_strip(curr_set["pretty_report"]) for curr_set in predictions
    ]
    predictions_findings = [
        optional_strip(curr_set["findings"]) for curr_set in predictions
    ]
    predictions_birads = [
        try_get_map(optional_strip(curr_set["birads"]), birads_assessment_reverse)
        for curr_set in predictions
    ]
    prediction_modality = [
        try_get_map(optional_strip(curr_set["modality"]), modality_mapping)
        for curr_set in predictions
    ]
    rg_ids = [curr_set["report_id"] for curr_set in predictions]

    return (
        predictions_pretty_report,
        predictions_findings,
        predictions_birads,
        prediction_modality,
        rg_ids,
    )


def compute_significance_tests(
    baseline_scores, challenger_scores, title_prefix: str = "Models Comparison"
):
    results = {
        "wilcoxon statistic": -1.0,
        "wilcoxon pvalue": -1.0,
    }

    try:
        wilcoxon_stat, wilcoxon_pvalue = try_wilcoxon(
            baseline_scores, challenger_scores
        )

        results["wilcoxon statistic"] = wilcoxon_stat
        results["wilcoxon pvalue"] = wilcoxon_pvalue

    except Exception as e:
        logging.error(f"Error computing significance tests: {e}")

    plot_basic_performance_table(results, f"{title_prefix} - Significante Tests")


def try_wilcoxon(
    x: Any, y: Any | None = None, alternative: str = "two-sided"
) -> tuple[float, float]:
    """
    H0: Two related paired samples come from the same distribution. Tests if the distributions difference is symmetric about zero
    H1: Two related paired samples come from different distributions.

    Args:
        x: Samples from model A
        y: Samples from model B
        alternative: The alternative hypothesis can be either two-sided, greater or less

    Returns: The test statistic and the p-value

    """
    try:
        if y is None:
            res = wilcoxon(x, alternative=alternative)
        else:
            res = wilcoxon(x, y, alternative=alternative)
        return res.statistic, res.pvalue
    except Exception as e:
        logging.error(f"Error in Wilcoxon test: {e}")
        return -1.0, -1.0  # Default values in case of error
