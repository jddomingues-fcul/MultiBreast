from typing import Literal, get_args

from components.plotting.plotting import plot_confidence_scores

_SCOPES = Literal["modality", "birads", "findings"]


class ConfidenceEvaluator:
    """ConfidenceEvaluator is a class that collects and evaluates confidence scores for different scopes in a model's predictions."""

    def __init__(self) -> None:
        self.modality_trues_confidence = []
        self.modality_falses_confidence = []

        self.birads_trues_confidence = []
        self.birads_falses_confidence = []

        self.findings_trues_confidence = []
        self.findings_falses_confidence = []

    def add_confidence_scores(
        self, confidences: list, predictions: list, gt: list, scope: _SCOPES
    ) -> None:
        """Adds confidence scores to the evaluator for a specific scope.

        Args:
            confidences (list): List of confidence scores for each prediction
            predictions (list): List of predictions made by the model
            gt (list): Ground truth labels for the predictions
            scope (_SCOPES): Scope for which the confidence scores are being added. Options are "modality", "birads", or "findings".
        """
        options = get_args(_SCOPES)
        assert scope in options, (
            f"'{scope}' in add_confidence_scores is not in the scope options: {options}"
        )
        assert len(confidences) == len(predictions) == len(gt), (
            "Length of confidences, predictions, and gt must be the same"
        )

        trues, falses = self.construct_confidence_lists(confidences, predictions, gt)

        if scope == "modality":
            self.modality_trues_confidence.extend(trues)
            self.modality_falses_confidence.extend(falses)
        elif scope == "birads":
            self.birads_trues_confidence.extend(trues)
            self.birads_falses_confidence.extend(falses)
        elif scope == "findings":
            self.findings_trues_confidence.extend(trues)
            self.findings_falses_confidence.extend(falses)

    def plot_confidence_scores(self, title_prefix: str = "") -> None:
        """Plots the confidence scores for each scope.

        Args:
            title_prefix (str, optional): Title prefix for the plots. Defaults to "".
        """
        plot_confidence_scores(
            self.modality_trues_confidence,
            self.modality_falses_confidence,
            f"{title_prefix} - Modality Confidence",
        )
        plot_confidence_scores(
            self.birads_trues_confidence,
            self.birads_falses_confidence,
            f"{title_prefix} - Birads Confidence",
        )
        plot_confidence_scores(
            self.findings_trues_confidence,
            self.findings_falses_confidence,
            f"{title_prefix} - Findings Confidence (Separated By Birads)",
        )

    @staticmethod
    def construct_confidence_lists(
        confidence_scores: list, predictions: list, gt: list
    ) -> tuple:
        """Construct two lists of confidence scores: one for true predictions and one for false predictions.

        Args:
            confidence_scores (list): List of confidence scores for each prediction
            predictions (list): List of predictions made by the model
            gt (list): Ground truth labels for the predictions

        Returns:
            tuple: Two lists, the first containing confidence scores for true predictions and the second for false predictions.
        """
        true_scores = []
        false_scores = []

        for i in range(len(predictions)):
            if predictions[i] == gt[i]:
                true_scores.append(confidence_scores[i])
            else:
                false_scores.append(confidence_scores[i])

        return true_scores, false_scores
