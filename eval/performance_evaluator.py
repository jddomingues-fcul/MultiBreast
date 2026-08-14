import math
from typing import Literal, get_args

from components.plotting.plotting import plot_basic_performance_table
from eval.report_generation_metrics import get_func_from_name

_EVAL_FUNCS = Literal["cls", "rg", "paired_cls", "paired_rg"]


class SingleClusterPredictionsHolder:
    def __init__(self, eval_func) -> None:
        self.predictions = []
        self.ground_truth = []
        self.scores = []
        self.eval_func = eval_func

        self.mean_performance: float = (
            0.0  # simple mean over the comparison between predictions and ground truth
        )
        self.standard_error_clt: float = 0.0  # based on the central limit theorem - and eval_func over the prediction results
        self.ci95: float = 0.0

    def extend(self, predictions: list, ground_truth: list):
        """Extends the predictions and ground truth lists with new values.

        Args:
            predictions (list): Predictions made by the model.
            ground_truth (list): Ground truth labels for the predictions.
        """
        self.predictions.extend(predictions)
        self.ground_truth.extend(ground_truth)

    def compute_scores(self):
        """Computes the scores based on the predictions and ground truth using the evaluation function.

        Raises:
            ValueError: If the evaluation function is not set before computing scores.
        """
        if self.eval_func is None:
            raise ValueError(
                "Eval function is not set. Please set it before computing scores."
            )
        self.scores = self.eval_func(self.predictions, self.ground_truth)

    def compute_mean_performance(self):
        """Computes the mean performance based on the scores."""
        self.mean_performance = (
            sum(self.scores) / len(self.scores) if self.scores else 0.0
        )

    def compute_standard_error_clt(self):
        """Computes the standard error based on the central limit theorem (CLT) using the scores."""
        n = len(self.scores)
        if n <= 1:
            self.standard_error_clt = 0.0
            return
        x = sum([(score - self.mean_performance) ** 2 for score in self.scores]) / (
            n - 1
        )
        self.standard_error_clt = math.sqrt(x / n) if n > 0 else 0.0

    def compute_ci95(self):
        self.ci95 = 1.96 * self.standard_error_clt

    def compute_all(self):
        self.compute_scores()
        self.compute_mean_performance()
        self.compute_standard_error_clt()
        self.compute_ci95()

    def to_plot_map(self) -> dict:
        return {
            "# questions": len(self.scores),
            f"mean performance (average of a {self.eval_func.__name__})": self.mean_performance,
            "standard error clt": self.standard_error_clt,
            "95% ci sec": self.ci95,
        }


class MultiClusterPredictionsHolder(SingleClusterPredictionsHolder):
    def __init__(self, eval_func) -> None:
        super().__init__(eval_func)
        self.clusters = []
        self.standard_error_clustered: float = 0.0
        self.ci95_clustered: float = 0.0
        self.ci99_clustered: float = 0.0

    def extend_with_clusters(
        self, predictions: list, ground_truth: list, clusters: list
    ):
        super().extend(predictions, ground_truth)
        self.clusters.extend(clusters)

    def compute_standard_error_clustered(self):
        acc = 0
        n = len(self.scores)

        if n == 0:
            self.standard_error_clustered = 0.0
            return

        for cluster in set(self.clusters):
            clusters_indexes = [i for i, v in enumerate(self.clusters) if v == cluster]
            scores_of_cluster = [self.scores[i] for i in clusters_indexes]
            for score_i in scores_of_cluster:
                for score_j in scores_of_cluster:
                    curr = (
                        (score_i - self.mean_performance)
                        * (score_j - self.mean_performance)
                        / n**2
                    )
                    acc += curr

        self.standard_error_clustered = acc

    def compute_ci95_clustered(self):
        self.ci95_clustered = 1.96 * self.standard_error_clustered

    def compute_ci99_clustered(self):
        self.ci99_clustered = 2.575 * self.standard_error_clustered

    def compute_all(self):
        super().compute_all()
        self.compute_standard_error_clustered()
        self.compute_ci95_clustered()

    def to_plot_map(self) -> dict:
        result = super().to_plot_map()
        result["standard error clustered"] = self.standard_error_clustered
        result["95% ci clustered"] = self.ci95_clustered
        result["99% ci clustered"] = self.ci99_clustered
        return result


class PerformanceEvaluator:
    """Performance Evaluator for evaluating model performance on classification assessment based on predictions, ground truth, birads, and modalities."""

    def __init__(
        self,
        eval_func: _EVAL_FUNCS = "cls",
        model_name: str = "Model #1",
        scope: str = "BI-RADS Acc",
    ) -> None:
        options = get_args(_EVAL_FUNCS)
        assert eval_func in options, f"'{eval_func}' is not in {options}"

        self.model_name = model_name
        self.scope = scope
        self.eval_func = get_func_from_name(eval_func)
        self.predictions_holder_func = get_func_from_name("paired_" + eval_func)

        # For results
        self.global_results = MultiClusterPredictionsHolder(
            eval_func=self.predictions_holder_func
        )
        self.modality_results: dict[str, SingleClusterPredictionsHolder] = {}
        self.birads_results: dict[str, SingleClusterPredictionsHolder] = {}

    def add_predictions(
        self, predictions: list, gt: list, birads: list, modalities: list
    ) -> None:
        """Adds predictions and ground truth to the evaluator, along with birads and modalities for better understanding performance per "cluster".

        Args:
            predictions (list): Predictions made by the model.
            gt (list): Ground truth labels for the predictions.
            birads (list): BI-RADS categories for the predictions.
            modalities (list): Modalities of the images corresponding to the predictions.
        """
        # Saving the predictions and target, but also birads and modalities for better understanding performance per "cluster"
        self.global_results.extend_with_clusters(
            predictions, gt, [f"{b}_{m}" for b, m in zip(birads, modalities)]
        )

        # Saving the results per modality
        for modality in set(modalities):
            if modality not in self.modality_results:
                self.modality_results[modality] = SingleClusterPredictionsHolder(
                    eval_func=self.predictions_holder_func
                )

            self.modality_results[modality].extend(
                [predictions[i] for i, m in enumerate(modalities) if m == modality],
                [gt[i] for i, m in enumerate(modalities) if m == modality],
            )

        # Saving the results per birads
        for curr_birads in set(birads):
            if curr_birads not in self.birads_results:
                self.birads_results[curr_birads] = SingleClusterPredictionsHolder(
                    eval_func=self.predictions_holder_func
                )

            self.birads_results[curr_birads].extend(
                [predictions[i] for i, b in enumerate(birads) if b == curr_birads],
                [gt[i] for i, b in enumerate(birads) if b == curr_birads],
            )

    def compute_metrics(self, with_plot: bool = False) -> dict | None:
        """Computes the metrics for the predictions and ground truth, both globally and per modality and birads.

        Args:
            with_plot (bool, optional): If in addition to the computation we want to plot too. Defaults to False.

        Returns:
            dict | None: A dictionary containing the global results, modality performance, and birads performance. if `with_plot` is True, a plot of each performance will be generated.
        """
        # Compute the metrics globally
        global_results = self.eval_func(
            self.global_results.predictions, self.global_results.ground_truth
        )

        # Compute the metrics per modality
        modality_performance = self._compute_metrics_per_modality()

        # Compute the metrics per birads
        birads_performance = self._compute_metrics_per_birads()

        if with_plot:
            plot_basic_performance_table(
                global_results, f"{self.model_name} {self.scope} Global Performance"
            )

            for m, performance_map in modality_performance.items():
                plot_basic_performance_table(
                    performance_map,
                    f"{self.model_name}  {self.scope}  Performance per modality  {m}",
                )

            for b, performance_map in birads_performance.items():
                plot_basic_performance_table(
                    performance_map,
                    f"{self.model_name}  {self.scope}  Performance per birads  {b}",
                )

        return {
            "global": global_results,
            "modality": modality_performance,
            "birads": birads_performance,
        }

    def _compute_metrics_per_modality(self) -> dict:
        result = {}
        for modality, metrics_holder in self.modality_results.items():
            result[modality] = self.eval_func(
                metrics_holder.predictions, metrics_holder.ground_truth
            )
        return result

    def _compute_metrics_per_birads(self) -> dict:
        result = {}
        for birads, metrics_holder in self.birads_results.items():
            result[birads] = self.eval_func(
                metrics_holder.predictions, metrics_holder.ground_truth
            )
        return result

    def compute_detailed_metrics(self, with_plot: bool = False) -> dict | None:
        """
        Computes detailed metrics for the predictions and ground truth, both globally and per modality and birads, including standard error and confidence intervals.
        Args:
            with_plot (bool, optional): If in addition to the computation we want to plot too. Defaults to False.
        Returns:
            dict | None: A dictionary containing the global results, modality performance, and birads performance with detailed metrics. If `with_plot` is True, a plot of each performance will be generated.
        """
        # Compute the metrics globally
        self.global_results.compute_all()

        # Compute the metrics per modality
        for _, metrics_holder in self.modality_results.items():
            metrics_holder.compute_all()

        # Compute the metrics per birads
        for _, metrics_holder in self.birads_results.items():
            metrics_holder.compute_all()

        if with_plot:
            # Plotting the global performance
            plot_basic_performance_table(
                self.global_results.to_plot_map(),
                f"{self.model_name} {self.scope} Detailed Global Performance",
            )

            # Plotting the performance per modality
            for m, metrics_holder in self.modality_results.items():
                plot_basic_performance_table(
                    metrics_holder.to_plot_map(),
                    f"{self.model_name} {self.scope} Detailed Performance per modality {m}",
                )

            # Plotting the performance per birads
            for b, metrics_holder in self.birads_results.items():
                plot_basic_performance_table(
                    metrics_holder.to_plot_map(),
                    f"{self.model_name} {self.scope} Detailed Performance per birads {b}",
                )

        return {
            "global": self.global_results,
            "modality": self.modality_results,
            "birads": self.birads_results,
        }
