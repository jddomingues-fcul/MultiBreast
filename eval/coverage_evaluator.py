from components.plotting.plotting import plot_findings_scores


class CoverageEvaluator:
    """CoverageEvaluator is responsible for evaluating the coverage of findings in predicted reports against ground truth reports."""

    def __init__(self, model_name: str = "Model #1") -> None:
        self.findings_coverage = []
        self.findings_effective_equality = []
        self.model_name = model_name

    def add_findings(self, predicted_reports: list, gt_reports: list) -> None:
        """Evaluates the coverage of findings in predicted reports against ground truth reports.

        Args:
            predicted_reports (list): List of predicted reports, where each report is a string containing findings.
            gt_reports (list): List of ground truth reports, where each report is a string containing findings.
        """
        coverage, effective_equality = self.construct_findings_coverage(
            predicted_reports, gt_reports
        )
        self.findings_coverage.extend(coverage)
        self.findings_effective_equality.extend(effective_equality)

    def compute_metrics(self, with_plot: bool = False) -> dict:
        coverage_avg = (
            sum(self.findings_coverage) / len(self.findings_coverage)
            if len(self.findings_coverage) > 0
            else 0
        )  # avoid division by zero if there are no coverages
        effective_eqs_avg = (
            sum(self.findings_effective_equality)
            / len(self.findings_effective_equality)
            if len(self.findings_effective_equality) > 0
            else 0
        )  # avoid division by zero if there are no effective_eqs

        if with_plot:
            plot_findings_scores(
                self.findings_coverage,
                self.findings_effective_equality,
                f"{self.model_name} Findings Scores",
            )

        return {
            "findings_coverage": coverage_avg,
            "findings_effective_equality": effective_eqs_avg,
        }

    @staticmethod
    def extract_key_value_from_str(additional_info: str) -> tuple[str, str]:
        """Extracts key and value from a string formatted as 'key: value'.

        Args:
            additional_info (str): String containing the key and value separated by a colon.

        Raises:
            ValueError: If the string is empty or does not contain a colon, it raises a ValueError.

        Returns:
            tuple[str, str]: A tuple containing the key and value, both stripped of leading and trailing whitespace.
        """
        if not additional_info:
            # skip empty strings
            raise ValueError("Empty string provided")

        index_of_col = additional_info.find(":")
        if index_of_col == -1:
            # skip if there is no column separator
            raise ValueError("No column separator found")

        key, value = additional_info[:index_of_col], additional_info[index_of_col + 1 :]
        return key.strip().lower(), value.strip().lower()

    @staticmethod
    def construct_findings_coverage(predicted_reports: list, gt_reports: list) -> tuple:
        """Constructs coverage and effective equality scores for findings in predicted reports against ground truth reports.

        Args:
            predicted_reports (list): List of predicted reports, where each report is a string containing findings.
            gt_reports (list): Description of the ground truth reports.

        Returns:
            tuple: A tuple containing two lists:
                - coverage: List of coverage scores for each report.
                - effective_equality: List of effective equality scores for each report.
        """
        coverage = []
        effective_equality = []

        for i, pred in enumerate(predicted_reports):
            # create a dict for the current gt
            current_gt = {}
            for gt in gt_reports[i].split(";"):
                try:
                    key, value = CoverageEvaluator.extract_key_value_from_str(gt)
                    current_gt[key] = value
                except ValueError:
                    # skip empty strings or if there is no column separator
                    continue

            # create a dict for the current pred
            current_pred = {}
            for pred in pred.split(";"):
                try:
                    key, value = CoverageEvaluator.extract_key_value_from_str(pred)
                    current_pred[key] = value
                except ValueError:
                    # skip empty strings or if there is no column separator
                    continue

            # compute coverage
            common_elements = set(current_gt).intersection(set(current_pred))
            size_of_intersection = len(common_elements)
            current_coverage = (
                size_of_intersection / len(current_gt)
                if size_of_intersection > 0
                else 0
            )
            coverage.append(
                current_coverage
            )  # coverage = size of intersection / size of gt

            # for the effective equality we go through the common elements of predicted and gt and check if the values of the common descriptors are the same
            sum_of_effective_equality = sum(
                [1 for ce in common_elements if current_gt[ce] == current_pred[ce]]
            )
            current_eq = (
                sum_of_effective_equality / len(common_elements)
                if sum_of_effective_equality > 0
                else 0
            )
            effective_equality.append(
                current_eq
            )  # effective equality = size of intersection with same values / size of intersection

        return coverage, effective_equality
