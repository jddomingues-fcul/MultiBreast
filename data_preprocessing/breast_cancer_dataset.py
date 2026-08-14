import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import pandas as pd
from yaml import safe_load

from data_preprocessing.error_handling import trycatch_func
from data_preprocessing.utils import sanitize_string


@dataclass(frozen=True)
class ExamInformation: 
    # demographics and exam basic info
    id: str
    patient: str
    dataset: str
    modality: str
    birads: str | None
    race: str | None
    laterality: str | None  # just relevant at treatment because the left breast near the hear requires a more careful approach
    exam_date: str | None
    machine: str | None
    exam_type: str | None  # set to None in cases we do not know the type of exam. assumed as a "regular" modality exam

    # Image related data
    exam_imgs: str | None  # all the exam images in case we are. Singular images
    num_exam_imgs: int  # number of images in the exam

    segmentations_path: str | None  # segmentations for the exam imgs
    num_segmentations: int  # number of segmentations in the exam

    slices_imgs_path: str | None  # in cases where we have slices, then we need to have them so we save them alltogether
    num_slices_imgs: int  # number of slices in the exam

    slices_index: tuple[int, ...] | None  # indexes in the slices imgs that are relevant for the exam and that were kept in exam_imgs

    # Report related data
    current_report: str | None
    full_report: str | None
    previous_exams: tuple[Any, ...] | None  # This is a list of previous exams ids


class BreastCancerDataset(ABC):
    def __init__(self, csv_save_path: str):
        self.save_path = csv_save_path
        self.data = []
        self.dataset_name = None

    @abstractmethod
    def process_info(self):
        pass

    def get_dataset_name(self) -> str:
        assert self.dataset_name is not None, "Dataset name is not set"
        return self.dataset_name

    def set_dataset_name(self, dataset_name: str):
        self.dataset_name = dataset_name

    def append_exam(self, exam: ExamInformation | None):
        if exam is not None:
            self.data.append(exam.__dict__)

    def find_exam(self, exam_id: str) -> ExamInformation | None:
        for exam in self.data:
            if exam["id"] == exam_id:
                return ExamInformation(**exam)
        return None

    def save_csv(self):
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        df = pd.DataFrame(self.data)
        df.to_csv(self.save_path, index=False)

    @trycatch_func
    def create_report(
        self, pandas_row: pd.Series, cols: list, previous_reports: str | None = None
    ) -> str:
        report = ""

        for col in cols:
            if pd.notna(pandas_row[col]):
                stringified_col = sanitize_string(str(col))
                stringified_value = sanitize_string(str(pandas_row[col]))
                report += f"{stringified_col} : {stringified_value} ; "

        if previous_reports:
            report = (
                "previous reports indicate : "
                + previous_reports
                + "\ncurrent reports indicate : "
                + report
            )

        return report

    @staticmethod
    def from_config(dataset_name: str, dataset_class, yaml_config_path: str):
        with open(yaml_config_path, "r") as file:
            yaml_configs = safe_load(file)

        data_class = dataset_class(**yaml_configs[dataset_name])
        data_class.set_dataset_name(dataset_name)
        return data_class

    def get_previous_report(self, buffer: list) -> str | None:
        # Since we are sorting the buffer, and the keys are inserted in order, the last key is either the current one, or the previous
        # we just want the previous one, which should be summarised already due to the update when dates change
        # as such, we just need to check the last key, and if it is the same as the current date, then we need to get the previous one,
        # otherwise we just return the current one

        result = ""
        for exam in buffer:
            curr_exam = self.find_exam(exam)
            if curr_exam:
                exam = curr_exam.full_report
                if exam:
                    result += exam + "\n"

        return result
