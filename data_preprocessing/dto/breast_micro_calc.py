import os
import uuid
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from tqdm import tqdm

from data_preprocessing.breast_cancer_dataset import (
    BreastCancerDataset,
    ExamInformation,
)
from data_preprocessing.error_handling import trycatch_func
from data_preprocessing.image_processor import ImageProcessor
from data_preprocessing.medical_mappings import (
    UNKNOWN,
    birads_assessment,
    breast_density,
    get_value,
    laterality,
)
from data_preprocessing.preprocessing_configs import BreastMicroCalcConfig
from data_preprocessing.utils import csv_column_cleaning


class BreastMicroCalc(BreastCancerDataset):
    def __init__(self, configs: BreastMicroCalcConfig):
        super().__init__(csv_save_path=configs.csv_save_path)
        self.configs = configs
        self.image_processor = ImageProcessor(
            raw_imgs_path=configs.raw_imgs_path,
            processed_imgs_path=configs.processed_imgs_path,
            image_preprocessing_config=configs.img_preprocessing_config,
        )

        self.normal_cases = pd.read_excel(
            configs.description_path, sheet_name="Normal_cases_modified"
        )
        self.suspicious_cases = pd.read_excel(
            configs.description_path, sheet_name="Suspicious_cases_modified"
        )

    def process_info(self):

        # Adjust normal cases
        self.normal_cases.columns = csv_column_cleaning(list(self.normal_cases.columns))
        self.normal_cases = self.normal_cases.rename(
            columns={
                "bi-rads categories for breast density": "breast density",
                "bi-rads categories for classification": "birads",
                "available mammograms": "mammograms time interval",
                "breast right/left ": "laterality",
                "age at the time of the recent mammogram ": "age",
            }
        )
        self.normal_cases["subfolder"] = "Normal_cases"

        # Adjust suspicious cases
        self.suspicious_cases.columns = csv_column_cleaning(
            list(self.suspicious_cases.columns)
        )
        self.suspicious_cases = self.suspicious_cases.rename(
            columns={
                "bi-rads categories for breast density": "breast density",
                "bi-rads categories for classification": "birads",
                "available mammograms": "mammograms time interval",
                "breast right/left": "laterality",
                "age at the time of the recent mammogram": "age",
            }
        )
        self.suspicious_cases["subfolder"] = "Suspicious_cases"

        # Combine normal and suspicious cases
        breast_micro_calc = pd.concat(
            [self.normal_cases, self.suspicious_cases], axis=0, ignore_index=True
        )
        breast_micro_calc = breast_micro_calc.replace(np.nan, None)
        breast_micro_calc = breast_micro_calc.replace("NOT AVAILABLE", None)

        # Replace the values accordingly
        breast_micro_calc["birads"] = breast_micro_calc["birads"].apply(
            lambda x: get_value(x, birads_assessment)
        )
        breast_micro_calc["laterality"] = breast_micro_calc["laterality"].apply(
            lambda x: get_value(x, laterality)
        )
        breast_micro_calc["breast density"] = breast_micro_calc["breast density"].apply(
            lambda x: get_value(x, breast_density)
        )

        # Add the modality
        breast_micro_calc["modality"] = self.configs.image_modality

        # Process the exams
        n = cpu_count() - 1
        df_split = [
            breast_micro_calc.iloc[idx]
            for idx in np.array_split(np.arange(len(breast_micro_calc)), n)
            if len(idx) > 0
        ]
        with Pool(processes=n) as p:
            results = p.map(self.process_small_batch, df_split)

        for result in results:
            for exam in result:
                self.append_exam(exam)

    @staticmethod
    def get_current_report_cols():
        return [
            "breast density",
        ]

    @staticmethod
    def get_full_report_cols():
        return BreastMicroCalc.get_current_report_cols() + [
            "laterality",
            "age",
            "mammograms time interval",
            "years between screenings",
            "biopsy results",
        ]

    @trycatch_func
    def process_small_batch(self, df):
        curr_exams = []

        # Im sorry
        with tqdm(
            total=len(df),
            desc=f"Processing cmmd batch {df.index[0]} to {df.index[-1]}",
            unit="exam",
            ncols=100,
            position=0,
            leave=True,
        ) as pbar:
            for _, row in df.iterrows():
                exams = self.process_row(row)
                curr_exams.extend(exams)
                pbar.update(1)
        return curr_exams

    @trycatch_func
    def process_row(self, row):
        cc_prior_id, cc_prior_img = (
            str(uuid.uuid4()),
            os.path.join(
                self.configs.raw_imgs_path,
                row["subfolder"],
                str(row["folder #"]),
                f"CC_prior{self.configs.raw_imgs_extension}",
            ),
        )
        mlo_prior_id, mlo_prior_img = (
            str(uuid.uuid4()),
            os.path.join(
                self.configs.raw_imgs_path,
                row["subfolder"],
                str(row["folder #"]),
                f"MLO_prior{self.configs.raw_imgs_extension}",
            ),
        )
        cc_recent_id, cc_recent_img = (
            str(uuid.uuid4()),
            os.path.join(
                self.configs.raw_imgs_path,
                row["subfolder"],
                str(row["folder #"]),
                f"CC_recent{self.configs.raw_imgs_extension}",
            ),
        )
        mlo_recent_id, mlo_recent_img = (
            str(uuid.uuid4()),
            os.path.join(
                self.configs.raw_imgs_path,
                row["subfolder"],
                str(row["folder #"]),
                f"MLO_recent{self.configs.raw_imgs_extension}",
            ),
        )

        patient = f"{row['subfolder']}_{row['folder #']!s}"
        exam_dates = [eval(date) for date in row["mammograms time interval"].split("-")]
        prior_date, recent_date = min(exam_dates), max(exam_dates)

        curr_exams = []
        for exam_id, exam_img, date in [
            (cc_prior_id, cc_prior_img, prior_date),
            (mlo_prior_id, mlo_prior_img, prior_date),
            (cc_recent_id, cc_recent_img, recent_date),
            (mlo_recent_id, mlo_recent_img, recent_date),
        ]:
            # Apparanetly the images are not always in lower case. Random images with the extension in upper case for some reason
            if not os.path.exists(exam_img):
                exam_img = exam_img.replace(
                    self.configs.raw_imgs_extension,
                    self.configs.raw_imgs_extension.upper(),
                )

            exam_imgs_path = self.image_processor.process_and_save_image(
                exam_img, exam_id, patient
            )

            exam = ExamInformation(
                id=exam_id,
                patient=f"{self.get_dataset_name()}-{patient}",
                dataset=self.get_dataset_name(),
                modality=self.configs.image_modality,
                birads=row["birads"] if date == recent_date else None,
                race=UNKNOWN,
                laterality=row["laterality"],
                exam_date=str(date),
                machine=None,
                exam_type=(
                    "cranial caudal" if "CC" in exam_img else "mediolateral oblique"
                ),
                exam_imgs=exam_imgs_path,
                num_exam_imgs=1 if exam_imgs_path is not None else 0,
                segmentations_path=None,
                num_segmentations=0,
                slices_imgs_path=None,
                num_slices_imgs=0,
                slices_index=None,
                current_report=(
                    self.create_report(row, self.get_current_report_cols())
                    if date == recent_date
                    else None
                ),
                full_report=(
                    self.create_report(row, self.get_full_report_cols())
                    if date == recent_date
                    else None
                ),
                previous_exams=(
                    (cc_prior_id, mlo_prior_id) if date == recent_date else None
                ),
            )

            curr_exams.append(exam)

        return curr_exams
