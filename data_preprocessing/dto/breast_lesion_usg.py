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
from data_preprocessing.image_processor import ImageProcessor
from data_preprocessing.medical_mappings import UNKNOWN, birads_assessment, get_value
from data_preprocessing.preprocessing_configs import BreastLesionUSGConfigs
from data_preprocessing.utils import csv_column_cleaning


class BreastLesionUSG(BreastCancerDataset):
    def __init__(self, configs: BreastLesionUSGConfigs):
        super().__init__(csv_save_path=configs.csv_save_path)
        self.lesions_usg_df = pd.read_excel(configs.lesions_usg_path, sheet_name=0)
        self.lesions_usg_df.columns = csv_column_cleaning(
            list(self.lesions_usg_df.columns)
        )
        self.configs = configs
        self.image_processor = ImageProcessor(
            raw_imgs_path=configs.raw_imgs_path,
            processed_imgs_path=configs.processed_imgs_path,
            image_preprocessing_config=configs.img_preprocessing_config,
        )

    def process_info(self):
        # Drop columns that are not needed
        self.lesions_usg_df = self.lesions_usg_df.drop(
            columns=["case id", "mask other filename", "pixel size"]
        )

        # Adjust the & tokens by replacing them with "and"
        ampersand_cols = [
            "interpretation",
            "diagnosis",
            "tissue composition",
            "signs",
            "symptoms",
            "margin",
        ]
        self.lesions_usg_df[ampersand_cols] = self.lesions_usg_df[ampersand_cols].map(
            lambda x: self.clean_ampersands(x)
        )

        # replace not applicable and not available with None
        self.lesions_usg_df = self.lesions_usg_df.replace(
            ["not applicable", "not available"], None
        )
        self.lesions_usg_df = self.lesions_usg_df.replace(np.nan, None)

        # If diagnosis is not available, use the interpretation as pointed in the paper
        self.lesions_usg_df["diagnosis"] = self.lesions_usg_df.apply(
            lambda row: (
                row["interpretation"] if row["diagnosis"] is None else row["diagnosis"]
            ),
            axis=1,
        )

        # Adjust the birads values
        self.lesions_usg_df["birads"] = self.lesions_usg_df["birads"].apply(
            lambda x: get_value(x, birads_assessment)
        )

        # Add the modality
        self.lesions_usg_df["modality"] = self.configs.image_modality

        # column renaming
        self.lesions_usg_df = self.lesions_usg_df.rename(
            columns={
                "signs": "breast signs",
                "calcifications": "calcifications characterization",
                "tissue composition": "breast tissue composition",
                "shape": "mass shape",
                "margin": "mass margin",
            }
        )

        # Process the exams
        n = cpu_count() - 1
        df_split = [
            self.lesions_usg_df.iloc[idx]
            for idx in np.array_split(np.arange(len(self.lesions_usg_df)), n)
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
            "breast signs",
            "calcifications characterization",
            "breast tissue composition",
            "mass shape",
            "mass margin",
            "echogenicity",
            "posterior features",
            "halo",
            "skin thickening",
            "classification",
            "interpretation",
            "diagnosis",
        ]

    @staticmethod
    def get_all_report_cols():
        return BreastLesionUSG.get_current_report_cols() + [
            "age",
            "symptoms",
            "verification",
        ]

    def process_small_batch(self, df):
        curr_exams = []

        # Im sorry
        with tqdm(
            total=len(df),
            desc=f"Processing breast lesions usg {df.index[0]} to {df.index[-1]}",
            unit="exam",
            ncols=100,
            position=0,
            leave=True,
        ) as pbar:
            for _, row in df.iterrows():
                exam = self.process_row(row)
                curr_exams.append(exam)
                pbar.update(1)
        return curr_exams

    def process_row(self, row):
        # get needed info for exam
        exam_id = str(uuid.uuid4())
        patient_id = f"{self.get_dataset_name()}-{exam_id}"

        exam_imgs_path = self.image_processor.process_and_save_image(
            os.path.join(self.configs.raw_imgs_path, row["image filename"]),
            exam_id,
            patient_id,
        )

        if pd.isna(row["mask tumor filename"]):
            segmentation_path = None
        else:
            segmentation_path = self.image_processor.save_segmentation_mask(
                os.path.join(self.configs.raw_imgs_path, row["mask tumor filename"]),
                exam_id,
                patient_id,
            )

        return ExamInformation(
            id=exam_id,
            patient=patient_id,
            dataset=self.get_dataset_name(),
            modality=self.configs.image_modality,
            birads=row["birads"],
            race=UNKNOWN,
            laterality=None,
            exam_date=None,
            machine=self.configs.machine,
            exam_type=self.configs.exam_type,
            exam_imgs=exam_imgs_path,
            num_exam_imgs=1 if exam_imgs_path is not None else 0,
            segmentations_path=segmentation_path,
            num_segmentations=1 if segmentation_path is not None else 0,
            slices_imgs_path=None,
            num_slices_imgs=0,
            slices_index=None,  # There are no slices, only the one image, so this is None
            current_report=self.create_report(row, self.get_current_report_cols()),
            full_report=self.create_report(row, self.get_all_report_cols()),
            previous_exams=None,
        )

    @staticmethod
    def clean_ampersands(text):
        return ", ".join(text.split("&"))
