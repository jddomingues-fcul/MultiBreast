import glob
import logging
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
from data_preprocessing.image_utils import (
    convert_to_left_right_comparison,
    is_left_right_breast,
)
from data_preprocessing.medical_mappings import birads_assessment, get_value, laterality
from data_preprocessing.preprocessing_configs import ChineseMGConfig


class CMMD(BreastCancerDataset):
    def __init__(self, configs: ChineseMGConfig):
        super().__init__(csv_save_path=configs.csv_save_path)
        self.configs = configs
        self.image_processor = ImageProcessor(
            raw_imgs_path=configs.raw_imgs_path,
            processed_imgs_path=configs.processed_imgs_path,
            image_preprocessing_config=configs.img_preprocessing_config,
        )

        self.clinical_data = pd.read_excel(configs.clinical_data_path)
        self.clinical_data = self.clinical_data.rename(
            columns={
                "ID1": "patient id",
                "LeftRight": "laterality",
                "Age": "age",
                "classification": "birads",
                "abnormality": "abnormality type",
            }
        )
        self.clinical_data = self.clinical_data.replace(np.nan, None)

    def process_info(self):
        # Replace the values accordingly
        self.clinical_data["laterality"] = self.clinical_data["laterality"].apply(
            lambda x: get_value(x, laterality)
        )
        self.clinical_data["abnormality type"] = self.clinical_data[
            "abnormality type"
        ].apply(lambda x: "mass and calcification" if x == "both" else x)
        self.clinical_data["birads"] = self.clinical_data["birads"].apply(
            lambda x: get_value(x, birads_assessment)
        )

        # Add the modality
        self.clinical_data["modality"] = self.configs.image_modality

        # Process the exams
        n = cpu_count() - 1
        df_split = [
            self.clinical_data.iloc[idx]
            for idx in np.array_split(np.arange(len(self.clinical_data)), n)
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
            "abnormality type",
        ]

    @staticmethod
    def get_full_report_cols():
        return CMMD.get_current_report_cols() + ["age", "subtype", "laterality"]

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
                exam = self.process_row(row)
                curr_exams.append(exam)
                pbar.update(1)
        return curr_exams

    @trycatch_func
    def process_row(self, row):
        exam_id = str(uuid.uuid4())
        exam_imgs_path, number_exams = self.save_imgs(
            row["patient id"],
            exam_id,
            row["patient id"],
            self.configs.raw_imgs_extension,
            row["laterality"],
        )

        return ExamInformation(
            id=exam_id,
            patient=f"{self.get_dataset_name()}-{row['patient id']}",
            dataset=self.get_dataset_name(),
            modality=self.configs.image_modality,
            birads=row["birads"],
            race=self.configs.race,
            laterality=row["laterality"],
            exam_date=None,
            machine=self.configs.machine,
            exam_type=self.configs.image_modality,
            # NOTE: Normal exam, BUT technically each patient has an MLO and CC images. But no flag to differentiate them so we also save them together
            exam_imgs=exam_imgs_path,
            num_exam_imgs=number_exams if number_exams is not None else 0,
            segmentations_path=None,
            num_segmentations=0,
            slices_imgs_path=None,
            num_slices_imgs=0,
            slices_index=None,
            current_report=self.create_report(row, self.get_current_report_cols()),
            full_report=self.create_report(row, self.get_full_report_cols()),
            previous_exams=None,
        )

    @trycatch_func
    def save_imgs(
        self,
        folder_location: str,
        exam_id: str,
        patient_id: str,
        img_extension: str,
        laterality: str,
    ):

        searchable_folder = os.path.join(
            self.image_processor.raw_imgs_path, folder_location
        )
        available_imgs = sorted(
            glob.glob(f"{searchable_folder}/**/*{img_extension}", recursive=True)
        )

        if len(available_imgs) == 0:
            logging.warning(
                f"No images found for {folder_location} with extension {img_extension}"
            )
            return None, None

        acc = []
        img_shape = None

        for im in available_imgs:
            res = self.image_processor.process_image(im)

            if res is None:
                continue

            if is_left_right_breast(res) != convert_to_left_right_comparison(
                laterality
            ):
                logging.warning(
                    f"Left right breast mismatch for {folder_location}. Laterality: {laterality}, Image: {im}"
                )
                continue

            if img_shape is None:
                img_shape = res.shape
            elif img_shape != res.shape:
                continue

            acc.append(res)

        if len(acc) == 0:
            logging.warning(f"No images processed for {folder_location}")
            return None, None

        save_path = os.path.join(
            self.image_processor.processed_imgs_path,
            f"{patient_id}-{exam_id}{self.image_processor.SLICES_SUFFIX}",
        )
        return self.image_processor.save_process(save_path, acc), len(acc)
