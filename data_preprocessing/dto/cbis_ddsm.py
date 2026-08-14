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
from data_preprocessing.error_handling import log_func_info, trycatch_func
from data_preprocessing.image_processor import ImageProcessor
from data_preprocessing.medical_mappings import (
    UNKNOWN,
    birads_assessment,
    breast_density,
    dview,
    get_value,
    get_value_default,
    laterality,
)
from data_preprocessing.preprocessing_configs import CbisDDSMConfigs
from data_preprocessing.utils import column_cleaning_csv_reading


class CbisDDSM(BreastCancerDataset):
    def __init__(self, configs: CbisDDSMConfigs):
        super().__init__(csv_save_path=configs.csv_save_path)
        self.configs = configs
        self.image_processor = ImageProcessor(
            raw_imgs_path=configs.raw_imgs_path,
            processed_imgs_path=configs.processed_imgs_path,
            image_preprocessing_config=configs.img_preprocessing_config,
        )

        # combine calcifications
        calc_desc_train_df = column_cleaning_csv_reading(
            configs.calc_case_description_train
        )
        calc_desc_test_df = column_cleaning_csv_reading(
            configs.calc_case_description_test
        )
        calc_desc = pd.concat(
            [calc_desc_train_df, calc_desc_test_df], ignore_index=True
        )

        # combine masses
        mass_desc_train_df = column_cleaning_csv_reading(
            configs.mass_case_description_train
        )
        mass_desc_test_df = column_cleaning_csv_reading(
            configs.mass_case_description_test
        )
        mass_desc = pd.concat(
            [mass_desc_train_df, mass_desc_test_df], ignore_index=True
        )

        # merge them together
        self.full_df = pd.concat([calc_desc, mass_desc], ignore_index=True)
        self.full_df = self.full_df.replace(np.nan, None)
        self.full_df.drop_duplicates(
            subset=[
                "patient id",
                "breast density",
                "left or right breast",
                "image view",
                "abnormality type",
                "calc type",
                "calc distribution",
                "assessment",
                "pathology",
                "image file path",
                "mass shape",
                "mass margins",
            ],
            inplace=True,
        )

    def process_info(self):
        self.full_df = self.full_df.rename(
            columns={
                "left or right breast": "laterality",
                "calc type": "calcification type",
                "calc distribution": "calcification distribution",
            }
        )

        # adjust column values
        self.full_df["laterality"] = self.full_df["laterality"].apply(
            lambda x: get_value(x, laterality)
        )
        self.full_df["image view"] = self.full_df["image view"].apply(
            lambda x: get_value(x, dview)
        )
        self.full_df["breast density"] = self.full_df["breast density"].apply(
            lambda x: get_value(x, breast_density)
        )
        self.full_df["assessment"] = self.full_df["assessment"].apply(
            lambda x: get_value(x, birads_assessment)
        )

        # Add the modality
        self.full_df["modality"] = self.configs.image_modality

        # Second rename cleaning
        self.full_df = self.full_df.rename(
            columns={
                "image view": "exam view",
                "mass margins": "mass margin",
                "pathology": "classification",
            }
        )

        # Process the exams
        n = cpu_count() - 1
        df_split = [
            self.full_df.iloc[idx]
            for idx in np.array_split(np.arange(len(self.full_df)), n)
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
            "exam view",
            "breast density",
            "abnormality type",
            "calcification type",
            "calcification distribution",
            "mass shape",
            "mass margin",
            "classification",
        ]

    @staticmethod
    def get_all_report_cols():
        return CbisDDSM.get_current_report_cols() + ["laterality"]

    @trycatch_func
    def process_small_batch(self, df):
        curr_exams = []

        # Im sorry
        with tqdm(
            total=len(df),
            desc=f"Processing cbis-ddsm batch {df.index[0]} to {df.index[-1]}",
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

        exam_imgs_path, img_shape = self.save_slice(
            row["image file path"], exam_id, str(row["patient id"])
        )
        segmentation_path = self.save_segmentation(
            exam_id, str(row["patient id"]), row["roi mask file path"], img_shape
        )

        return ExamInformation(
            id=exam_id,
            patient=f"{self.get_dataset_name()}-{row['patient id']}",
            dataset=self.get_dataset_name(),
            modality=self.configs.image_modality,
            birads=row["assessment"],
            race=UNKNOWN,
            laterality=get_value_default(row["laterality"], laterality),
            exam_date=None,  # Left the study date out due to unmanageable reasons
            machine=self.configs.machine,
            # There is no immediate way to know unless we had to dig through and dirty to know it from the oirignal dataset. We leave as a conjunction
            exam_type=row["exam view"],
            exam_imgs=exam_imgs_path,
            num_exam_imgs=1 if exam_imgs_path is not None else 0,
            segmentations_path=segmentation_path,
            num_segmentations=1 if segmentation_path is not None else 0,
            slices_imgs_path=None,
            num_slices_imgs=0,
            slices_index=None,
            current_report=self.create_report(row, self.get_current_report_cols()),
            full_report=self.create_report(row, self.get_all_report_cols()),
            previous_exams=None,
        )

    @trycatch_func
    @log_func_info
    def save_slice(self, img_file_path: str, exam_id: str, patient_id: str):
        base_folder = img_file_path.split("/")[-2]
        path = os.path.join(
            self.configs.raw_imgs_path,
            base_folder,
            f"*{self.configs.raw_imgs_extension}",
        )
        im_path = glob.glob(path, recursive=True)[0]

        img = self.image_processor.process_image(im_path)
        if img is None:
            return None, None

        save_path = os.path.join(
            self.configs.processed_imgs_path,
            f"{patient_id}-{exam_id}{ImageProcessor.IMGS_SUFFIX}",
        )
        return self.image_processor.save_process(save_path, [img]), img.shape

    @trycatch_func
    @log_func_info
    def save_segmentation(
        self, exam_id: str, patient_id: str, roi_image_path: str, img_shape
    ) -> str | None:
        base_folder = roi_image_path.split("/")[-2]
        path = os.path.join(
            self.configs.raw_imgs_path,
            base_folder,
            f"*{self.configs.raw_imgs_extension}",
        )
        available_img_paths = glob.glob(path, recursive=True)

        if not available_img_paths or img_shape is None:
            return None

        mask_path = self.get_adequate_mask_path(available_img_paths)
        seg_mask = self.image_processor.process_segmentation_mask(mask_path)

        if seg_mask is not None:
            seg_save_path = os.path.join(
                self.configs.processed_imgs_path,
                f"{patient_id}-{exam_id}{ImageProcessor.SEGMENTATION_SUFFIX}",
            )
            return self.image_processor.save_process(seg_save_path, [seg_mask])

        return None

    def get_adequate_mask_path(self, available_img_paths: list):
        if len(available_img_paths) == 1:
            logging.warning(
                f"Only one image found for segmentation mask. Using {available_img_paths[0]}"
            )
            return available_img_paths[0]
        else:
            # read each image and select the one with the highest resolution
            shapes = []
            for img in available_img_paths:
                current_img = self.image_processor.read_image(img)
                if current_img is None:
                    continue
                shapes.append(current_img.shape[0] * current_img.shape[1])

            if len(shapes) == 0:
                return None

            return available_img_paths[np.argmax(shapes)]
