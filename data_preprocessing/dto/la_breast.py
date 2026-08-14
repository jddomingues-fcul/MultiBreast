import logging
import os
import uuid
from glob import glob
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from tqdm import tqdm

from data_preprocessing.breast_cancer_dataset import (
    BreastCancerDataset,
    ExamInformation,
)
from data_preprocessing.image_processor import ImageProcessor
from data_preprocessing.medical_mappings import (
    birads_assessment,
    breast_density,
    get_value,
)
from data_preprocessing.preprocessing_configs import LABreastConfig
from data_preprocessing.utils import column_cleaning_csv_reading


class LABreast(BreastCancerDataset):
    def __init__(self, configs: LABreastConfig):
        super().__init__(csv_save_path=configs.csv_save_path)
        self.configs = configs
        self.image_processor = ImageProcessor(
            raw_imgs_path=configs.raw_imgs_path,
            processed_imgs_path=configs.processed_imgs_path,
            image_preprocessing_config=configs.img_preprocessing_config,
        )

        self.train_df = column_cleaning_csv_reading(configs.train_csv_path)
        self.test_df = column_cleaning_csv_reading(configs.test_csv_path)
        self.val_df = column_cleaning_csv_reading(configs.val_csv_path)
        self.clinical_df = pd.concat(
            [self.train_df, self.test_df, self.val_df], ignore_index=True
        )

    def process_info(self):
        self.clinical_df["birads"] = self.clinical_df["birads"].apply(
            lambda x: get_value(x, birads_assessment)
        )
        self.clinical_df["acr"] = self.clinical_df["acr"].apply(
            lambda x: get_value(int(x), breast_density) if pd.notna(x) else None
        )
        self.clinical_df = self.clinical_df.replace(np.nan, None)

        # column renameing
        self.clinical_df.rename(
            columns={
                "acr": "breast density",
            },
            inplace=True,
        )

        # Split DataFrame by patient id
        grouped_patients = [group for _, group in self.clinical_df.groupby("patient")]

        # Process the exams
        n = cpu_count() - 1
        with Pool(processes=n) as p:
            results = list(
                tqdm(
                    p.imap(self.process_patient, grouped_patients),
                    total=len(grouped_patients),
                    desc="Processing LA Breast patients",
                    unit="patient",
                    ncols=100,
                    position=0,
                    leave=True,
                )
            )
        for result in results:
            for exam in result:
                self.append_exam(exam)

    def process_patient(self, patient_df):
        imgs_types = os.listdir(self.configs.raw_imgs_path)
        curr_exams = []

        if len(patient_df) == 0:
            logging.warning("Empty patient DataFrame, skipping...")
            return curr_exams

        # Get the patient id from the first row
        patient = patient_df.iloc[0]["patient"].replace("Breast_", "")
        patient_imgs = glob(
            os.path.join(
                self.configs.raw_imgs_path,
                "**",
                "**",
                f"{patient}*{self.configs.raw_imgs_extension}",
            )
        )

        # for each entry of the patient
        for _, row in patient_df.iterrows():
            # for each possible exam type, check if an image for the patient exists that matches it
            for im_type in imgs_types:
                curr_type_value = row[im_type].split("/")[-1].split(".")[0]
                roi_value = row["roi"].lower()
                found_value = False

                # find the image path that matches the current type and roi
                for im_path in patient_imgs:
                    if (
                        curr_type_value.lower() in im_path.lower()
                        and roi_value in im_path.lower()
                    ):
                        dx, dy, cx, cy = row[
                            [
                                f"distancia x {im_type}",
                                f"distancia y {im_type}",
                                f"centro x {im_type}",
                                f"centro y {im_type}",
                            ]
                        ]
                        exam_id = str(uuid.uuid4())
                        patient_id = f"{self.get_dataset_name()}-{row['patient']}"

                        exam_imgs_path, img_shape = self.process_and_save_slice(
                            im_path, exam_id, patient_id
                        )
                        segmentation_path = (
                            self.create_and_save_segmentation(
                                exam_id, patient_id, img_shape, (dx, dy, cx, cy)
                            )
                            if exam_imgs_path is not None
                            else None
                        )

                        exam = ExamInformation(
                            id=exam_id,
                            patient=patient_id,
                            dataset=self.get_dataset_name(),
                            modality=self.configs.image_modality,
                            birads=row["birads"],
                            race=self.configs.race,
                            laterality=None,
                            exam_date=None,
                            machine=self.configs.machine,
                            exam_type=LABreast.get_descriptive_exam_type(im_type),
                            exam_imgs=exam_imgs_path,
                            num_exam_imgs=1 if exam_imgs_path is not None else 0,
                            segmentations_path=segmentation_path,
                            num_segmentations=1 if segmentation_path is not None else 0,
                            slices_imgs_path=None,
                            num_slices_imgs=0,
                            slices_index=None,
                            current_report=self.create_report(
                                row, LABreast.get_current_report_cols(), None
                            ),
                            full_report=self.create_report(
                                row, LABreast.get_all_report_cols(), None
                            ),
                            previous_exams=None,
                        )
                        curr_exams.append(exam)

                        found_value = True
                        break

                # Ideally never happens
                if not found_value:
                    print(
                        f"Exam type: {im_type}, Image path not found for value: {curr_type_value}, ROI: {roi_value}, patient: {patient}"
                    )

        return curr_exams

    def process_and_save_slice(self, file_path, exam_id, patient_id):
        img = self.image_processor.read_image(file_path)

        if img is None:
            logging.warning(f"Could not read image for {file_path}")
            return None, None

        img_shape = img.shape
        img = self.image_processor.apply_processing(img, is_segmentation=False)

        if img is None:
            logging.warning(f"Could not process image for {file_path}")
            return None, None

        img_save_path = self.image_processor.save_image_set([img], exam_id, patient_id)
        return img_save_path, img_shape

    def create_and_save_segmentation(self, exam_id, patient_id, img_shape, coordinates):
        dx, dy, cx, cy = coordinates

        mask = np.zeros(img_shape, dtype=np.uint8)
        x1, y1 = int(cx - dx / 2), int(cy - dy / 2)
        x2, y2 = int(cx + dx / 2), int(cy + dy / 2)
        self.image_processor.draw_rectangle_on_image(mask, x1, y1, x2, y2)

        seg_mask = self.image_processor.apply_processing(mask, is_segmentation=True)
        seg_mask_path = self.image_processor.save_segmentation_set(
            [seg_mask], exam_id, patient_id
        )
        return seg_mask_path

    @staticmethod
    def get_current_report_cols():
        return [
            "breast density",
        ]

    @staticmethod
    def get_all_report_cols():
        return LABreast.get_current_report_cols()

    @staticmethod
    def get_descriptive_exam_type(exam_type: str) -> str:
        return {
            "d0": "pre-contrast T1 fat saturated Dynamic",
            "d1": "first postcontrast T1 fat saturated dynamics",
            "d2": "second postcontrast T1 fat saturated dynamics",
            "d3": "third postcontrast T1 fat saturated dynamics",
            "d4": "fourth postcontrast T1 fat saturated dynamics",
            "d5": "fifth postcontrast T1 fat saturated dynamics",
            "t1": "T1 with no fat saturation",
            "t2": "T2 with no fat saturation",
            "adc": "apparent diffusion coefficient image",
            "dif": "diffusion image",
            "f1": "first post-contrast phase sequence",
            "f2": "second post-contrast phase sequence",
            "f3": "third post-contrast phase sequence",
            "f4": "fourth post-contrast phase sequence",
            "f5": "fifth post-contrast phase sequence",
        }[exam_type]
