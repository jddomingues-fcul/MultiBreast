import glob
import logging
import os
import plistlib
import uuid
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from skimage.draw import polygon
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
    dview,
    get_value,
    laterality,
    yes_no_mapping,
)
from data_preprocessing.preprocessing_configs import InbreastConfig
from data_preprocessing.utils import csv_column_cleaning


class Inbreast(BreastCancerDataset):
    def __init__(self, configs: InbreastConfig):
        super().__init__(csv_save_path=configs.csv_save_path)
        self.configs = configs
        self.image_processor = ImageProcessor(
            raw_imgs_path=configs.raw_imgs_path,
            processed_imgs_path=configs.processed_imgs_path,
            image_preprocessing_config=configs.img_preprocessing_config,
        )

        self.clinical_df = pd.read_excel(configs.clinical_data_path)
        self.clinical_df.columns = csv_column_cleaning(list(self.clinical_df.columns))
        self.clinical_df = self.clinical_df.iloc[
            :-2
        ]  # the last two rows are totals, not needed

    def process_info(self):
        # Clean the clinical data
        self.clinical_df["laterality"] = self.clinical_df["laterality"].apply(
            lambda x: get_value(x, laterality)
        )
        self.clinical_df["view"] = self.clinical_df["view"].apply(
            lambda x: get_value(x, dview)
        )
        self.clinical_df["acquisition date"] = self.clinical_df[
            "acquisition date"
        ].apply(lambda x: Inbreast.clean_date(x) if pd.notna(x) else None)
        self.clinical_df["file name"] = self.clinical_df["file name"].apply(
            lambda x: str(int(x))
        )
        self.clinical_df["acr"] = self.clinical_df["acr"].apply(
            lambda x: get_value(x, breast_density)
        )  # the empty strings will be converted to None
        self.clinical_df["bi-rads"] = self.clinical_df["bi-rads"].apply(
            lambda x: get_value(x, birads_assessment)
        )
        self.clinical_df["mass"] = self.clinical_df["mass"].apply(
            lambda x: get_value(x, yes_no_mapping) if pd.notna(x) else None
        )
        self.clinical_df["micros"] = self.clinical_df["micros"].apply(
            lambda x: get_value(x, yes_no_mapping) if pd.notna(x) else None
        )
        self.clinical_df["distortion"] = self.clinical_df["distortion"].apply(
            lambda x: get_value(x, yes_no_mapping) if pd.notna(x) else None
        )
        self.clinical_df["asymmetry"] = self.clinical_df["asymmetry"].apply(
            lambda x: get_value(x, yes_no_mapping) if pd.notna(x) else None
        )
        self.clinical_df["lesion annotation status"] = self.clinical_df[
            "lesion annotation status"
        ].apply(lambda x: str(x).strip().lower() if pd.notna(x) else None)
        self.clinical_df["pectoral muscle annotation"] = self.clinical_df[
            "pectoral muscle annotation"
        ].apply(
            lambda x: (
                str(x).strip().lower()
                if pd.notna(x) and len(str(x).strip().lower()) > 0
                else None
            )
        )
        self.clinical_df["other annotations"] = self.clinical_df[
            "other annotations"
        ].apply(lambda x: str(x).strip().lower() if pd.notna(x) else None)
        self.clinical_df = self.clinical_df.replace(np.nan, None)

        # column renameing
        self.clinical_df.rename(
            columns={
                "acr": "breast density",
                "micros": "has microcalcifications",
                "view": "exam view",
                "distortion": "has distortion",
            },
            inplace=True,
        )

        # Process the exams
        n = cpu_count() - 1
        df_split = [
            self.clinical_df.iloc[idx]
            for idx in np.array_split(np.arange(len(self.clinical_df)), n)
            if len(idx) > 0
        ]
        with Pool(processes=n) as p:
            results = p.map(self.process_small_batch, df_split)

        for result in results:
            for exam in result:
                self.append_exam(exam)

    @trycatch_func
    def process_small_batch(self, df):
        curr_exams = []

        # Im sorry
        with tqdm(
            total=len(df),
            desc=f"Processing INBreast batch {df.index[0]} to {df.index[-1]}",
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
        patient_id = f"{self.get_dataset_name()}-{row['file name']}"

        exam_imgs_path, img_shape = self.find_and_save_slice(
            row["file name"], exam_id, patient_id
        )
        segmentation_path = (
            self.create_and_save_segmentation(
                exam_id, patient_id, img_shape, row["file name"]
            )
            if exam_imgs_path is not None
            else None
        )

        return ExamInformation(
            id=exam_id,
            patient=patient_id,
            dataset=self.get_dataset_name(),
            modality=self.configs.image_modality,
            birads=row["bi-rads"],
            race=UNKNOWN,
            laterality=row["laterality"],
            exam_date=row["acquisition date"],
            machine=UNKNOWN,
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

    def find_and_save_slice(self, file_name, exam_id, patient_id):
        available_imgs = sorted(
            glob.glob(
                f"{self.image_processor.raw_imgs_path}/**/*{self.configs.raw_imgs_extension}",
                recursive=True,
            )
        )
        available_imgs = [
            img for img in available_imgs if str(file_name) in os.path.basename(img)
        ]
        if len(available_imgs) == 0:
            logging.warning(f"Could not find image for {file_name}")
            return None, None

        if len(available_imgs) > 1:
            logging.warning(
                f"Found multiple images for {file_name}, using the first one"
            )

        img_path = available_imgs[0]
        img = self.image_processor.read_image(img_path)

        if img is None:
            logging.warning(f"Could not read image for {file_name}")
            return None, None

        img_shape = img.shape
        img = self.image_processor.apply_processing(img, is_segmentation=False)

        if img is None:
            logging.warning(f"Could not process image for {file_name}")
            return None, None

        img_save_path = self.image_processor.save_image_set([img], exam_id, patient_id)
        return img_save_path, img_shape

    def create_and_save_segmentation(self, exam_id, patient_id, img_shape, file_name):
        search_path = f"{self.configs.segmentations_xml_folder}/**/*{self.configs.raw_segs_extension}"
        available_imgs = sorted(glob.glob(search_path, recursive=True))
        available_imgs = [
            img for img in available_imgs if str(file_name) in os.path.basename(img)
        ]

        if len(available_imgs) == 0:
            logging.info(
                f"Could not find segmentation for {file_name} in {search_path}"
            )
            return None

        if len(available_imgs) > 1:
            logging.info(
                f"Found multiple segmentations for {file_name}, using the first one"
            )

        seg_path = available_imgs[0]
        seg_mask = Inbreast.load_inbreast_mask(seg_path, img_shape)

        if seg_mask is None:
            logging.warning(f"Could not process segmentation for {file_name}")
            return None

        seg_mask = self.image_processor.apply_processing(seg_mask, is_segmentation=True)
        seg_mask_path = self.image_processor.save_segmentation_set(
            [seg_mask], exam_id, patient_id
        )
        return seg_mask_path

    @staticmethod
    def get_current_report_cols():
        return [
            "exam view",
            "breast density",
            "has microcalcifications",
            "has distortion",
        ]

    @staticmethod
    def get_all_report_cols():
        return Inbreast.get_current_report_cols() + [
            "laterality",
            "pectoral muscle annotation",
            "other annotations",
            "findings notes in english",
        ]

    @trycatch_func
    @staticmethod
    def clean_date(date_str):
        stringed_date = str(int(date_str))
        return "-".join([stringed_date[:-2], stringed_date[-2:]])

    @trycatch_func
    @staticmethod
    def load_inbreast_mask(mask_path, imshape):
        # taken from: https://www.kaggle.com/code/lethanhnghia/breastcancercnn
        def load_point(point_string):
            x, y = tuple([float(num) for num in point_string.strip("()").split(",")])
            return y, x

        mask = np.zeros(imshape)
        with open(mask_path, "rb") as mask_file:
            plist_dict = plistlib.load(mask_file, fmt=plistlib.FMT_XML)["Images"][0]
            numRois = plist_dict["NumberOfROIs"]
            rois = plist_dict["ROIs"]
            assert len(rois) == numRois
            for roi in rois:
                numPoints = roi["NumberOfPoints"]
                points = roi["Point_px"]
                assert numPoints == len(points)
                points = [load_point(point) for point in points]
                if len(points) <= 2:
                    for point in points:
                        mask[int(point[0]), int(point[1])] = 1
                else:
                    x, y = zip(*points)
                    x, y = np.array(x), np.array(y)
                    poly_x, poly_y = polygon(x, y, shape=imshape)
                    mask[poly_x, poly_y] = 1
        return mask
