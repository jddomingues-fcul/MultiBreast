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
from data_preprocessing.medical_mappings import (
    UNKNOWN,
    birads_assessment,
    breast_density,
    dview,
    get_value,
    get_value_default,
    laterality,
    yes_no_mapping,
)
from data_preprocessing.preprocessing_configs import RsnaBCDConfig
from data_preprocessing.utils import column_cleaning_csv_reading


class RsnaBCD(BreastCancerDataset):
    def __init__(self, configs: RsnaBCDConfig):
        super().__init__(csv_save_path=configs.csv_save_path)
        self.train_df = column_cleaning_csv_reading(configs.train_csv_path)
        self.test_df = column_cleaning_csv_reading(configs.test_csv_path)
        self.train_df = self.train_df.replace(np.nan, None)
        self.test_df = self.test_df.replace(np.nan, None)

        self.configs = configs
        self.image_processor = ImageProcessor(
            raw_imgs_path=configs.raw_imgs_path,
            processed_imgs_path=configs.processed_imgs_path,
            image_preprocessing_config=configs.img_preprocessing_config,
        )

    def process_info(self):
        # Drop irrelevant columns
        train_cols_to_drop = ["site id"]
        test_cols_to_drop = ["site id", "prediction id"]
        self.train_df = self.train_df.drop(columns=train_cols_to_drop)
        self.test_df = self.test_df.drop(columns=test_cols_to_drop)

        # Add column for set of images folder
        self.train_df["raw folder"] = "train_images"
        self.test_df["raw folder"] = "test_images"

        # Birads mapping
        self.train_df["birads"] = self.train_df["birads"].apply(
            lambda x: get_value(int(x), birads_assessment) if x is not None else None
        )

        # test df does not have birads, so we set all to none, as we can stil use the images for SSL pre-train
        self.test_df["birads"] = None

        # cancer, biopsy, implant, and invasive mapping to yes and no responses
        self.train_df["cancer"] = self.train_df["cancer"].apply(
            lambda x: get_value(x, yes_no_mapping)
        )
        self.train_df["biopsy"] = self.train_df["biopsy"].apply(
            lambda x: get_value(x, yes_no_mapping)
        )
        self.train_df["implant"] = self.train_df["implant"].apply(
            lambda x: get_value(x, yes_no_mapping)
        )
        self.train_df["invasive"] = self.train_df["invasive"].apply(
            lambda x: get_value(x, yes_no_mapping)
        )
        self.test_df["implant"] = self.test_df["implant"].apply(
            lambda x: get_value(x, yes_no_mapping)
        )
        self.train_df["laterality"] = self.train_df["laterality"].apply(
            lambda x: get_value(x, laterality)
        )
        self.test_df["laterality"] = self.test_df["laterality"].apply(
            lambda x: get_value(x, laterality)
        )
        self.train_df["density"] = self.train_df["density"].apply(
            lambda x: get_value(x, breast_density)
        )

        self.train_df["view"] = self.train_df["view"].apply(
            lambda x: get_value(x, dview)
        )
        self.test_df["view"] = self.test_df["view"].apply(lambda x: get_value(x, dview))

        # adjust the age to be an integer
        self.train_df["age"] = self.train_df["age"].apply(
            lambda x: int(x) if x is not None else None
        )

        # merge both dfs and rename cols
        col_rename = {
            "cancer": "is malign cancer",
            "biopsy": "follow up biopsy",
            "invasive": "has cancer",
            "implant": "breast implants",
            "density": "breast density",
            "view": "exam view",
        }
        combined_df = pd.concat([self.train_df, self.test_df], ignore_index=True)
        combined_df = combined_df.replace(np.nan, None)
        combined_df = combined_df.rename(columns=col_rename)

        # Add the modality
        combined_df["modality"] = self.configs.image_modality

        # Process the exams
        n = cpu_count() - 1
        df_split = [
            combined_df.iloc[idx]
            for idx in np.array_split(np.arange(len(combined_df)), n)
            if len(idx) > 0
        ]
        with Pool(processes=n) as p:
            results = p.map(self.process_small_batch, df_split)

        for result in results:
            for exam in result:
                self.append_exam(exam)

    def process_small_batch(self, df):
        curr_exams = []

        # Im sorry
        with tqdm(
            total=len(df),
            desc=f"Processing advanced rsna bcd batch {df.index[0]} to {df.index[-1]}",
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

    @staticmethod
    def get_current_report_columns():
        return [
            "exam view",
            "breast density",
            "has cancer",
            "is malign cancer",
        ]

    @staticmethod
    def get_full_report_columns():
        return RsnaBCD.get_current_report_columns() + [
            "age",
            # "breast implants", #!: Breast implants removed from the current report because there is not indication of the breast implants per image, and for the same patients there are images with and without implants
            "laterality",
            "follow up biopsy",
        ]

    def process_row(self, row):
        # get needed info for exam
        exam_id = str(uuid.uuid4())
        img_path = os.path.join(
            self.configs.raw_imgs_path,
            row["raw folder"],
            str(row["patient id"]),
            f"{row['image id']!s}{self.configs.raw_imgs_extension}",
        )
        exam_imgs_path = self.image_processor.process_and_save_image(
            img_path, exam_id, str(row["patient id"])
        )

        return ExamInformation(
            id=exam_id,
            patient=f"{self.get_dataset_name()}-{row['patient id']}",
            dataset=self.get_dataset_name(),
            modality=self.configs.image_modality,
            birads=row["birads"],
            race=UNKNOWN,
            laterality=get_value_default(row["laterality"], laterality),
            exam_date=None,
            machine=row[
                "machine id"
            ],  # this machine id is not a description of the machine, but a unique id for the machine within the dataset
            exam_type=row["exam view"],
            exam_imgs=exam_imgs_path,
            num_exam_imgs=1 if exam_imgs_path is not None else 0,
            segmentations_path=None,
            num_segmentations=0,
            slices_imgs_path=None,
            num_slices_imgs=0,
            slices_index=None,
            current_report=self.create_report(row, self.get_current_report_columns()),
            full_report=self.create_report(row, self.get_full_report_columns()),
            previous_exams=None,
        )
