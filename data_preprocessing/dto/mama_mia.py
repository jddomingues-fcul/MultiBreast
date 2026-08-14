import logging
import os
import uuid
from collections import namedtuple
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
    breast_density,
    get_value,
    mammaprint_70_gene_risk,
    pos_neg_mapping,
    race_mappings,
    yes_no_mapping,
)
from data_preprocessing.preprocessing_configs import MamaMiaConfig
from data_preprocessing.utils import csv_column_cleaning


class MamaMia(BreastCancerDataset):
    def __init__(self, configs: MamaMiaConfig):
        super().__init__(csv_save_path=configs.csv_save_path)
        self.configs = configs
        self.image_processor = ImageProcessor(
            raw_imgs_path=configs.raw_imgs_path,
            processed_imgs_path=configs.processed_imgs_path,
            image_preprocessing_config=configs.img_preprocessing_config,
        )

        self.clinical_df = pd.read_excel(configs.clinical_data_path)
        self.clinical_df = self.clinical_df.replace(np.nan, None)
        self.clinical_df.columns = csv_column_cleaning(list(self.clinical_df.columns))

    def process_info(self):

        # Adjust values
        self.clinical_df["bilateral breast cancer"] = self.clinical_df[
            "bilateral breast cancer"
        ].apply(lambda x: get_value(x, yes_no_mapping))
        self.clinical_df["multifocal cancer"] = self.clinical_df[
            "multifocal cancer"
        ].apply(lambda x: get_value(int(x), yes_no_mapping) if x is not None else None)
        self.clinical_df["endocrine therapy"] = self.clinical_df[
            "endocrine therapy"
        ].apply(lambda x: get_value(int(x), yes_no_mapping) if x is not None else None)
        self.clinical_df["anti her2 neu therapy"] = self.clinical_df[
            "anti her2 neu therapy"
        ].apply(lambda x: get_value(int(x), yes_no_mapping) if x is not None else None)
        self.clinical_df["pcr"] = self.clinical_df["pcr"].apply(
            lambda x: get_value(int(x), yes_no_mapping) if x is not None else None
        )
        self.clinical_df["mastectomy post nac"] = self.clinical_df[
            "mastectomy post nac"
        ].apply(lambda x: get_value(int(x), yes_no_mapping) if x is not None else None)
        self.clinical_df["hr"] = self.clinical_df["hr"].apply(
            lambda x: get_value(int(x), pos_neg_mapping) if x is not None else None
        )
        self.clinical_df["pr"] = self.clinical_df["pr"].apply(
            lambda x: get_value(int(x), pos_neg_mapping) if x is not None else None
        )
        self.clinical_df["her2"] = self.clinical_df["her2"].apply(
            lambda x: get_value(int(x), pos_neg_mapping) if x is not None else None
        )
        self.clinical_df["mammaprint"] = self.clinical_df["mammaprint"].apply(
            lambda x: (
                get_value(int(x), mammaprint_70_gene_risk) if x is not None else None
            )
        )
        self.clinical_df["ethnicity"] = self.clinical_df["ethnicity"].apply(
            lambda x: get_value(x, race_mappings) if x is not None else None
        )
        self.clinical_df["has implant"] = self.clinical_df["has implant"].apply(
            lambda x: get_value(x, yes_no_mapping)
        )
        self.clinical_df["breast density"] = self.clinical_df["breast density"].apply(
            lambda x: get_value(x, breast_density) if x is not None else None
        )
        self.clinical_df["bilateral mri"] = self.clinical_df["bilateral mri"].apply(
            lambda x: get_value(x, yes_no_mapping)
        )
        self.clinical_df["fat suppressed"] = self.clinical_df["fat suppressed"].apply(
            lambda x: get_value(x, yes_no_mapping)
        )

        # Column renaming for better context
        col_rename = {
            "days to follow up": "days to last follow up or death",
            "mammaprint": "mammaprint 70-gene assay test score (risk class)",
            "oncotype": "oncotype dx recurrence score",
            "tumor subtype": "tumor subtype from receptor status",
            "age": "age at first screening",
            "menopause": "menopause status at first screening",
            "view": "exam view",
            "has implant": "breast implants",
        }

        self.clinical_df.rename(columns=col_rename, inplace=True)

        # Split DataFrame by patient id
        grouped_patients = [
            group for _, group in self.clinical_df.groupby("patient id")
        ]

        # Process the exams
        n = cpu_count() - 1
        with Pool(processes=n) as p:
            results = list(
                tqdm(
                    p.imap(self.process_patient, grouped_patients),
                    total=len(grouped_patients),
                    desc="Processing mama-mia patients",
                    unit="patient",
                    ncols=100,
                    position=0,
                    leave=True,
                )
            )
        for result in results:
            for exam in result:
                self.append_exam(exam)

    def process_patient(self, grpo):
        curr_exams = []

        if len(grpo) > 1:
            logging.warning(
                f"Patient {grpo['patient id'].iloc[0]} has more than one row in the clinical data. Only the first row will be used."
            )

        grpo = grpo.iloc[
            0
        ]  # NOTE: The dataframe per patient will always only 1 row, so we access it this way

        available_imgs = sorted(
            os.listdir(os.path.join(self.configs.raw_imgs_path, grpo["patient id"]))
        )
        available_imgs = [
            path
            for path in available_imgs
            if path.endswith(self.configs.raw_imgs_extension)
        ]

        exams_types = (
            self.get_exams_types(eval(grpo["acquisition times"]))
            if pd.notna(grpo["acquisition times"])
            else [UNKNOWN] * len(available_imgs)
        )

        seg_path = os.path.join(
            self.configs.raw_segs_path,
            f"{grpo['patient id']}{self.configs.raw_imgs_extension}",
        )
        seg_slices = self.image_processor.read_image(seg_path)
        if seg_slices is not None:
            seg_slices = np.array(
                [
                    self.image_processor.apply_processing(ss, is_segmentation=True)
                    for ss in seg_slices
                ]
            )
        else:
            logging.warning(f"Segmentation {seg_path} was not able to be read")

        for i, exam_path in enumerate(available_imgs):
            exam_id = str(uuid.uuid4())

            exam_path = os.path.join(
                self.configs.raw_imgs_path, grpo["patient id"], exam_path
            )

            (
                imgs_path,
                n_imgs,
                segs_path,
                n_segs,
                slices_indexes,
                slices_path,
                n_slices,
            ) = self.process_mri_exam(
                exam_path, seg_slices, exam_id, grpo["patient id"]
            )

            exam = ExamInformation(
                id=exam_id,
                patient=grpo["patient id"],
                dataset=grpo["dataset"],
                modality=self.configs.image_modality,
                birads=self.configs.birads,
                race=grpo["ethnicity"] if grpo["ethnicity"] is not None else UNKNOWN,
                laterality=None,
                exam_date=grpo["acquisition date"],
                machine=f"{grpo['manufacturer']} {grpo['scanner model']}",
                exam_type=exams_types[i],
                exam_imgs=imgs_path,
                num_exam_imgs=n_imgs,
                segmentations_path=segs_path,
                num_segmentations=n_segs,
                slices_imgs_path=slices_path,
                num_slices_imgs=n_slices,
                slices_index=slices_indexes,
                current_report=self.create_report(grpo, self.get_current_report_cols()),
                full_report=self.create_report(grpo, self.get_all_report_cols()),
                previous_exams=None,
            )
            curr_exams.append(exam)

        return curr_exams

    @trycatch_func
    @log_func_info
    def process_mri_exam(
        self,
        exam_path: str,
        seg_slices: np.ndarray | None,
        exam_id: str,
        patient_id: str,
    ):
        MRIExamResult = namedtuple(
            "MRIExamResult",
            [
                "sois",
                "num_sois",
                "segs",
                "num_segs",
                "sois_indexes",
                "slices_path",
                "n_slices",
            ],
        )

        if seg_slices is None:
            return MRIExamResult(
                sois=None,
                num_sois=0,
                segs=None,
                num_segs=0,
                sois_indexes=None,
                slices_path=None,
                n_slices=0,
            )

        exam_slices = self.image_processor.read_image(exam_path)
        if exam_slices is not None:
            processed_slices = np.array(
                [self.image_processor.apply_processing(slice) for slice in exam_slices]
            )
        else:
            logging.warning(f"Exam images of {exam_path} were not able to be read")
            return MRIExamResult(
                sois=None,
                num_sois=0,
                segs=None,
                num_segs=0,
                sois_indexes=None,
                slices_path=None,
                n_slices=0,
            )

        relevant_exams = []
        relevant_segs = []
        relevant_indexes = []

        for i in range(seg_slices.shape[0]):
            if np.sum(seg_slices[i]) > 0:  # we have a non-empty segmentation
                relevant_exams.append(processed_slices[i])
                relevant_segs.append(seg_slices[i])
                relevant_indexes.append(i)

        sois_path = self.image_processor.save_image_set(
            imgs=relevant_exams, exam_id=exam_id, patient_id=patient_id
        )
        segs_path = self.image_processor.save_segmentation_set(
            segs=relevant_segs, exam_id=exam_id, patient_id=patient_id
        )
        slices_path = self.image_processor.save_slices_set(
            imgs=list(processed_slices), exam_id=exam_id, patient_id=patient_id
        )

        return MRIExamResult(
            sois=sois_path,
            num_sois=len(relevant_exams),
            sois_indexes=tuple(relevant_indexes),
            segs=segs_path,
            num_segs=len(relevant_segs),
            slices_path=slices_path,
            n_slices=len(processed_slices),
        )

    @staticmethod
    def get_current_report_cols():
        return [
            "bilateral mri",
            "exam view",
            "breast implants",
            "breast density",
            "multifocal cancer",
        ]

    @staticmethod
    def get_all_report_cols():
        return MamaMia.get_current_report_cols() + [
            "bilateral breast cancer",
            "nac agent",
            "endocrine therapy",
            "anti her2 neu therapy",
            "pcr",
            "mastectomy post nac",
            "days to last follow up or death",
            "days to recurrence",
            "days to metastasis",
            "days to death",
            "hr",
            "er",
            "pr",
            "her2",
            "mammaprint 70-gene assay test score (risk class)",
            "oncotype score",
            "nottingham grade",
            "tumor subtype from receptor status",
            "age at first screening",
            "menopause status at first screening",
            "ethnicity",
            "weight",
            "patient size",
            "bmi group",
            "fat suppressed",
        ]

    def get_exams_types(self, acquisition_times: list) -> list[str]:
        desc = []

        for i, value in enumerate(acquisition_times):
            if i == 0:
                curr = "pre contrast phase"
            else:
                curr = f"contrast phase #{i}"

            curr = curr + f". Acquisition time: {value}s"
            desc.append(curr)

        return desc
