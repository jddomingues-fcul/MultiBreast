import glob
import logging
import os
import uuid
from copy import deepcopy
from functools import partial
from multiprocessing import Pool, cpu_count

import cv2
import numpy as np
import pandas as pd
import pydicom as dicom
from tqdm import tqdm

from data_preprocessing.breast_cancer_dataset import (
    BreastCancerDataset,
    ExamInformation,
)
from data_preprocessing.image_processor import ImageProcessor
from data_preprocessing.medical_mappings import (
    UNKNOWN,
    adjust_ki67,
    adjust_rsii,
    birads_assessment,
    get_pos_coord_slice,
    get_value,
    grade_values,
    laterality,
    referral_reasons,
    tumor_benign_col,
    tumor_pathology_values,
    yes_no_mapping,
)
from data_preprocessing.preprocessing_configs import AdvancedMRILesionsConfig
from data_preprocessing.utils import column_cleaning_csv_reading, csv_column_cleaning


class AdvancedMRILesions(BreastCancerDataset):
    def __init__(self, configs: AdvancedMRILesionsConfig):
        super().__init__(csv_save_path=configs.csv_save_path)
        self.configs = configs
        self.image_processor = ImageProcessor(
            raw_imgs_path=configs.raw_imgs_path,
            processed_imgs_path=configs.processed_imgs_path,
            image_preprocessing_config=configs.img_preprocessing_config,
        )
        self.mri_lesions_df = pd.read_excel(
            configs.mri_lesions_path, sheet_name=0, header=1
        )
        self.metadata_df = column_cleaning_csv_reading(
            configs.metadata_path
        ).reset_index()

    def process_info(self):
        # clean mri lesions
        self.mri_lesions_df.columns = [
            col.replace("id#", "")
            for col in csv_column_cleaning(list(self.mri_lesions_df.columns))
        ]
        self.mri_lesions_df = self.mri_lesions_df.replace(np.nan, None)
        self.mri_lesions_df = self.mri_lesions_df.replace(-1, None)
        self.mri_lesions_df = self.mri_lesions_df.replace(-1.0, None)
        self.mri_lesions_df = self.mri_lesions_df.rename(
            columns={"patient id": "subject id"}
        )

        # clean metadata
        adjusted_columns = list(self.metadata_df.columns[1:])
        adjusted_columns.insert(adjusted_columns.index("file size"), UNKNOWN)
        self.metadata_df.columns = adjusted_columns
        self.metadata_df = self.metadata_df.replace(np.nan, None)

        # only keep needed columns
        meta_cols_to_keep = [
            "subject id",
            "study date",
            "study description",
            "series description",
            "file location",
            "modality",
        ]
        self.metadata_df = self.metadata_df[meta_cols_to_keep]

        # Unnamed columns are dropped
        self.mri_lesions_df = self.mri_lesions_df.drop(
            columns=[col for col in self.mri_lesions_df.columns if "unnamed" in col]
        )

        # Create a copy for MRIs and SEG
        seg_metadata_df = deepcopy(
            self.metadata_df[self.metadata_df["modality"] == "SEG"]
        ).reset_index(drop=True)
        mr_metadata_df = deepcopy(
            self.metadata_df[self.metadata_df["modality"] == "MR"]
        ).reset_index(drop=True)

        # Create the segmentation paths in the segmentation metadata
        seg_metadata_df["segmentation path"] = seg_metadata_df["file location"].apply(
            lambda x: glob.glob(
                os.path.join(self.configs.raw_imgs_path, x[2:], "*"), recursive=True
            )[0]
        )
        seg_metadata_df = seg_metadata_df[["subject id", "segmentation path"]]

        # join both datasets
        self.mri_lesions_df = pd.merge(
            self.mri_lesions_df, mr_metadata_df, on=["subject id"], how="inner"
        )
        self.mri_lesions_df = pd.merge(
            self.mri_lesions_df, seg_metadata_df, on=["subject id"], how="left"
        )

        # Remove the segmentation paths for the images that do not match metadata indication
        self.mri_lesions_df["segmentation path"] = self.mri_lesions_df.apply(
            lambda x: self.remove_non_mri_segmentation_paths(x), axis=1
        )  # type: ignore

        # fill in contents
        self.fill_in_contents()

        # Rename dictionary
        rename_dict = {
            "laterality lesion1": "first lesion laterality",
            "laterality lesion2": "second lesion laterality",
            "laterality lesion3": "third lesion laterality",
            "laterality lesion4": "fourth lesion laterality",
            "laterality lesion5": "fifth lesion laterality",
            "laterality lesion6": "sixth lesion laterality",
            "tumor/benign1": "first lesion tumor or benign",
            "pathology1": "first lesion pathology",
            "grade1": "first lesion grade",
            "er [sii] 1": "first lesion ER staining intensity",
            "pr [sii] 1": "first lesion PR staining intensity",
            "her2 [sii] 1": "first lesion HER2 staining intensity",
            "is tn1": "first lesion triple negative",
            "er [%] 1": "first lesion ER percentage",
            "pr [%] 1": "first lesion PR percentage",
            "her2 [%] 1": "first lesion HER2 percentage",
            "ki67[%] 1": "first lesion KI67 percentage",
            "tumor/benign2": "second lesion tumor or benign",
            "pathology2": "second lesion pathology",
            "grade2": "second lesion grade",
            "er [sii] 2": "second lesion ER staining intensity",
            "pr [sii] 2": "second lesion PR staining intensity",
            "her2 [sii] 2": "second lesion HER2 staining intensity",
            "is tn2": "second lesion triple negative",
            "er [%] 2": "second lesion ER percentage",
            "pr [%] 2": "second lesion PR percentage",
            "her [%] 2": "second lesion HER2 percentage",
            "ki67[%] 2": "second lesion KI67 percentage",
            "tumor/benign3": "third lesion tumor or benign",
            "pathology3": "third lesion pathology",
            "grade3": "third lesion grade",
            "er [sii] 3": "third lesion ER staining intensity",
            "pr [sii] 3": "third lesion PR staining intensity",
            "her [sii] 3": "third lesion HER2 staining intensity",
            "is tn3": "third lesion triple negative",
            "er [%] 3": "third lesion ER percentage",
            "pr [%] 3": "third lesion PR percentage",
            "her [%] 3": "third lesion HER2 percentage",
            "ki67[%] 3": "third lesion KI67 percentage",
            "tumor/benign4": "fourth lesion tumor or benign",
            "pathology4": "fourth lesion pathology",
            "tumor/benign5": "fifth lesion tumor or benign",
            "pathology5": "fifth lesion pathology",
            "tumor/benign6": "sixth lesion tumor or benign",
            "pathology6": "sixth lesion pathology",
        }

        self.mri_lesions_df = self.mri_lesions_df.rename(columns=rename_dict)
        self.mri_lesions_df = self.mri_lesions_df.replace(np.nan, None)

        # Process the exams
        n = cpu_count() - 1
        df_split = [
            self.mri_lesions_df.iloc[idx]
            for idx in np.array_split(np.arange(len(self.mri_lesions_df)), n)
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
            "breast implants",
            "benign lesions pathology",
            "malign lesions pathology",
        ]

    @staticmethod
    def get_all_report_cols():
        return AdvancedMRILesions.get_current_report_cols() + [
            "study description",
            "series description",
            "age at mri",
            "reason for referral ",
            "additional reason for referral ",
            "first lesion laterality",
            "second lesion laterality",
            "third lesion laterality",
            "fourth lesion laterality",
            "fifth lesion laterality",
            "sixth lesion laterality",
            "first lesion tumor or benign",
            "first lesion grade",
            "first lesion ER staining intensity",
            "first lesion PR staining intensity",
            "first lesion HER2 staining intensity",
            "first lesion triple negative",
            "first lesion ER percentage",
            "first lesion PR percentage",
            "first lesion HER2 percentage",
            "first lesion KI67 percentage",
            "second lesion tumor or benign",
            "second lesion grade",
            "second lesion ER staining intensity",
            "second lesion PR staining intensity",
            "second lesion HER2 staining intensity",
            "second lesion triple negative",
            "second lesion ER percentage",
            "second lesion PR percentage",
            "second lesion HER2 percentage",
            "second lesion KI67 percentage",
            "third lesion tumor or benign",
            "third lesion grade",
            "third lesion ER staining intensity",
            "third lesion PR staining intensity",
            "third lesion HER2 staining intensity",
            "third lesion triple negative",
            "third lesion ER percentage",
            "third lesion PR percentage",
            "third lesion HER2 percentage",
            "third lesion KI67 percentage",
            "fourth lesion tumor or benign",
            "fifth lesion tumor or benign",
            "sixth lesion tumor or benign",
            "study date",
        ]

    def process_small_batch(self, df):
        curr_exams = []

        # Im sorry
        with tqdm(
            total=len(df),
            desc=f"Processing advanced mri lesions batch {df.index[0]} to {df.index[-1]}",
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
        exam_id = str(uuid.uuid4())
        folder_path = row["file location"][2:]  # to remove the "./" from the beginning

        # None of the indicated positions match in these images slices so we skip them
        sois_path, sois_indexes, segs_sois_path, n_segmentations = (
            self.save_sois_and_segs(
                folder_path,
                exam_id,
                str(row["subject id"]),
                row["segmentation path"],
                [
                    row["slice coord lesion1"],
                    row["slice coord lesion2"],
                    row["slice coord lesion3"],
                    row["slice coord lesion4"],
                    row["slice coord lesion5"],
                    row["slice coord lesion6"],
                ],
            )
        )

        # Regardless of the previous condition, we always save slices in case of need
        slices_imgs_path, slices_slices = self.image_processor.save_all_slices(
            folder_path, exam_id, str(row["subject id"])
        )

        return ExamInformation(
            id=exam_id,
            patient=str(row["subject id"]),
            dataset=self.get_dataset_name(),
            modality=self.configs.image_modality,
            birads=row["birads"],
            race=UNKNOWN,
            laterality=None,
            exam_date=row["study date"],
            machine=self.configs.machine,
            exam_type=f"{row['study description']} {row['series description']}",
            exam_imgs=sois_path,
            num_exam_imgs=len(sois_indexes),
            segmentations_path=segs_sois_path,
            num_segmentations=n_segmentations,
            slices_imgs_path=slices_imgs_path,
            num_slices_imgs=len(slices_slices) if slices_slices is not None else 0,
            slices_index=tuple(sois_indexes),
            # reports are only used for the images that have the segmentation associated. The others are used for pre-training an encoder model e.g.
            current_report=self.create_report(row, self.get_current_report_cols())
            if row["segmentation path"] is not None
            else None,
            full_report=self.create_report(row, self.get_all_report_cols())
            if row["segmentation path"] is not None
            else None,
            previous_exams=None,
        )

    @staticmethod
    def remove_non_mri_segmentation_paths(row):
        # NOTE: Only these samples actually match on the segmentation when manually validating metadata
        desc = row["series description"].lower()
        if desc != "Registered AX Sen Vibrant MultiPhase".lower():
            return None
        return row["segmentation path"]

    def compose_lesion_pathology(self, row, for_benign: bool = True):
        pathologies = []
        for i in range(1, 7):
            if (
                pd.isna(row[f"pathology{i}"]) or row[f"pathology{i}"] == 15
            ):  # special case for benign cases, not seen in imri guided biopsy
                continue
            if (
                row[f"pathology{i}"] in range(11, 24, 1) and for_benign
            ):  # from 11 to 23 are benign lesions
                pathologies.append(
                    get_value(int(row[f"pathology{i}"]), tumor_pathology_values)
                )
            elif (
                row[f"pathology{i}"] not in range(11, 24, 1) and not for_benign
            ):  # the rest are malign lesions
                pathologies.append(
                    get_value(int(row[f"pathology{i}"]), tumor_pathology_values)
                )

        pathologies = list(set(pathologies))  # remove duplicates
        return ", ".join(pathologies) if pathologies else None

    def save_sois_and_segs(self, folder_path, exam_id, patient_id, seg_path, slices):
        sois = []
        sois_indexes = []
        searchable_folder = os.path.join(
            self.image_processor.raw_imgs_path, folder_path
        )
        available_imgs = sorted(glob.glob(f"{searchable_folder}/*", recursive=True))

        segs = []
        seg_data = None

        if pd.isna(seg_path) or not os.path.exists(seg_path):
            logging.warning(
                f"Segmentation path does not exist for patient {patient_id} on the images at path {folder_path}. Saving images only."
            )
        else:
            seg_data = dicom.dcmread(seg_path)

        for i, img in enumerate(available_imgs):
            if os.path.isdir(img) or not os.path.exists(img):
                continue

            dicom_data = dicom.dcmread(img)

            for arg in slices:
                if pd.isna(arg):
                    continue

                if arg in str(
                    round(dicom_data["SliceLocation"].value, 2)
                ):  # rounding to 2 decimals to match the values in the provided excel
                    curr_img = self.image_processor.process_image(img)

                    if seg_data is None:
                        sois.append(curr_img)
                        sois_indexes.append(i)
                        continue

                    corresponding_seg = self.get_corresponding_seg(seg_data, arg)

                    if corresponding_seg is not None:
                        sois.append(curr_img)
                        sois_indexes.append(i)
                        segs.append(corresponding_seg)

        if len(sois_indexes) == 0:
            return None, sois_indexes, None, 0

        sois_path = self.image_processor.save_image_set(sois, exam_id, patient_id)

        segs_path = None
        if len(segs) > 0:
            segs_path = self.image_processor.save_segmentation_set(
                segs, exam_id, patient_id
            )

        return sois_path, sois_indexes, segs_path, len(segs)

    def get_corresponding_seg(self, seg_data, slice):
        seg_index = -1
        for i, gs in enumerate(seg_data["PerFrameFunctionalGroupsSequence"].value):
            slic_coord = str(
                round(
                    gs["PlanePositionSequence"]
                    .value[0]["ImagePositionPatient"]
                    .value[-1],
                    2,
                )
            )
            if slice in slic_coord:
                seg_index = i
                break

        if seg_index == -1:
            return None

        seg_index = seg_index + 1  # 1-based index
        seg_slice = seg_data.pixel_array[seg_index]

        data_min, data_max = seg_slice.min(), seg_slice.max()

        if data_max == data_min:
            return None

        # Normalize the image for cv2
        data = cv2.normalize(seg_slice, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)  # type: ignore
        data = self.image_processor.apply_processing(data, is_segmentation=True)
        return data

    def fill_in_contents(self):
        # fill contents of referral
        self.mri_lesions_df["reason for referral "] = self.mri_lesions_df[
            "reason for referral "
        ].apply(lambda x: get_value(x, referral_reasons))
        self.mri_lesions_df["additional reason for referral "] = self.mri_lesions_df[
            "additional reason for referral "
        ].apply(lambda x: get_value(x, referral_reasons))

        # breast implants info
        self.mri_lesions_df["breast implants"] = self.mri_lesions_df[
            "breast implants"
        ].apply(lambda x: get_value(x, yes_no_mapping))

        # birads info
        self.mri_lesions_df["birads"] = self.mri_lesions_df["birads"].apply(
            lambda x: get_value(x, birads_assessment)
        )

        # tumor/benign infos
        for i in range(1, 7):
            self.mri_lesions_df[f"tumor/benign{i}"] = self.mri_lesions_df[
                f"tumor/benign{i}"
            ].apply(
                lambda x: get_value(int(x), tumor_benign_col) if x is not None else None
            )

        # positions for tumors
        for i in range(1, 7):
            self.mri_lesions_df[f"laterality lesion{i}"] = self.mri_lesions_df[
                f"pos{i}"
            ].apply(lambda x: get_value(x[0], laterality) if pd.notna(x) else None)
            self.mri_lesions_df[f"slice coord lesion{i}"] = self.mri_lesions_df[
                f"pos{i}"
            ].apply(lambda x: get_pos_coord_slice(x))
        self.mri_lesions_df = self.mri_lesions_df.drop(
            [f"pos{i}" for i in range(1, 7)], axis=1
        )

        # Create the pathology columns
        self.mri_lesions_df["benign lesions pathology"] = self.mri_lesions_df.apply(
            self.compose_lesion_pathology, axis=1
        )
        self.mri_lesions_df["malign lesions pathology"] = self.mri_lesions_df.apply(
            partial(self.compose_lesion_pathology, for_benign=False), axis=1
        )

        # pathology info
        for i in range(1, 7):
            self.mri_lesions_df[f"pathology{i}"] = self.mri_lesions_df[
                f"pathology{i}"
            ].apply(
                lambda x: (
                    get_value(int(x), tumor_pathology_values) if x is not None else None
                )
            )

        # grade info
        for i in range(1, 4):
            self.mri_lesions_df[f"grade{i}"] = self.mri_lesions_df[f"grade{i}"].apply(
                lambda x: get_value(x, grade_values) if x is not None else None
            )

        # Adjust SII info and KI67 percentage
        for i in range(1, 4):
            self.mri_lesions_df[f"er [sii] {i}"] = self.mri_lesions_df[
                f"er [sii] {i}"
            ].apply(lambda x: adjust_rsii(x))
            self.mri_lesions_df[f"pr [sii] {i}"] = self.mri_lesions_df[
                f"pr [sii] {i}"
            ].apply(lambda x: adjust_rsii(x))
            if i == 3:
                self.mri_lesions_df[f"her [sii] {i}"] = self.mri_lesions_df[
                    f"her [sii] {i}"
                ].apply(lambda x: adjust_rsii(x))
            else:
                self.mri_lesions_df[f"her2 [sii] {i}"] = self.mri_lesions_df[
                    f"her2 [sii] {i}"
                ].apply(lambda x: adjust_rsii(x))
            self.mri_lesions_df[f"ki67[%] {i}"] = self.mri_lesions_df[
                f"ki67[%] {i}"
            ].apply(lambda x: adjust_ki67(x))
