import math
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
from data_preprocessing.image_utils import create_segmentation_mask, resize_breast_image
from data_preprocessing.medical_mappings import (
    UNKNOWN,
    birads_assessment,
    dview,
    get_value,
    get_value_default,
    laterality,
    race_mappings,
)
from data_preprocessing.preprocessing_configs import EmbedConfig
from data_preprocessing.utils import csv_column_cleaning


class Embed(BreastCancerDataset):
    def __init__(self, configs: EmbedConfig):

        super().__init__(csv_save_path=configs.csv_save_path)
        self.clinical_df = pd.read_csv(configs.clinical_data_path, low_memory=False)
        self.metadata_df = pd.read_csv(configs.metadata_path, low_memory=False)
        self.metadata_df.columns = csv_column_cleaning(list(self.metadata_df.columns))
        self.clinical_legend_df = pd.read_csv(configs.clinical_legend_path)
        self.imgs_size_df = pd.read_csv(
            configs.imgs_size_path,
            names=[
                "image path",
                "original width",
                "original height",
                "resized width",
                "resized height",
            ],
        )

        self.clinical_df = self.clinical_df.replace(np.nan, None)
        self.metadata_df = self.metadata_df.replace(np.nan, None)
        self.clinical_legend_df = self.clinical_legend_df.replace(np.nan, None)

        self.configs = configs
        self.image_processor = ImageProcessor(
            raw_imgs_path=configs.raw_imgs_path,
            processed_imgs_path=configs.processed_imgs_path,
            image_preprocessing_config=configs.img_preprocessing_config,
        )

    def process_info(self):
        # Get the column renaming so we can have better descriptions when creating the report
        # construct the legend dictionary to replace the values
        clinical_column_renaming = self.construct_column_renaming()
        legend_dict = self.construct_legend_dict()

        # replace the values and then the columns name with cleaning
        # manual adjustments do not get cleaned again
        manual_adjustments = [
            "tissueden",
            "her2",
            "tnmdesc",
            "stable",
            "new",
            "path_severity",
        ]
        self.clinical_df["tissueden"] = self.clinical_df["tissueden"].apply(
            lambda x: str(x).replace(".0", "") if pd.notna(x) else None
        )
        self.clinical_df["tissueden"] = self.clinical_df["tissueden"].apply(
            lambda x: legend_dict["tissueden"][str(x)] if pd.notna(x) else None
        )

        self.clinical_df["her2"] = self.clinical_df["her2"].apply(
            lambda x: str(x).replace(".0", "") if pd.notna(x) else None
        )
        self.clinical_df["her2"] = self.clinical_df["her2"].apply(
            lambda x: legend_dict["her2"][str(x)] if pd.notna(x) else None
        )

        self.clinical_df["tnmdesc"] = self.clinical_df["tnmdesc"].apply(
            lambda x: str(x).lower() if pd.notna(x) else None
        )
        self.clinical_df["tnmdesc"] = self.clinical_df["tnmdesc"].apply(
            lambda x: legend_dict["tnmdesc"][str(x)] if pd.notna(x) else None
        )

        self.clinical_df["stable"] = self.clinical_df["stable"].apply(
            lambda x: "yes" if x == 0 else "no"
        )  # edge case, -1 is no, 0 is yes

        self.clinical_df["new"] = self.clinical_df["new"].apply(
            lambda x: "yes" if x == 0 else "no"
        )

        self.clinical_df["path_severity"] = self.clinical_df["path_severity"].apply(
            lambda x: legend_dict["path_severity"][str(x)] if pd.notna(x) else None
        )  # "never biopsied"

        for header in legend_dict:
            if header in self.clinical_df.columns and header not in manual_adjustments:
                self.clinical_df[header] = self.clinical_df[header].apply(
                    lambda x: self.apply_legend_dict(x, header, legend_dict)
                )

        self.clinical_df = self.clinical_df.rename(columns=clinical_column_renaming)
        self.clinical_df.columns = csv_column_cleaning(list(self.clinical_df.columns))

        # Remove the columns that are not needed. Rationale on notebook
        clinical_cols_to_remove_context = [
            "unnamed: 0",
            "technologist identifier",
            "radiologist identifier",
            "biopsy site",
            "biopsy location",
            "location id for pathology lab.",
            "sent to external facility",
            "location of exam",
            "procedure code for the exam. same as sprocs",
            "surgery location id",
            "concordance",
            "tnm staging",
            "specimen number",
            "gender desc",
            "ethnic group desc",
            "marital status desc",
            "encounter qty",
            "cohort num",
            "path group",
            "first 3 zip",
            "special case type",
            "additional location",
            "distance in cm breast",
            "depth breast",
            "unique study identifier",
            "procedure date",
            "pathology report date",
        ]

        clinical_cols_to_remove_value = [
            "post op complications",
            "long term complications",
            "tnm residual",
            "focality",
            "number of focality",
            "specimen integrity",
            "specimen embedded",
        ]

        clinical_columns_to_remove = (
            clinical_cols_to_remove_context + clinical_cols_to_remove_value
        )
        self.clinical_df = self.clinical_df.drop(columns=clinical_columns_to_remove)

        # drop rows whose number of calcifcations, size, or distance are negative.
        self.clinical_df = self.clinical_df[
            self.clinical_df["number of calcifications"] >= 0
        ]
        self.clinical_df = self.clinical_df[self.clinical_df["size in mm"] >= 0]
        self.clinical_df = self.clinical_df[self.clinical_df["distance in cm"] >= 0]

        # Deal with age
        self.clinical_df["age at study"] = self.clinical_df["age at study"].apply(
            lambda x: math.floor(x) if pd.notna(x) else None
        )

        # Deal with ethnicity
        ethnicity_edge_cases_dict = {
            "Unknown, Unavailable or Unreported": None,
            "Not Recorded": None,
            "Patient Declines": None,
        }
        self.clinical_df["ethnicity desc"] = self.clinical_df["ethnicity desc"].apply(
            lambda x: ethnicity_edge_cases_dict.get(x, x) if pd.notna(x) else None
        )

        # Deal with metadata columns
        meta_columns_to_keep = [
            "image laterality final",
            "acc anon",
            "empi anon",
            "anon dicom path",
            "study description",
            "series description",
            "final image type",
            "roi coords",
            "view position",
            "manufacturer",
            "manufacturer model name",
            "protocol name",
        ]
        self.metadata_df = self.metadata_df[meta_columns_to_keep]

        # merging the dataframes
        # Join on the exam id (acc_anon), and then match the laterality of the findings
        all_info_df = pd.merge(
            self.metadata_df,
            self.clinical_df,
            on=["acc anon", "empi anon"],
            suffixes=(" metadata", " clinical"),
            how="inner",
        )
        all_info_df["image laterality final"] = all_info_df[
            "image laterality final"
        ].apply(lambda x: legend_dict["side"][x])

        # The "side" column in the clinical data represents the laterality of the finding in that row, and can be
        # L (left), R (right), B (bilateral), or NaN (when there is no finding). Therefore when merging clinical
        # and metadata, we must first match by exam ID and then match the laterality of the clinical finding (side)
        # to the laterality of the image (ImageLateralityFinal)/ Side "B" and "NaN" can be matched to
        # ImageLateralityFinal both "L" and "R"
        all_info_df = all_info_df.loc[
            (all_info_df["side"] == all_info_df["image laterality final"])
            | (all_info_df["side"] == "Both")
            | (all_info_df["side"].isnull())
        ]

        all_info_df = all_info_df.loc[
            all_info_df["series description"].notnull()
        ]  # the images that do not have a series description are not what we want (not in scope)

        # Add information about the image size
        all_info_df["image id"] = all_info_df["anon dicom path"].apply(
            lambda x: os.path.basename(x)[:-4]
        )
        self.imgs_size_df["image id"] = self.imgs_size_df["image path"].apply(
            lambda x: os.path.basename(x)[:-4]
        )
        all_info_df = pd.merge(
            all_info_df, self.imgs_size_df, on=["image id"], how="inner"
        )

        # To ensure consistency, keep the rows where:
        # 1. "number of calcifications" > 0 and others are NOT null
        # 2. "number of calcifications" == 0 and others ARE null
        all_info_df = all_info_df[
            (
                (all_info_df["number of calcifications"] > 0)
                & (~all_info_df["calcification distribution"].isnull())
                & (~all_info_df["calcification finding"].isnull())
            )
            | (
                (all_info_df["number of calcifications"] == 0)
                & (all_info_df["calcification distribution"].isnull())
                & (all_info_df["calcification finding"].isnull())
            )
        ]

        # 3. all mass characterizations are NOT null
        # 4. all mass characterizations ARE null
        all_info_df = all_info_df[
            (
                (~all_info_df["mass shape"].isnull())
                & (~all_info_df["mass margin"].isnull())
                & (~all_info_df["mass density"].isnull())
            )
            | (
                (all_info_df["mass shape"].isnull())
                & (all_info_df["mass margin"].isnull())
                & (all_info_df["mass density"].isnull())
            )
        ]

        # Adjust the view position to be more descriptive
        all_info_df["view position"] = all_info_df["view position"].apply(
            lambda x: get_value(x, dview)
        )

        # adjusted renaming
        all_info_df = all_info_df.rename(
            columns={
                "view position": "exam view",
            }
        )

        # Split DataFrame by patient id
        grouped_patients = [
            group.sort_values(by=["acc anon", "study date anon"])
            for _, group in all_info_df.groupby("empi anon")
        ]

        # Process the exams
        n = cpu_count() - 1
        with Pool(processes=n) as p:
            results = list(
                tqdm(
                    p.imap(self.process_patient, grouped_patients),
                    total=len(grouped_patients),
                    desc="Processing embed patients",
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
        # Use a buffer to control the previous report input
        curr_date = patient_df["study date anon"].iloc[0]
        prev_exams = []
        buff = []

        curr_exams = []

        for _, row in patient_df.iterrows():
            if row["study date anon"] != curr_date:
                prev_exams += buff
                buff.clear()
                curr_date = row["study date anon"]

            # only gets the previous report
            previous = self.get_previous_report(prev_exams)

            # generate the full report with previous info
            full_report_for_exam = self.create_report(
                row, self.get_all_report_cols(), previous
            ).lower()

            # update the buffer with the reports for the date
            exam_id = str(uuid.uuid4())
            buff.append(exam_id)

            race = get_value_default(row["ethnicity desc"], race_mappings)

            # get the image and the segmentation
            exam_imgs_path = self.image_processor.process_and_save_image(
                os.path.join(
                    self.configs.raw_imgs_path,
                    f"{row['image id']}{self.configs.raw_imgs_extension}",
                ),
                exam_id,
                str(row["empi anon"]),
            )

            segmentation_path = self.save_segmentation_from_rois(
                exam_id,
                str(row["empi anon"]),
                row["original height"],
                row["original width"],
                row["resized height"],
                row["resized width"],
                row["roi coords"],
            )

            exam = ExamInformation(
                id=exam_id,
                patient=f"{self.get_dataset_name()}-{row['empi anon']}",
                dataset=self.get_dataset_name(),
                modality=self.configs.image_modality,
                birads=get_value_default(row["assessment bi-rads"], birads_assessment),
                race=race if race is not None else UNKNOWN,
                laterality=get_value_default(row["image laterality final"], laterality),
                exam_date=row["study date anon"],
                machine=row["manufacturer"] + " " + row["manufacturer model name"],
                exam_type=row["description of the procedure"]
                + " "
                + row["series description"],
                exam_imgs=exam_imgs_path,
                num_exam_imgs=1 if exam_imgs_path is not None else 0,
                segmentations_path=segmentation_path,
                num_segmentations=1 if segmentation_path is not None else 0,
                slices_imgs_path=None,
                num_slices_imgs=0,
                slices_index=None,
                current_report=self.create_report(
                    row, self.get_current_report_cols(), None
                ),
                full_report=full_report_for_exam,
                previous_exams=tuple(prev_exams),
            )
            curr_exams.append(exam)

        return curr_exams

    @staticmethod
    def get_current_report_cols():
        return [
            "exam view",
            "number of calcifications",
            "calcification distribution",
            "calcification finding",
            "mass density",
            "mass margin",
            "mass shape",
            "consistent with",
            "path severity",
            "reccommendation",
        ]

    @staticmethod
    def get_all_report_cols():
        return Embed.get_current_report_cols() + [
            "description of the procedure",
            "image laterality final",
            "series description",
            "location",
            "side",
            "# of isolated cells",
            "# of macrometastases",
            "# of micrometastases",
            "# of nodes removed: sentinel",
            "# of nodes removed: total",
            "# of nodes removed: total positive",
            "age at study",
            "anterior margin in mm",
            "any change noted in the finding",
            "biopsy complications",
            "biopsy side",
            "biopsy technique",
            "biopsy type",
            "dcis size",
            "depth",
            "distance in cm",
            "eic",
            "estrogen",
            "estrogen percentage",
            "extracapsular extension",
            "ethnicity desc",
            "finding number",
            "fish/cish",
            "her2",
            "histological grade",
            "implant findings",
            "inferior margin in mm",
            "invasive size",
            "ki-67",
            "largest deposit in mm",
            "lateral margin in mm",
            "lymph node surgery type.",
            "medial margin in mm",
            "method of evaluation",
            "other related finding",
            "path1",
            "path10",
            "path2",
            "path3",
            "path4",
            "path5",
            "path6",
            "path7",
            "path8",
            "path9",
            "posterior margin in mm",
            "size in mm",
            "specimen size",
            "specsize2",
            "specsize3",
            "study date anon",
            "superior margin in mm",
            "surgery type",
            "tissue density",
            "tnm descriptors",
            "tnm distant metastasis",
            "tnm primary tumor",
            "tnm regional lymph nodes p n",
            "total l find",
            "total r find",
            "visit type",
            "whether finding is new",
            "whether finding is stable",
        ]

    def save_segmentation_from_rois(
        self,
        exam_id: str,
        patient_id: str,
        original_height: int,
        original_width: int,
        resized_height: int,
        resized_width: int,
        roi_coords: str,
    ):
        # roi coords are a string representation of a tuple
        tup = eval(roi_coords)

        if len(tup) == 0:
            return None

        mask = create_segmentation_mask(
            height=original_height, width=original_width, rois=tup
        )
        mask = resize_breast_image(
            mask, (resized_width, resized_height)
        )  # NOTE: Resized due to the image being resized at download time

        for process in self.image_processor.segmentation_pipeline:
            mask = process(mask)

        if mask.max() == 0:
            print("Segmentation is empty")
            return None

        seg_save_path = os.path.join(
            self.configs.processed_imgs_path,
            f"{patient_id}-{exam_id}{ImageProcessor.SEGMENTATION_SUFFIX}",
        )
        return self.image_processor.save_process(seg_save_path, [mask])

    @staticmethod
    def apply_legend_dict(val, header, legend_dict):
        assert header in legend_dict.keys(), (
            f"Header {header} not found in the legend dictionary"
        )

        if val is None:
            return None

        res = ""
        vals = list(filter(lambda x: x != "", str(val).split(",")))
        for item in vals:
            res += f"{legend_dict[header].get(item, '')},"  # In case the value is not found, then we append nothing

        # if the result is empty, i.e "" or ","
        if len(res) <= 1:
            return None

        return res[:-1]

    def construct_column_renaming(self) -> dict:
        clinical_column_renaming = self.clinical_legend_df.drop_duplicates(
            subset=["Header in export", "Discription"]
        )
        clinical_column_renaming = dict(
            clinical_column_renaming[["Header in export", "Discription"]].values
        )

        # Manual adjustments
        clinical_column_renaming["sprocs"] = "Procedure code for the exam"
        clinical_column_renaming["sdate_anon"] = "Unique study identifier"
        clinical_column_renaming["procdate_anon"] = "Procedure Date"
        clinical_column_renaming["pdate_anon"] = "Pathology report date"
        clinical_column_renaming["study_anon"] = "Exam date"
        clinical_column_renaming["stage"] = "TNM Staging"
        clinical_column_renaming["specembed"] = "Specimen embedded"
        clinical_column_renaming["bdistance"] = "Distance in cm (breast)"
        clinical_column_renaming["bdepth"] = "Depth (breast)"
        clinical_column_renaming["loc"] = "Additional location"
        clinical_column_renaming["her2"] = "her2"
        return clinical_column_renaming

    def construct_legend_dict(self) -> dict:
        legend_dict = dict()
        for _, row in self.clinical_legend_df.iterrows():
            if pd.isna(row["Code"]) or pd.isna(row["Meaning"]):
                continue

            header = row["Header in export"]
            code = row["Code"].strip()
            meaning = row["Meaning"].strip()

            # Check if header already exists in the map, if not, create an entry
            if header not in legend_dict:
                legend_dict[header] = dict()

            # Add code and meaning to the corresponding header in the map
            legend_dict[header][code] = meaning

        # Manual observation adjustment
        tnmpn_items = dict(legend_dict["tnmpn"].items())
        for k, v in tnmpn_items.items():
            if k == "PXT":
                legend_dict["tnmpn"]["PXT"] = v
            else:
                legend_dict["tnmpn"][k.replace("PT", "PN")] = v

        for i in range(1, 11):
            legend_dict[f"path{i}"] = legend_dict["path (1-10)"]

        # Obtained from the github page description
        # https://github.com/Emory-HITI/EMBED_Open_Data/tree/main
        legend_dict["path_severity"] = {
            "0.0": "invasive cancer",
            "1.0": "non-invasive cancer",
            "2.0": "high-risk lesion",
            "3.0": "borderline lesion",
            "4.0": "benign findings",
            "5.0": "negative (normal breast tissue)",
            "6.0": "non-breast cancer",
        }

        return legend_dict
