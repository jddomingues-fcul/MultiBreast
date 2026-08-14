import os
import uuid
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
import pypandoc
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
    laterality,
    modality,
)
from data_preprocessing.preprocessing_configs import CddCESMConfig
from data_preprocessing.utils import csv_column_cleaning


class CddCESM(BreastCancerDataset):
    def __init__(self, configs: CddCESMConfig):
        super().__init__(csv_save_path=configs.csv_save_path)
        self.all_df = pd.read_excel(configs.medical_excel_path, sheet_name="all")
        self.all_df.columns = csv_column_cleaning(list(self.all_df.columns))
        self.segmentations = pd.read_csv(configs.segmentations_path)
        self.folder_separation_by_type = {
            "cesm": configs.cesm_folder,
            "dm": configs.dm_folder,
        }

        self.configs = configs
        self.image_processor = ImageProcessor(
            raw_imgs_path=configs.raw_imgs_path,
            processed_imgs_path=configs.processed_imgs_path,
            image_preprocessing_config=configs.img_preprocessing_config,
        )

    def process_info(self):
        # Join all dataframes on the all_df using a left join on patient id and image name
        sheets = [
            "mass_description",
            "distortion",
            "postoperative",
            "postneoajuvant chemotherapy",
            "asymmetry",
            "calcifications",
            "mass enhancement_description",
            "nonmass enhancement_description",
        ]

        agg_df = self.join_dfs_on_all_df(
            self.configs.medical_excel_path,
            self.all_df,
            sheets,
            ["patient id", "image name"],
        )
        agg_df = agg_df.drop_duplicates()
        agg_df = agg_df.replace(np.nan, None)
        agg_df = agg_df.replace("_", None)

        # Remove unneded columns (repeated essentially)
        keywords_in_cols_to_remove = [
            "side:",
            "type:",
            "age:",
            "breast density acr :",
            "birads:",
            "findings:",
            "view:",
            "machine:",
            "pathology classification/ follow up:",
            "acr:",
        ]

        cols_to_remove = ["acr"]
        for col in agg_df.columns:
            for kw in keywords_in_cols_to_remove:
                if kw in col:
                    cols_to_remove.append(col)
                    break

        agg_df = agg_df.drop(columns=cols_to_remove)

        # NOTE: We do not use the actual clinical report as they consider multiple exams and we are only considering one. Also, we are focused on a more structured report (bullet points type)
        # cached_reports = {}
        # agg_df["report"] = agg_df.apply(lambda x: self.get_report(cached_reports, x["patient id"]), axis=1)

        # Replce columns with actual value
        machines = {
            "1": "GE Healthcare Senographe DS",
            "2": "Hologic Selenia Dimensions Mammography Systems",
        }

        # This is needed because the type of the image has influence on its folder of origin
        agg_df["type_folder"] = agg_df["type"].apply(
            lambda x: get_value(x, self.folder_separation_by_type)
        )
        agg_df["machine"] = agg_df["machine"].apply(lambda x: get_value(x, machines))
        agg_df["side"] = agg_df["side"].apply(lambda x: get_value(x, laterality))
        agg_df["type"] = agg_df["type"].apply(lambda x: get_value(x, modality))
        agg_df["birads"] = agg_df["birads"].apply(
            lambda x: get_value(x, birads_assessment)
        )
        agg_df["view"] = agg_df["view"].apply(lambda x: get_value(x, dview))
        agg_df["findings"] = agg_df["findings"].apply(lambda x: self.clean_dollar(x))
        agg_df["breast density acr"] = agg_df["breast density acr"].apply(
            lambda x: get_value(x, breast_density)
        )

        # Rename columns for better column description
        agg_df = agg_df.rename(
            columns={
                "view": "exam view",
                "tags": "medical tags",
                "pathology classification/ follow up": "pathology",
                "single/multiple": "issue on single or multiple breasts",
                "mass density/enhancement pattern": "mass density enhancement pattern",
                "single/multiple:mass_enhancement_description": "mass enhancement description for single or multiple breasts",
                "mass density/enhancement pattern:mass_enhancement_description": "mass enhancement density",
                "mass shape:mass_enhancement_description": "mass enhancement shape",
                "mass margin:mass_enhancement_description": "mass enhancement margin",
                "breast density acr": "breast density",
                "findings": "additional notes",
            }
        )

        # Process the exams
        n = cpu_count() - 1
        df_split = [
            agg_df.iloc[idx]
            for idx in np.array_split(np.arange(len(agg_df)), n)
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
            desc=f"Processing cdd-cesm batch {df.index[0]} to {df.index[-1]}",
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
        exam_imgs_path, segmentations_path = self.save_image_and_segmentation(
            row["image name"].strip(), row["type_folder"], exam_id, row["patient id"]
        )

        return ExamInformation(
            id=exam_id,
            patient=f"{self.get_dataset_name()}-{row['patient id']!s}",
            dataset=self.get_dataset_name(),
            modality=row["type"],
            birads=row["birads"],
            race=UNKNOWN,
            laterality=row["side"],
            exam_date=None,
            machine=row["machine"],
            exam_type=row["exam view"] + ", " + row["type_folder"],
            exam_imgs=exam_imgs_path,
            num_exam_imgs=1 if exam_imgs_path is not None else 0,
            segmentations_path=segmentations_path,
            num_segmentations=1 if segmentations_path is not None else 0,
            slices_imgs_path=None,
            num_slices_imgs=0,
            slices_index=None,
            current_report=self.create_report(row, self.get_current_report_cols()),
            full_report=self.create_report(row, self.get_all_report_cols()),
            previous_exams=None,
        )

    def save_image_and_segmentation(
        self, image_name, type_folder, exam_id, patient_id
    ) -> tuple:
        # get the image
        image_path = os.path.join(
            self.configs.raw_imgs_path,
            type_folder,
            f"{image_name}{self.configs.raw_imgs_extension}",
        )

        img = self.image_processor.process_image(image_path)
        if img is None:
            return None, None

        save_img_path = os.path.join(
            self.configs.processed_imgs_path,
            f"{patient_id}-{exam_id}{ImageProcessor.IMGS_SUFFIX}",
        )
        save_img_path = self.image_processor.save_process(save_img_path, [img])

        # get the segmentation
        seg = self.segmentations[self.segmentations["#filename"] == f"{image_name}.jpg"]
        seg_save_path = None
        if seg.empty:
            return save_img_path, seg_save_path

        img_shape = self.image_processor.read_process(image_path).shape
        mask = np.zeros((img_shape[0], img_shape[1]), dtype=np.uint8)

        for _, row in seg.iterrows():
            modifications = eval(row["region_shape_attributes"])
            if modifications["name"] == "polyline":
                xpoints = modifications["all_points_x"]
                ypoints = modifications["all_points_y"]
                mask = self.image_processor.draw_polyline_on_image(
                    mask, xpoints, ypoints
                )
            elif modifications["name"] == "point":
                cx, cy = modifications["cx"], modifications["cy"]
                mask = self.image_processor.draw_point_on_image(mask, cx, cy)
            elif modifications["name"] == "circle":
                cx, cy, r = modifications["cx"], modifications["cy"], modifications["r"]
                mask = self.image_processor.draw_circle_on_image(mask, cx, cy, r)
            elif modifications["name"] == "ellipse":
                cx, cy, rx, ry = (
                    modifications["cx"],
                    modifications["cy"],
                    modifications["rx"],
                    modifications["ry"],
                )
                mask = self.image_processor.draw_ellipse_on_image(mask, cx, cy, rx, ry)
            elif modifications["name"] == "polygon":
                xpoints = modifications["all_points_x"]
                ypoints = modifications["all_points_y"]
                mask = self.image_processor.draw_polygon_on_image(
                    mask, xpoints, ypoints
                )

        for process in self.image_processor.segmentation_pipeline:
            mask = process(mask)
        seg_save_path = os.path.join(
            self.configs.processed_imgs_path,
            f"{patient_id}-{exam_id}{ImageProcessor.SEGMENTATION_SUFFIX}",
        )
        seg_save_path = self.image_processor.save_process(seg_save_path, [mask])
        return save_img_path, seg_save_path

    # Join all dataframes on the all_df using a left join on patient id and image name
    @staticmethod
    def join_dfs_on_all_df(
        medical_data_path: str,
        agg_df: pd.DataFrame,
        shts: list[str],
        on_cols: list[str],
    ) -> pd.DataFrame:
        for sheet_name in shts:
            curr_df = pd.read_excel(medical_data_path, sheet_name=sheet_name)
            curr_df.columns = csv_column_cleaning(list(curr_df.columns))
            agg_df = pd.merge(
                agg_df,
                curr_df,
                on=on_cols,
                how="left",
                suffixes=("", f":{sheet_name.strip().replace(' ', '_')}"),
            )

        return agg_df

    def get_report(self, cache: dict, patient_id: int) -> str:
        if patient_id in cache:
            return cache[patient_id]

        report = pypandoc.convert_file(
            os.path.join(
                self.configs.raw_reports_path,
                f"P{patient_id}.{self.configs.reports_extension}",
            ),
            "plain",
        ).lower()
        report = "\n".join(
            report.split("\n")[2:]
        )  # to remove the id indication of the patient that is not useful
        cache[patient_id] = report
        return report

    @staticmethod
    def get_current_report_cols():
        return [
            "exam view",
            "breast density",
            "abnormality type",
            "mass shape",
            "mass margin",
            "mass density enhancement pattern",
            "mass enhancement density",
            "mass enhancement shape",
            "mass enhancement margin",
            "additional notes",
            "medical tags",
        ]

    @staticmethod
    def get_all_report_cols():
        return CddCESM.get_current_report_cols() + [
            "age",
            "side",
            "pathology",
            "issue on single or multiple breasts",
            "mass enhancement description for single or multiple breasts",
        ]

    @staticmethod
    def clean_dollar(text):
        return " and ".join(text.split("$"))
