import uuid
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count

import numpy as np
from tqdm import tqdm

from data_preprocessing.breast_cancer_dataset import (
    BreastCancerDataset,
    ExamInformation,
)
from data_preprocessing.error_handling import trycatch_func
from data_preprocessing.image_processor import ImageProcessor
from data_preprocessing.image_utils import process_us_from_mat
from data_preprocessing.medical_mappings import UNKNOWN, birads_assessment, get_value
from data_preprocessing.preprocessing_configs import OasbudConfig


@dataclass(frozen=True)
class USDataHolder:
    patient_id: str
    img: np.ndarray
    segmentation: np.ndarray
    birads: str


class Oasbud(BreastCancerDataset):
    def __init__(self, configs: OasbudConfig):
        super().__init__(csv_save_path=configs.csv_save_path)
        self.configs = configs
        self.image_processor = ImageProcessor(
            raw_imgs_path="",
            processed_imgs_path=configs.processed_imgs_path,
            image_preprocessing_config=configs.img_preprocessing_config,
        )

        self.source_data = self.image_processor.read_image(configs.raw_data_path)

    def process_info(self):
        acc_data: list[USDataHolder] = []
        assert self.source_data is not None, "No data found in the dataset"

        for i, exam_set in tqdm(
            enumerate(self.source_data), desc="Processing examples oasbud"
        ):
            _, us1, us2, seg1, seg2, birads, _ = exam_set
            birads = birads[0]
            us1, us2 = process_us_from_mat(us1), process_us_from_mat(us2)
            birads = get_value(birads, birads_assessment)
            if birads is None:
                continue

            acc_data.append(
                USDataHolder(
                    patient_id=str(i),
                    img=us1,
                    segmentation=seg1,
                    birads=birads,
                )
            )
            acc_data.append(
                USDataHolder(
                    patient_id=str(i),
                    img=us2,
                    segmentation=seg2,
                    birads=birads,
                )
            )

        # Process the exams
        n = cpu_count() - 1
        with Pool(processes=n) as p:
            results = p.map(self.process_example, acc_data)

        for result in results:
            self.append_exam(result)

    @trycatch_func
    def process_example(self, example: USDataHolder):
        # get needed info for exam
        exam_id = str(uuid.uuid4())
        patient_id = f"{self.get_dataset_name()}-{example.patient_id}"

        exam, seg = (
            self.image_processor.apply_processing(example.img, False),
            self.image_processor.apply_processing(example.segmentation, True),
        )
        exam_path = self.image_processor.save_image_set([exam], exam_id, patient_id)
        segmentation_path = self.image_processor.save_segmentation_set(
            [seg], exam_id, patient_id
        )

        return ExamInformation(
            id=exam_id,
            patient=patient_id,
            dataset=self.get_dataset_name(),
            modality=self.configs.image_modality,
            birads=example.birads,
            race=UNKNOWN,
            laterality=None,
            exam_date=None,
            machine=None,
            exam_type=self.configs.image_modality,  # NOTE: For simplicity we are using the same modality for exam type
            exam_imgs=exam_path,
            num_exam_imgs=1 if exam_path is not None else 0,
            segmentations_path=segmentation_path,
            num_segmentations=1 if segmentation_path is not None else 0,
            slices_imgs_path=None,  # slices only on continuous images (us and mri video). Here we have singular images that stay at the slices
            num_slices_imgs=0,
            slices_index=None,  # no slices so no indexes
            current_report="",
            full_report="",
            previous_exams=None,
        )
