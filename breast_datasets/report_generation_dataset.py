import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from data_preprocessing.medical_mappings import birads_assessment_reverse


class ReportGenerationDataset(Dataset):
    def __init__(
        self,
        imgs_path: str,
        csv_path: str,
        imgs_shape: tuple,
        transform=None,
        return_birads=False,
        return_modalities=False,
        return_exam_type=False,
        return_origin_dataset=False,
        return_patient_id=False,
        return_exam_ids=False,
    ):
        self.data = pd.read_csv(csv_path, low_memory=False)
        self.imgs_path = imgs_path
        self.imgs_data_shape = (len(self.data), imgs_shape[0], imgs_shape[1])
        self.transform = transform
        self.return_birads = return_birads
        self.return_modalities = return_modalities
        self.return_exam_type = return_exam_type
        self.return_origin_dataset = return_origin_dataset
        self.return_patient_id = return_patient_id
        self.return_exam_ids = return_exam_ids
        self.birads_class_ratios = None
        self.modalities_class_ratios = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        imgs = np.memmap(
            self.imgs_path, dtype=np.uint8, mode="r", shape=self.imgs_data_shape
        )
        image = (imgs[idx]).astype(np.uint8)
        report = self.data.iloc[idx]["report"]

        # if there is no report, we set it to an empty string
        if pd.isna(report):
            report = ""

        image = torch.from_numpy(image)
        if self.transform:
            image = self.transform(image)

        result = [image, report]

        if self.return_birads:
            birads = self.data.iloc[idx]["birads"]
            birads = birads_assessment_reverse[birads]
            birads = torch.tensor(birads)
            result.append(birads)

        if self.return_modalities:
            modality = self.data.iloc[idx]["modality"]
            result.append(modality)

        if self.return_exam_type:
            exam_type = self.data.iloc[idx]["exam_type"]
            result.append(exam_type)

        if self.return_origin_dataset:
            origin_dataset = self.data.iloc[idx]["dataset"]
            result.append(origin_dataset)

        if self.return_patient_id:
            patient_id = self.data.iloc[idx]["patient"]
            result.append(patient_id)

        if self.return_exam_ids:
            exam_id = self.data.iloc[idx]["id"]
            result.append(exam_id)

        return tuple(result)

    def make_weights_for_weighted_loss(self):
        res = self.data.groupby(by=["birads", "modality"]).count()
        self.class_counts = dict()

        for i in range(len(res)):
            g_name = "-".join(res.iloc[i].name)
            self.class_counts[g_name] = res.iloc[i].id

        classes_weights = {}
        for k, v in self.class_counts.items():
            classes_weights[k] = len(self.data) / (v * len(self.class_counts))

        return classes_weights

    def make_weights_for_weighted_sampler(self):
        res = self.data.groupby(by=["birads", "modality"]).count()
        self.class_counts = {}

        for i in range(len(res)):
            g_name = "-".join(res.iloc[i].name)
            self.class_counts[g_name] = res.iloc[i].id

        samples_weights = [
            1.0
            / self.class_counts[
                "-".join(
                    [self.data.iloc[idx]["birads"], self.data.iloc[idx]["modality"]]
                )
            ]
            for idx in range(len(self.data))
        ]

        return samples_weights

    def get_class_labels(self) -> list:
        self.data["grouped_label"] = self.data["birads"] + "-" + self.data["modality"]
        return self.data["grouped_label"].values.tolist()
