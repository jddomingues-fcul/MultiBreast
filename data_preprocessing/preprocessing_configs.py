from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial

import numpy as np

from data_preprocessing.image_utils import (
    convert_dcm_image,
    pad_to_largest_dim,
    read_breast_image,
    read_mat_images,
    read_nii_gz_images,
    resize_breast_image,
    save_images_as_npy,
)

RESIZE_DIMS = (512, 512)


@dataclass(frozen=True)
class ImagePreprocessingConfig:
    read_func: Callable = partial(convert_dcm_image)
    save_func: Callable = partial(
        save_images_as_npy, dtype=np.uint8
    )  # images values range from 0 to 255 so we can save them as uint8
    processing_pipeline: list[Callable] = field(
        default_factory=lambda: [
            pad_to_largest_dim,
            partial(resize_breast_image, resize_value=RESIZE_DIMS),
        ]
    )
    segmentation_pipeline: list[Callable] = field(
        default_factory=lambda: [
            pad_to_largest_dim,
            partial(resize_breast_image, resize_value=RESIZE_DIMS),
            # segmentations need to be resized to the same size as the images. Other operations are not required
        ]
    )


@dataclass(frozen=True)
class AdvancedMRILesionsConfig:
    raw_imgs_path: str = "../data/raw/advanced-mri-breast-lesions"
    raw_imgs_extension: str = ".dcm"
    processed_imgs_path: str = "../data/processed_old/advanced-mri-breast-lesions/imgs"
    mri_lesions_path: str = "../data/raw/advanced-mri-breast-lesions/Advanced-MRI-Breast-Lesions-DA-Clinical-Jan112024.xlsx"
    metadata_path: str = "../data/raw/advanced-mri-breast-lesions/metadata.csv"
    image_modality: str = "mr"
    machine: str = "Phillips 1.5 T"
    csv_save_path: str = "../data/processed_old/advanced-mri-breast-lesions/advanced-mri-breast-lesions.csv"
    img_preprocessing_config: ImagePreprocessingConfig = ImagePreprocessingConfig()


@dataclass(frozen=True)
class BreastLesionUSGConfigs:
    raw_imgs_path: str = (
        "../data/raw/breast-lesions-usg/BrEaST-Lesions_USG-images_and_masks"
    )
    processed_imgs_path: str = "../data/processed_old/breast-lesions-usg/imgs"
    lesions_usg_path: str = "../data/raw/breast-lesions-usg/BrEaST-Lesions-USG-clinical-data-Dec-15-2023.xlsx"
    image_modality: str = "us"
    machine: str = """
    Hitachi ARIETTA 70 equipped with linear array transducer L441 (frequency range: 2–12 MHz);
    Esaote 6150 equipped with linear array transducer LA523 (frequency range: 4–13 MHz);
    Samsung RS85 equipped with linear array transducer L3–12A (frequency range: 3–12 MHz);
    Philips Affiniti 70 G and EPIQ 5 G equipped with linear array transducers eL18-4 (frequency range: 2–22 MHz) and L12-5 (frequency range: 5–12 MHz).
    """
    exam_type: str = "ultrasound"
    csv_save_path: str = (
        "../data/processed_old/breast-lesions-usg/breast-lesions-usg.csv"
    )
    img_preprocessing_config: ImagePreprocessingConfig = ImagePreprocessingConfig(
        read_func=partial(read_breast_image)
    )


@dataclass(frozen=True)
class CbisDDSMConfigs:
    raw_imgs_path: str = "../data/raw/cbis-ddsm/jpeg"
    raw_imgs_extension: str = ".jpg"
    processed_imgs_path: str = "../data/processed_old/cbis-ddsm/imgs"
    calc_case_description_train: str = (
        "../data/raw/cbis-ddsm/calc_case_description_train_set.csv"
    )
    calc_case_description_test: str = (
        "../data/raw/cbis-ddsm/calc_case_description_test_set.csv"
    )
    mass_case_description_train: str = (
        "../data/raw/cbis-ddsm/mass_case_description_train_set.csv"
    )
    mass_case_description_test: str = (
        "../data/raw/cbis-ddsm/mass_case_description_test_set.csv"
    )
    image_modality: str = "mg"
    machine: str = "DBA | HOWTEK | LUMYSIS"
    csv_save_path: str = "../data/processed_old/cbis-ddsm/cbis-ddsm.csv"
    img_preprocessing_config: ImagePreprocessingConfig = ImagePreprocessingConfig(
        read_func=partial(read_breast_image)
    )


@dataclass(frozen=True)
class CddCESMConfig:
    raw_imgs_path: str = "../data/raw/cdd-cesm/imgs"
    raw_imgs_extension: str = ".jpg"
    processed_imgs_path: str = "../data/processed_old/cdd-cesm/imgs"
    raw_reports_path: str = "../data/raw/cdd-cesm/Medical reports for cases"
    reports_extension: str = "docx"
    medical_excel_path: str = "../data/raw/cdd-cesm/Radiology-manual-annotations.xlsx"
    segmentations_path: str = (
        "../data/raw/cdd-cesm/Radiology_hand_drawn_segmentations_v2.csv"
    )
    cesm_folder: str = "Subtracted images of CDD-CESM"
    dm_folder: str = "Low energy images of CDD-CESM"
    csv_save_path: str = "../data/processed_old/cdd-cesm/cdd-cesm.csv"
    img_preprocessing_config: ImagePreprocessingConfig = ImagePreprocessingConfig(
        read_func=partial(read_breast_image)
    )


@dataclass(frozen=True)
class EmbedConfig:
    raw_imgs_path: str = "../data/raw/embed/imgs"
    raw_imgs_extension: str = ".png"
    processed_imgs_path: str = "../data/processed_old/embed/imgs"
    clinical_data_path: str = "../data/raw/embed/EMBED_OpenData_clinical.csv"
    metadata_path: str = "../data/raw/embed/EMBED_OpenData_metadata_reduced.csv"
    clinical_legend_path: str = "../data/raw/embed/AWS_Open_Data_Clinical_Legend.csv"
    imgs_size_path: str = "../data/raw/embed/image_sizes.csv"
    image_modality: str = "mg"
    csv_save_path: str = "../data/processed_old/embed/embed.csv"
    img_preprocessing_config: ImagePreprocessingConfig = ImagePreprocessingConfig(
        read_func=partial(read_breast_image)
    )


@dataclass(frozen=True)
class RsnaBCDConfig:
    raw_imgs_path: str = "../data/raw/rsna-bcd"
    raw_imgs_extension: str = ".dcm"
    processed_imgs_path: str = "../data/processed_old/rsna-bcd/imgs"
    train_csv_path: str = "../data/raw/rsna-bcd/train.csv"
    test_csv_path: str = "../data/raw/rsna-bcd/test.csv"
    image_modality: str = "mg"
    csv_save_path: str = "../data/processed_old/rsna-bcd/rsna-bcd.csv"
    img_preprocessing_config: ImagePreprocessingConfig = ImagePreprocessingConfig()


@dataclass(frozen=True)
class BreastMicroCalcConfig:
    raw_imgs_path: str = "../data/raw/breast-micro-calc/imgs"
    raw_imgs_extension: str = ".dcm"
    processed_imgs_path: str = "../data/processed_old/breast-micro-calc/imgs"
    description_path: str = "../data/raw/breast-micro-calc/Description.xlsx"
    image_modality: str = "mg"
    csv_save_path: str = "../data/processed_old/breast-micro-calc/breast-micro-calc.csv"
    img_preprocessing_config: ImagePreprocessingConfig = ImagePreprocessingConfig()


@dataclass(frozen=True)
class ChineseMGConfig:
    raw_imgs_path: str = "../data/raw/cmmd/imgs"
    raw_imgs_extension: str = ".dcm"
    processed_imgs_path: str = "../data/processed_old/cmmd/imgs"
    clinical_data_path: str = "../data/raw/cmmd/CMMD_clinicaldata_revision.xlsx"
    image_modality: str = "mg"
    machine: str = "GE Senographe DS mammography system"
    race: str = "asian"
    csv_save_path: str = "../data/processed_old/cmmd/cmmd.csv"
    img_preprocessing_config: ImagePreprocessingConfig = ImagePreprocessingConfig()


@dataclass(frozen=True)
class MamaMiaConfig:
    raw_imgs_path: str = "../data/raw/mama-mia/images"
    raw_segs_path: str = "../data/raw/mama-mia/segmentations/expert"
    raw_imgs_extension: str = ".nii.gz"
    processed_imgs_path: str = "../data/processed_old/mama-mia/imgs"
    clinical_data_path: str = "../data/raw/mama-mia/clinical_and_imaging_info.xlsx"
    image_modality: str = "mr"
    birads: str = "known biopsy proven"
    csv_save_path: str = "../data/processed_old/mama-mia/mama-mia.csv"
    img_preprocessing_config: ImagePreprocessingConfig = ImagePreprocessingConfig(
        read_func=partial(read_nii_gz_images)
    )


@dataclass(frozen=True)
class OasbudConfig:
    raw_data_path: str = "../data/raw/oasbud/OASBUD.mat"
    processed_imgs_path: str = "../data/processed_old/oasbud/imgs"
    image_modality: str = "us"
    csv_save_path: str = "../data/processed_old/oasbud/oasbud.csv"
    img_preprocessing_config: ImagePreprocessingConfig = ImagePreprocessingConfig(
        read_func=partial(read_mat_images)
    )  # NOTE: this case, is very specific and needs to be manually handled in the dataset class


@dataclass(frozen=True)
class InbreastConfig:
    raw_imgs_path: str = "../data/raw/inbreast/AllDICOMs"  # NOTE: there will not be more than 1 image per patient
    raw_imgs_extension: str = ".dcm"
    segmentations_xml_folder = "../data/raw/inbreast/AllXML"
    raw_segs_extension: str = ".xml"
    clinical_data_path: str = "../data/raw/inbreast/INbreast.xls"
    processed_imgs_path: str = "../data/processed_old/inbreast/imgs"
    image_modality: str = "mg"
    csv_save_path: str = "../data/processed_old/inbreast/inbreast.csv"
    img_preprocessing_config: ImagePreprocessingConfig = ImagePreprocessingConfig()


@dataclass(frozen=True)
class LABreastConfig:
    raw_imgs_path: str = "../data/raw/la-breast/imgs"
    raw_imgs_extension: str = ".tiff"
    processed_imgs_path: str = "../data/processed_old/la-breast/imgs"
    train_csv_path: str = "../data/raw/la-breast/train.csv"
    test_csv_path: str = "../data/raw/la-breast/test.csv"
    val_csv_path: str = "../data/raw/la-breast/val.csv"
    image_modality: str = "mr"
    race: str = "latin american"
    machine: str = "Multiple 1.5T scanners"
    csv_save_path: str = "../data/processed_old/la-breast/la-breast.csv"
    img_preprocessing_config: ImagePreprocessingConfig = ImagePreprocessingConfig(
        read_func=partial(read_breast_image)
    )
