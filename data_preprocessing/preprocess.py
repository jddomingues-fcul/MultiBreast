import logging
from argparse import ArgumentParser

from data_preprocessing.breast_cancer_dataset import BreastCancerDataset
from data_preprocessing.dto.advanced_mri_lesions import AdvancedMRILesions
from data_preprocessing.dto.breast_lesion_usg import BreastLesionUSG
from data_preprocessing.dto.breast_micro_calc import BreastMicroCalc
from data_preprocessing.dto.cbis_ddsm import CbisDDSM
from data_preprocessing.dto.cdd_cesm import CddCESM
from data_preprocessing.dto.cmmd import CMMD
from data_preprocessing.dto.embed import Embed
from data_preprocessing.dto.inbreast import Inbreast
from data_preprocessing.dto.la_breast import LABreast
from data_preprocessing.dto.mama_mia import MamaMia
from data_preprocessing.dto.oasbud import Oasbud
from data_preprocessing.dto.rsna_bcd import RsnaBCD
from data_preprocessing.preprocessing_configs import (
    AdvancedMRILesionsConfig,
    BreastLesionUSGConfigs,
    BreastMicroCalcConfig,
    CbisDDSMConfigs,
    CddCESMConfig,
    ChineseMGConfig,
    EmbedConfig,
    InbreastConfig,
    LABreastConfig,
    MamaMiaConfig,
    OasbudConfig,
    RsnaBCDConfig,
)


def main(dataset_name: str, dataset_class: BreastCancerDataset, configs):
    dc = dataset_class(configs)
    dc.set_dataset_name(dataset_name)
    dc.process_info()
    dc.save_csv()


if __name__ == "__main__":
    dataset_mapping = {
        "advanced-mri-breast-lesions": (AdvancedMRILesions, AdvancedMRILesionsConfig()),
        "breast-lesions-usg": (BreastLesionUSG, BreastLesionUSGConfigs()),
        "cbis-ddsm": (CbisDDSM, CbisDDSMConfigs()),
        "cdd-cesm": (CddCESM, CddCESMConfig()),
        "embed": (Embed, EmbedConfig()),
        "rsna-bcd": (RsnaBCD, RsnaBCDConfig()),
        "cmmd": (CMMD, ChineseMGConfig()),
        "breast-micro-calc": (BreastMicroCalc, BreastMicroCalcConfig()),
        "mama-mia": (MamaMia, MamaMiaConfig()),
        "oasbud": (Oasbud, OasbudConfig()),
        "inbreast": (Inbreast, InbreastConfig()),
        "la-breast": (LABreast, LABreastConfig()),
    }

    args = ArgumentParser()
    args.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset to process",
        choices=dataset_mapping.keys(),
    )
    args.add_argument("--debug", action="store_true", help="Logging debug messages")
    args = args.parse_args()

    logging.basicConfig(
        filename=f"logs/{args.dataset}_processing.log",
        filemode="a",
        format="%(name)s - %(levelname)s - %(message)s",
        level=logging.DEBUG if args.debug else logging.INFO,
    )

    datatset_class, dataset_configs = dataset_mapping[args.dataset]
    main(args.dataset, datatset_class, dataset_configs)
