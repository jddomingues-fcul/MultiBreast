from models.amber.amber import Amber
from models.amber_cls.amber_cls import AmberCLS
from models.amber_no_modality_text.amber_no_modality import AmberNoModality
from models.amber_no_projector.amber_no_projector import AmberNoProjector


def get_model_class_by_name(class_name: str):
    try:
        return {
            "amber": Amber,
            "amber_no_modality_text": AmberNoModality,
            "amber_no_projector": AmberNoProjector,
            "amber_cls": AmberCLS,
        }[class_name]
    except KeyError:
        raise ValueError(f"Model class '{class_name}' is not recognized.")
