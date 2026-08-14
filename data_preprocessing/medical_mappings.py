from typing import Any

UNKNOWN = "unknown"


def get_value(value: Any, map_set: dict) -> str | None:
    if value is None:
        return None
    return map_set.get(str(value).strip().lower(), None)


def get_value_default(value: Any, map_set: dict) -> Any:
    if value is None:
        return None

    default_value = value.strip().lower() if isinstance(value, str) else value
    return map_set.get(str(value).strip().lower(), default_value)


def get_recurrence_score(x: Any) -> Any:
    if x is None:
        return None

    try:
        score = int(x)
        if score < 11:
            return "low"
        if 11 <= score <= 25:
            return "intermediate"
        if score > 25:
            return "high"

    except ValueError:
        return x


def adjust_rsii(x: Any) -> str | None:
    if x is None:
        return None

    if isinstance(x, int) or isinstance(x, float):
        if x == -1:
            return None

        return "negative" if x > 0.5 else "positive"

    return get_value(x, rsii)


def adjust_ki67(x: Any) -> str | None:
    if x is None:
        return None

    if isinstance(x, int):
        if x == -1:
            return None

        if x <= 15:
            return "low"
        elif 16 <= x <= 30:
            return "intermediate"

        return "high"

    return get_value(x, ki67)


def get_pos_coord_slice(x: Any) -> str | None:
    if x is None:
        return None

    if isinstance(x, str):
        return x[1:]
    return None


def get_oncotype_score(x: Any) -> Any:
    if x is None:
        return None

    try:
        score = int(x)
        if score < 11:
            return "low"
        if 11 <= score <= 25:
            return "intermediate"
        if score > 25:
            return "high"

    except ValueError:
        return x


birads_mapping = {
    0: "additional evaluation",
    1: "negative",
    2: "benign",
    3: "probably benign",
    4: "suspicious",
    5: "highly suggestive of malignancy",
    6: "known biopsy proven",
}

birads_mapping_with_unknown = {
    **birads_mapping,
    10: UNKNOWN,
}

birads_assessment_reverse = {v: k for k, v in birads_mapping.items()}

birads_assessment_reverse_with_unknown = {
    **birads_assessment_reverse,
    UNKNOWN: 10,
}

birads_assessment = {
    "**no assessment**": None,
    "0": "additional evaluation",
    "1": "negative",
    "normal": "negative",
    "negative": "negative",
    "2": "benign",
    "benign": "benign",
    "benign_without_callback": "benign",
    "3": "probably benign",
    "probably benign": "probably benign",
    "suspicious abnormality": "suspicious",
    "4": "suspicious",
    "4a": "suspicious",
    "4b": "suspicious",
    "4c": "suspicious",
    "malignant": "highly suggestive of malignancy",
    "highly suggestive of malignancy": "highly suggestive of malignancy",
    "5": "highly suggestive of malignancy",
    "6": "known biopsy proven",
    "known biopsy-proven malignancy": "known biopsy proven",
    "3, 4a": "suspicious",
}

modality = {
    "mr": "mr",
    "us": "us",
    "mg": "mg",  # digital mammography (dm) englobes mammography (mg), 2D and cview (cases from embed)
    "ct": "tomo",  # ea1141 cases
    "pt": "pt",
    "tomo": "tomo",
    "digital mammography": "mg",
    "contrast enhanced spectral mammography": "cesm",
    "dm": "mg",
    "cesm": "cesm",
    "dbt": "tomo",
}

modality_mapping = {"mr": 0, "us": 1, "mg": 2, "cesm": 3, "tomo": 4}
modality_mapping_reverse = {v: k for k, v in modality_mapping.items()}

breast_density = {
    "1": "the breasts are almost entirely fat",
    "a": "the breasts are almost entirely fat",
    "2": "scattered fibroglandular densities",
    "b": "scattered fibroglandular densities",
    "3": "heterogeneously dense",
    "c": "heterogeneously dense",
    "4": "extremely dense",
    "d": "extremely dense",
    "0": "normal",
}

# rename laterality and diagnose_view
laterality = {
    "r": "right",
    "l": "left",
    "right": "right",
    "left": "left",
}

# sources: https://dicom.nema.org/medical/dicom/2023c/output/chtml/part16/sect_CID_4015.html
dview = {
    "cc": "cranial caudal",
    "mlo": "mediolateral oblique",
    "ml": "mediolateral",
    "lm": "lateromedial",
    "lmo": "lateromedial oblique",
    "at": "axillary tail",
    "xxcl": "laterally exaggerated cranial caudal",
    "xccm": "medially exaggerated cranial caudal",
    "cv": "cleavage",
    "mloid": "mediolateral oblique implant displaced",
    "ccid": "craniocaudal implant displaced",
    "rl": "rolled laterally",
    "rm": "rolled medially",
    "mlid": "mediolateral implant displaced",
    "tan": "tangential",
    "sio": "superior inferior oblique",
    "fb": "from below",
    "iso": "inferior superior oblique",
    "lmid": "lateromedial implant displaced",
}

ki67 = {
    "high prolif": "high proliferation rate",
    "intermed prolif": "intermediate proliferation",
    "low prolif": "low proliferation",
    "pos (high prolif rate)": "positive (high proliferation rate)",
    "pos (strong)": "positive (strong proliferation rate)",
    "pos hi prolif (49% nucs)": "positive high proliferation rate (49% nucs)",
    "intermed prolif rate": "intermediate proliferation rate",
    "high": "high",
    "low": "low",
    "intermediate": "intermediate",
    "low to intermediate": "low to intermediate",
    "intermediate to high": "intermediate to high",
    "high to intermediate": "intermediate to high",
    "60 to 70": "high",
    "30 to 40": "high",
    "15 to 20": "low to intermediate",
    "10 to 20": "low to intermediate",
    "3 to 5": "low",
    "2 to 3": "low",
    "5 to 10": "low",
    "20 to 25": "intermediate",
    "6 to 9": "low",
    "50 to 60": "high",
}

rsii = {
    "pos": "positive",
    "positive": "positive",
    "neg": "negative",
    "neg (stain moderate)": "stain moderate negative",
    "pos (strong)": "strong positive",
    "pos (strongly)": "strong positive",
    "weak": "weak positive",
    "weak positive": "weak positive",
    "pos (weak)": "weak positive",
    "neg (weak)": "weak negative",
    "pos (2+)": "positive 2+",
    "moderate by fish": "moderate by fish (fluorescence in situ hybridization)",
    "strong": "strong positive",
    "strong positive": "strong positive",
    "moderate to strong": "moderate to strong positive",
    "moderate to strong positive": "moderate to strong positive",
    "moderate": "moderate positive",
    "moderate positive": "moderate positive",
    "intermediate": "moderate positive",
    "weak to moderate": "weak to moderate positive",
    "negative": "negative",
}

referral_reasons = {
    "1": "assessment of extent of disease (known tumor/s)",
    "2": "high risk follow up - family history",
    "3": "high risk follow up - previous breast cancer",
    "4": "high risk follow up - brca",
    "5": "investigation of lesion previously seen in mammography / us / self-exam",
    "6": "post treatment - response to therapy assessment (nat)",
    "only 6 min delay": "only 6 min delay",
    "severe motion in delayed series": "severe motion in delayed series",
}

yes_no_mapping = {"0": "no", "1": "yes", "x": "yes", "y": "yes", "n": "no"}

tumor_benign_col = {
    "1": "tumor determined from biopsy/surgery pathological results",
    "0": "benign determined from biopsy, or there was a followup of at least 1 year",
}

tumor_pathology_values = {
    # malign
    "1": "idc - invasive ductal carcinoma",
    "2": "ilc- infiltrating lobular carcinoma",
    "3": "idc+dcis",
    "4": "ilc+lcis",
    "5": "carcinoma - type unspecified",
    "6": "dcis - ductal carcinoma in situ (high risk)",
    "7": "lcis (high risk)",
    "8": "adh/alh - atypical ductal/lobular hyperplasia (high risk)",
    "9": "intraductal papillary lesion (high risk)",
    "10": "metaplastic carcinoma",
    "24": "liposarcoma",
    "25": "radial scar (high risk)",
    # benign
    "11": "fibroadenoma",
    "12": "fcc - fibrocystic/fibroadenomatic changes",
    "13": "inflamation",
    "14": "healing fat necrosis",
    "15": "not seen in imri guided biopsy",  # NOTE: Ultimately unused because we are focusing on confirmed benign lesions
    "16": "hemangioma",
    "17": "clinically benign",  # "not sent to biopsy with confirmation 1 year follow up", => NOTE: THIS IS THE ORIGINAL VALUE
    "18": "inflammatory cyst",
    "19": "fibrotic breast tissue",
    "20": "reaction to foreign body",
    "21": "lactational changes",
    "22": "breast tissue without significant changes",
    "23": "ductal hyperplasia (usual)",
}

grade_values = {
    "1": "grade 1 or low grade (in cases such as dcis)",
    "2": "grade 2 or intermediate grade (in cases such as dcis)",
    "3": "grade 3 or high grade (in cases such as dcis)",
    "1 to 2": "1 to 2",
    "2 to 3": "2 to 3",
    "2 to3": "2 to 3",
}

pos_neg_mapping = {"0": "negative", "1": "positive"}

mammaprint_70_gene_risk = {"0": "low risk", "1": "high risk"}

race_mappings = {
    "caucasian or white": "white",
    "caucasian": "white",
    "african american": "black",
    "native american": "american indian",
    "african american  or black": "black",
    "native hawaiian or other pacific islander": "native hawaiian",
    "american indian or alaskan native": "american indian",
    "black/african american": "black",
    "american indian/alaskan native": "american indian",
    "multiple races reported": "multiple",
    "amer indian": "american indian",
    "american indian": "american indian",
    "hawaian": "native hawaiian",
    "native hawaiian/pacific islander": "native hawaiian",
    "hawaiian/pacific islander": "native hawaiian",
    "multi": "multiple",
    "hispanic": "hispanic",
    "multiple race": "multiple",
    "hawa": "native hawaiian",
    "asian": "asian",
}
