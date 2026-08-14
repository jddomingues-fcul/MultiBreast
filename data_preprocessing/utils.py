import glob
import multiprocessing
import os
import re
from collections import Counter

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from data_preprocessing.error_handling import trycatch_func
from data_preprocessing.image_utils import read_breast_image
from data_preprocessing.preprocessing_configs import RESIZE_DIMS


class BasicDataset(Dataset):
    def __init__(self, imgs) -> None:
        self.imgs = imgs

    def __len__(self) -> int:
        return self.imgs.shape[0]

    def __getitem__(self, idx: int) -> tuple:
        img = self.imgs[idx]
        img = img / 255.0  # normalize
        mean = img.mean()
        std = img.std()
        return mean, std


def get_bin_training_set_mean_and_std(
    train_imgs: str, train_csv: str
) -> tuple[float, float]:
    """Get the training set mean and standard deviation for data store in binary format

    Args:
        train_imgs (str): train images. It has to be a binary dataset with a .bin extension
        train_csv (str): train csv. It has to be a binary dataset with a .bin extension

    Returns:
        tuple[float, float]: (mean, std) of the training dataset
    """

    df = pd.read_csv(train_csv)
    imgs = np.memmap(
        train_imgs,
        dtype=np.uint8,
        mode="r",
        shape=(len(df), RESIZE_DIMS[0], RESIZE_DIMS[1]),
    )
    dataset = BasicDataset(imgs)
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        pin_memory=True,
        num_workers=multiprocessing.cpu_count() - 1,
        persistent_workers=True,
    )

    mean = torch.zeros(1)
    std = torch.zeros(1)
    for im_mean, im_std in tqdm(dataloader):
        mean += im_mean.sum()
        std += im_std.sum()

    mean /= len(dataset)
    std /= len(dataset)
    return mean.item(), std.item()


# calculate mean and std of the training dataset
def get_mean_and_std(content: list[str]) -> tuple[float, float]:
    _sum = torch.zeros(1, dtype=torch.float32)
    sum_squared = torch.zeros(1, dtype=torch.float32)
    count = 0

    for im_path in content:
        data = torch.from_numpy(read_breast_image(im_path))
        _sum += torch.sum(data)
        sum_squared += torch.sum(data**2)
        count += data.numel()
        del data

    mean = _sum / count
    std = torch.sqrt(sum_squared / count - mean**2)
    return mean.item(), std.item()


def plot_counter_counts(litems: np.ndarray, hor=False):
    counts = Counter(litems).items()
    counts = sorted(counts)
    x, y = zip(*counts)

    if hor:
        plt.figure(figsize=(10, 10))
        plt.barh(x, y)
    else:
        plt.bar(x, y)


def check_spaced_out_exams(x):
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            hours = pd.Timedelta(x[j] - x[i]).seconds / 3600.0
            if hours >= 24:
                return True
    return False


@trycatch_func
def upper_to_lower_wspace(s):
    # This regex matches positions where an uppercase letter is followed by a lowercase letter
    # or where a lowercase letter is followed by an uppercase letter.
    return re.sub(r"([a-z])([A-Z])", r"\1 \2", s).lower()


@trycatch_func
def column_cleaning_csv_reading(csv_path: str, delimiter: str = ",") -> pd.DataFrame:
    df = pd.read_csv(csv_path, delimiter=delimiter)
    df.columns = csv_column_cleaning(list(df.columns))
    return df


@trycatch_func
def csv_column_cleaning(cols: list[str]) -> list[str]:
    cleaned_cols = [
        upper_to_lower_wspace(
            col.strip().replace("_", " ").replace("(", " ").replace(")", " ")
        )
        for col in cols
    ]

    # remove any duplicate spaces that may have been introduced
    cleaned_cols = [re.sub(r"\s+", " ", col) for col in cleaned_cols]
    cleaned_cols = [col.strip() for col in cleaned_cols]

    return cleaned_cols


def sanitize_string(s: str) -> str:
    s = s.strip()
    s = s.replace("_", " ")
    s = s.replace("(", " ")
    s = s.replace(")", " ")
    s = re.sub(r"\s+", " ", s)  # replace multiple spaces with a single space
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", s)
    s = s.lower()
    return s


@trycatch_func
def create_path_from_slice(
    slice_number: int, path: str, raw_path: str, img_extension: str
) -> str:
    # 999 is the maximum number of slices for an exam
    # this follows the example 1-001.dcm, 1-002.dcm, 1-003.dcm, etc
    available_slices = glob.glob(
        os.path.join(raw_path, path.replace("./", ""), f"**/*.{img_extension}"),
        recursive=True,
    )
    available_slices = sorted(
        [os.path.basename(slice_path) for slice_path in available_slices]
    )
    try:
        slice_file = available_slices[slice_number - 1]
    except IndexError:
        # remove the number of available slices from the slice number and get that value
        slice_number = len(available_slices) - (
            (len(available_slices) // slice_number) * slice_number
        )
        slice_file = available_slices[slice_number - 1]

    return os.path.join(raw_path, path.replace("./", ""), slice_file)


if __name__ == "__main__":
    split = "classification_split"
    modalities = ["all", "mg", "mr", "us", "cesm", "tomo"]
    for modality in modalities:
        mean, std = get_bin_training_set_mean_and_std(
            f"../data/{split}/{modality}-classification-train.bin",
            f"../data/{split}/{modality}-classification-train.csv",
        )

        print("For modality: ", modality)
        print(f"(mean, std): ({mean:.3f}, {std:.3f})")
        print("=====================================")

    print("Done!")
