import logging
import os

import cv2
import numpy as np
import pydicom as dicom
import scipy
import SimpleITK as sitk
from scipy.signal import hilbert
from skimage.metrics import structural_similarity as ssim

from data_preprocessing.error_handling import log_func_info

DATA_FOLDER = "data"
RAW_DATA_FOLDER = "raw"
PROCESSED_DATA_FOLDER = "processed"
IMGS_FOLDER = "imgs"
CENTER_CROP_MAGIC_VALUE = 5


class BrokenImageError(RuntimeError):
    def __init__(self, message) -> None:
        super().__init__(message)


def convert_if_inverted(img: np.ndarray) -> np.ndarray:
    # Simple mechanism to check if the image is negative inverted
    if np.mean(img) > 127:
        return cv2.bitwise_not(img)
    return img


@log_func_info
def read_breast_image(img_path: str, check_inverted: bool = False) -> np.ndarray:
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

    # convert to grayscale if the image is not already
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)  # type: ignore
    return convert_if_inverted(img) if check_inverted else img


@log_func_info
def read_nii_gz_images(img_path: str, check_inverted: bool = False) -> np.ndarray:
    img = sitk.ReadImage(img_path, sitk.sitkFloat32)
    img_np = sitk.GetArrayFromImage(img)
    img_np = cv2.normalize(img_np, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)  # type: ignore

    for i in range(img_np.shape[0]):
        img_np[i] = convert_if_inverted(img_np[i]) if check_inverted else img_np[i]
    return img_np


@log_func_info
def read_mat_images(img_path: str) -> np.ndarray:
    mat = scipy.io.loadmat(img_path)
    res = mat["data"][0]
    return res


@log_func_info
def process_us_from_mat(img: np.ndarray, db_threshold: int = -50) -> np.ndarray:
    # per https://github.com/tensorflow/datasets/pull/2428/files
    envelope_im = np.abs(hilbert(img))  # type: ignore
    compress_im = 20 * np.log10(envelope_im / np.max(envelope_im))
    compress_im[compress_im < db_threshold] = db_threshold
    result = compress_im.astype("float32")
    result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)  # type: ignore
    return result


@log_func_info
def load_images_from_npy(npy_path: str) -> np.ndarray:
    assert npy_path.endswith(".npy"), f"Invalid file extension for {npy_path}"
    assert os.path.exists(npy_path), f"File {npy_path} not found"
    return np.load(npy_path)


@log_func_info
def resize_breast_image(img: np.ndarray, resize_value: tuple) -> np.ndarray:
    return cv2.resize(img, resize_value)


@log_func_info
def gamma_correction_breast_image(
    img: np.ndarray, gamma_value: float = 1.5
) -> np.ndarray:
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    inv_gamma = 1.0 / gamma_value
    table = np.array(
        [((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]
    ).astype(np.uint8)
    return cv2.LUT(img, table)


@log_func_info
def pad_to_largest_dim(image: np.ndarray) -> np.ndarray:
    h, w = image.shape
    biggest_side = max(h, w)

    adjusted_width = (biggest_side - w) // 2
    adjusted_height = (biggest_side - h) // 2

    image = cv2.copyMakeBorder(
        image,
        adjusted_height,
        adjusted_height,
        adjusted_width,
        adjusted_width,
        cv2.BORDER_CONSTANT,
    )
    return image


@log_func_info
def hist_equalization_breast_image(
    img: np.ndarray, clip_limit: float = 2.0, patch_size: tuple = (8, 8)
) -> np.ndarray:
    # Applies Contrast Limited Adaptive Histogram Equalization (CLAHE)
    # Check here: https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=patch_size)
    return clahe.apply(img)


@log_func_info
def center_crop_breast_image(img: np.ndarray, crop_values: tuple) -> np.ndarray:
    width, height = img.shape[1], img.shape[0]

    # process crop width and height for max available dimension
    crop_width = min(img.shape[1], crop_values[0])
    crop_height = min(img.shape[0], crop_values[1])

    mid_x, mid_y = int(width / 2), int(height / 2)
    cw2, ch2 = int(crop_width / 2), int(crop_height / 2)
    crop_img = img[mid_y - ch2 : mid_y + ch2, mid_x - cw2 : mid_x + cw2]

    if crop_img.shape[0] <= 0 or crop_img.shape[1] <= 0:
        raise BrokenImageError("The image center crop operation broke the image")

    return crop_img


@log_func_info
def save_breast_image(save_path: str, img: np.ndarray, dtype=None) -> str:
    save_path = f"{save_path}.png"
    if dtype:
        img = img.astype(dtype)
    if not cv2.imwrite(save_path, img):
        raise RuntimeError(f"Failed to save image: {save_path}")
    return save_path


@log_func_info
def save_images_as_npy(save_path: str, images: list, dtype=None) -> str:
    save_path = f"{save_path}.npy"
    if dtype:
        np.save(save_path, np.array(images, dtype=dtype))
    else:
        np.save(save_path, np.array(images))
    return save_path


@log_func_info
def crop_breast_image(img: np.ndarray) -> np.ndarray:
    positions = np.nonzero(img)
    top = int(positions[0].min())
    bottom = int(positions[0].max())
    left = int(positions[1].min())
    right = int(positions[1].max())
    result = img[top:bottom, left:right]

    if result.shape[0] <= 0 or result.shape[1] <= 0:
        raise BrokenImageError("The image crop operation broke the image")

    return result


@log_func_info
def remove_breast_artefacts(img: np.ndarray, connectivity: int = 4) -> np.ndarray:
    # Find connected components
    _, labels, stats, _ = cv2.connectedComponentsWithStats(
        img, connectivity=connectivity
    )

    # Find the label of the largest connected component (excluding background)
    largest_label = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1

    # Create a mask for the largest connected component
    largest_component_mask = np.uint8(labels == largest_label)

    # Apply the mask to the original grayscale image
    result = cv2.bitwise_and(src1=img, src2=img, dst=None, mask=largest_component_mask)  # type: ignore
    return result


@log_func_info
def convert_dcm_image(img_path: str, check_inverted: bool = False) -> np.ndarray:
    ds = dicom.dcmread(img_path)
    data = ds.pixel_array

    if data.ndim == 3:  # Check if the image has more than one channel
        data = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
        logging.warning(
            f"The DCM image {img_path} has more than one channel. Using the mean of all channels"
        )

    if data.max() == data.min():
        raise BrokenImageError(f"The DCM image {img_path} is empty")

    # Normalize the image for cv2
    data = cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)  # type: ignore

    return convert_if_inverted(data) if check_inverted else data


@log_func_info
def create_segmentation_mask_aux(mask: np.ndarray, rois):
    for roi in rois:
        if len(roi) != 0:
            if isinstance(roi[0], int):
                ymin, xmin, ymax, xmax = roi
                mask[ymin:ymax, xmin:xmax] = 255
            else:
                mask = create_segmentation_mask_aux(mask, roi)
    return mask


@log_func_info
def create_segmentation_mask(
    height: int, width: int, rois: tuple[tuple[int, int, int, int]]
) -> np.ndarray:
    # rois are tuples of (ymin, xmin, ymax, xmax)
    # Initialize a blank mask with zeros
    mask = np.zeros((height, width), dtype=np.uint8)
    return create_segmentation_mask_aux(mask, rois)


def estimate_noise_snr(image: np.ndarray) -> float:
    return float(np.mean(image) / np.std(image))


def estimate_psnr(og_image: np.ndarray, noisy_image: np.ndarray, epsilon: float = 1e-5):
    maxf = og_image.max()
    mse = np.sum((og_image - noisy_image) ** 2) / noisy_image.size
    return 20 * np.log10(maxf / (np.sqrt(mse) + epsilon))


def estimate_ssim(og_image: np.ndarray, n_image: np.ndarray):
    return ssim(og_image, n_image, data_range=n_image.max() - n_image.min())


def add_random_noise(img: np.ndarray, noise_intensity: int = 50, seed: int = 42):
    h, w = img.shape
    noisy_img = img.astype(np.float32)
    g = np.random.default_rng(seed=seed)
    noisy_img += noise_intensity * g.standard_normal((h, w))
    return noisy_img


def fast_denoising(noisy_image: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoising(noisy_image.astype(np.uint8))


def is_left_right_breast(img: np.ndarray) -> tuple[bool, bool]:
    half_width = img.shape[1] // 2
    left_half = img[:, :half_width]
    right_half = img[:, half_width:]
    l, r = np.sum(left_half), np.sum(right_half)
    if l > r:
        return True, False
    elif r > l:
        return False, True
    else:
        return False, False


def convert_to_left_right_comparison(text: str) -> tuple[bool, bool]:
    left_aliases = ["left", "left breast", "left side", "l"]
    right_aliases = ["right", "right breast", "right side", "r"]

    for alias in left_aliases:
        if alias in text.lower():
            return True, False
    for alias in right_aliases:
        if alias in text.lower():
            return False, True

    # If no alias is found, return False for both
    return False, False
