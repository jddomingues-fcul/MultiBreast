import glob
import logging
import os

import cv2
import numpy as np

from data_preprocessing.error_handling import log_func_info, trycatch_func
from data_preprocessing.preprocessing_configs import ImagePreprocessingConfig


class ImageProcessor:
    IMGS_SUFFIX = "_imgs"
    SEGMENTATION_SUFFIX = "_segmentation"
    SLICES_SUFFIX = "_slices"

    SLICES_RANGE = 21  # Used to get the slices from the middle of an exam with multiple. Ideally we would save all the slices but for space constraints we save the middle ones only

    def __init__(
        self,
        raw_imgs_path: str,
        processed_imgs_path: str,
        image_preprocessing_config: ImagePreprocessingConfig,
    ):
        self.raw_imgs_path = raw_imgs_path
        self.processed_imgs_path = processed_imgs_path
        self.read_process = image_preprocessing_config.read_func
        self.save_process = image_preprocessing_config.save_func
        self.pipeline = image_preprocessing_config.processing_pipeline
        self.segmentation_pipeline = image_preprocessing_config.segmentation_pipeline

        # Create the processed imgs folder
        os.makedirs(self.processed_imgs_path, exist_ok=True)

    @trycatch_func
    @log_func_info
    def read_image(self, img_path: str):
        assert os.path.exists(img_path), f"Image {img_path} not found"
        res = None

        try:
            res = self.read_process(img_path)
        except Exception as e:
            logging.error(f"Error reading the image {img_path}: {e}")

        return res

    @trycatch_func
    @log_func_info
    def apply_processing(self, img: np.ndarray, is_segmentation: bool = False):
        res = img
        try:
            if is_segmentation:
                for processing_step in self.segmentation_pipeline:
                    res = processing_step(res)
            else:
                for processing_step in self.pipeline:
                    res = processing_step(res)
        except Exception as e:
            logging.error(f"Error processing image: {e}")
            logging.error("Returning the original image")

        return res

    @trycatch_func
    @log_func_info
    def process_image(self, img_path: str):
        assert os.path.exists(img_path), f"Image {img_path} not found"
        res = None

        try:
            res = self.read_process(img_path)
            for processing_step in self.pipeline:
                res = processing_step(res)
        except Exception as e:
            logging.error(f"Error processing image {img_path}: {e}")

        return res

    @trycatch_func
    @log_func_info
    def process_and_save_image(self, img_path: str, exam_id: str, patient_id: str):
        res = self.process_image(img_path)

        if res is None:
            return None

        save_path = os.path.join(
            self.processed_imgs_path, f"{patient_id}-{exam_id}{self.IMGS_SUFFIX}"
        )
        return self.save_process(save_path, [res])

    @trycatch_func
    @log_func_info
    def save_image_set(self, imgs: list, exam_id: str, patient_id: str):
        save_path = os.path.join(
            self.processed_imgs_path, f"{patient_id}-{exam_id}{self.IMGS_SUFFIX}"
        )
        return self.save_process(save_path, imgs)

    @trycatch_func
    @log_func_info
    def save_slices_set(self, imgs: list, exam_id: str, patient_id: str):
        save_path = os.path.join(
            self.processed_imgs_path, f"{patient_id}-{exam_id}{self.SLICES_SUFFIX}"
        )
        return self.save_process(save_path, imgs)

    @trycatch_func
    @log_func_info
    def save_segmentation_set(self, segs: list, exam_id: str, patient_id: str):
        save_path = os.path.join(
            self.processed_imgs_path,
            f"{patient_id}-{exam_id}{self.SEGMENTATION_SUFFIX}",
        )
        return self.save_process(save_path, segs)

    @trycatch_func
    @log_func_info
    def save_all_slices(
        self,
        folder_location: str,
        exam_id: str,
        patient_id: str,
        raw_imgs_extension: str | None = None,
    ):
        searchable_folder = os.path.join(self.raw_imgs_path, folder_location)
        if raw_imgs_extension is not None:
            available_imgs = sorted(
                glob.glob(
                    f"{searchable_folder}/**/*{raw_imgs_extension}", recursive=True
                )
            )
        else:
            available_imgs = sorted(glob.glob(f"{searchable_folder}/*", recursive=True))

        if len(available_imgs) == 0:
            return None, None

        middle_slice = len(available_imgs) // 2
        lower_slice = max(0, middle_slice - self.SLICES_RANGE)
        upper_slice = min(len(available_imgs), middle_slice + self.SLICES_RANGE + 1)
        available_imgs = available_imgs[lower_slice:upper_slice]
        slices = []
        i = 0

        acc = []
        img_shape = None

        for im in available_imgs:
            res = self.process_image(im)

            if res is None:
                continue

            if img_shape is None:
                img_shape = res.shape
            elif img_shape != res.shape:
                continue

            acc.append(res)
            slices.append(lower_slice + i)
            i += 1

        if len(acc) == 0:
            return None, None

        save_path = os.path.join(
            self.processed_imgs_path, f"{patient_id}-{exam_id}{self.SLICES_SUFFIX}"
        )
        return self.save_process(save_path, acc), tuple(slices)

    @trycatch_func
    @log_func_info
    def process_segmentation_mask(self, segmentation_path: str | None):
        if segmentation_path is None:
            return None

        if not os.path.exists(segmentation_path):
            return None

        seg = None

        try:
            seg = self.read_process(segmentation_path)
            for processing_step in self.segmentation_pipeline:
                seg = processing_step(seg)
            seg = seg > 0
            if seg is None or seg.max() == 0:  # type: ignore
                return None
        except Exception as e:
            logging.error(
                f"Error processing segmentation mask {segmentation_path}: {e}"
            )

        return seg

    @trycatch_func
    @log_func_info
    def save_segmentation_mask(
        self, segmentation_path: str | None, exam_id: str, patient_id: str
    ):
        seg = self.process_segmentation_mask(segmentation_path)
        if seg is None:
            return None

        save_path = os.path.join(
            self.processed_imgs_path,
            f"{patient_id}-{exam_id}{self.SEGMENTATION_SUFFIX}",
        )
        return self.save_process(save_path, [seg])

    def save_combined_segmentations(
        self, segmentation_search_pattern: str, exam_id: str, patient_id: str
    ):
        available_masks = glob.glob(
            os.path.join(self.raw_imgs_path, segmentation_search_pattern),
            recursive=True,
        )

        if len(available_masks) == 0:
            return None

        # Combine all the masks into one
        mask = None
        for mask_path in available_masks:
            curr_mask = self.process_segmentation_mask(mask_path)
            if mask is None and curr_mask is not None:
                mask = curr_mask
            elif curr_mask is not None:
                mask = mask + curr_mask

        if mask is not None and mask.max() > 0:
            segmentation_save_path = os.path.join(
                self.processed_imgs_path,
                f"{patient_id}-{exam_id}{self.SEGMENTATION_SUFFIX}",
            )
            return self.save_process(segmentation_save_path, [mask])

        return None

    @trycatch_func
    @log_func_info
    def save_slices(self, folder_location: str, exam_id: str, patient_id: str, *slices):
        """
        Save the slices of a folder of images and return the path to the saved images and the slices indexes
        Args:
            folder_location: Folder where the images are located
            exam_id: Exam id
            patient_id: Patient id
            *slices: 1-based indexes of the slices to save

        Returns: Tuple of the path to the saved images and the slices indexes (0-based)
        """

        searchable_folder = os.path.join(self.raw_imgs_path, folder_location)
        available_imgs = sorted(glob.glob(f"{searchable_folder}/*", recursive=True))

        if len(slices) == 0:
            return None, None

        res_imgs = []
        res_slices = []
        img_shape = None

        for s in slices:
            if s is None:
                continue

            slice_number = int(s - 1)
            try:
                curr_img = available_imgs[slice_number]
            except IndexError:
                # remove the number of available slices from the slice number and get that value
                slice_number = slice_number % len(available_imgs)
                curr_img = available_imgs[slice_number]

            curr_img = self.process_image(curr_img)
            if curr_img is None:
                continue

            if img_shape is None:
                img_shape = curr_img.shape
            elif img_shape != curr_img.shape:
                continue

            res_slices.append(slice_number)
            res_imgs.append(curr_img)

        save_path = os.path.join(
            self.processed_imgs_path, f"{patient_id}-{exam_id}{self.IMGS_SUFFIX}"
        )
        return self.save_process(save_path, res_imgs), tuple(res_slices)

    @staticmethod
    @trycatch_func
    @log_func_info
    def draw_polyline_on_image(
        image: np.ndarray, xpoints: list[int], ypoints: list[int]
    ) -> np.ndarray:
        pts = np.array(
            [[xpoints[i], ypoints[i]] for i in range(len(xpoints))], np.int32
        )
        pts = pts.reshape((-1, 1, 2))
        cv2.fillPoly(image, [pts], color=(255, 255, 255))
        return image

    @staticmethod
    @trycatch_func
    @log_func_info
    def draw_point_on_image(image: np.ndarray, cx: int, cy: int) -> np.ndarray:
        cv2.circle(image, (cx, cy), 1, (255, 255, 255), -1)
        return image

    @staticmethod
    @trycatch_func
    @log_func_info
    def draw_circle_on_image(image: np.ndarray, cx: int, cy: int, r: int) -> np.ndarray:
        cv2.circle(image, (cx, cy), r, (255, 255, 255), -1)
        return image

    @staticmethod
    @trycatch_func
    @log_func_info
    def draw_ellipse_on_image(
        image: np.ndarray, cx: int, cy: int, rx: int, ry: int
    ) -> np.ndarray:
        cv2.ellipse(image, (cx, cy), (rx, ry), 0, 0, 360, (255, 255, 255), -1)
        return image

    @staticmethod
    @trycatch_func
    @log_func_info
    def draw_polygon_on_image(
        image: np.ndarray, xpoints: list[int], ypoints: list[int]
    ) -> np.ndarray:
        pts = np.array(
            [[xpoints[i], ypoints[i]] for i in range(len(xpoints))], np.int32
        )
        pts = pts.reshape((-1, 1, 2))
        cv2.fillPoly(image, [pts], color=(255, 255, 255))
        return image

    @staticmethod
    @trycatch_func
    @log_func_info
    def draw_rectangle_on_image(
        image: np.ndarray, x1: int, y1: int, x2: int, y2: int
    ) -> np.ndarray:
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), -1)
        return image
