import logging
from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from data_preprocessing.image_utils import load_images_from_npy


@dataclass(frozen=True)
class AuxiliarySplitInformation:
    id: str
    patient: str
    dataset: str
    modality: str
    birads: str
    race: str
    machine: str
    exam_type: str | None
    laterality: str | None
    image_path: str
    segmentation_path: str | None
    report: str | None
    full_report: str | None
    slice: int | None  # slice index for the saved images of the corresponding instance interest´


class StratifiedSplitter:
    TRAIN_SPLIT_BIN_EXT = "-train.bin"
    VAL_SPLIT_BIN_EXT = "-val.bin"
    TEST_SPLIT_BIN_EXT = "-test.bin"

    TRAIN_SPLIT_CSV_EXT = "-train.csv"
    VAL_SPLIT_CSV_EXT = "-val.csv"
    TEST_SPLIT_CSV_EXT = "-test.csv"

    @staticmethod
    def split_data_into_two(
        df: pd.DataFrame,
        test_size: float,
        val_size: float,
        seed: int,
        stratify_cols: list[str],
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        # Split
        logging.info("Splitting data")

        # Group by patient and collect associated rows
        grouped = df.groupby("patient")
        patient_groups = {patient: group for patient, group in grouped}

        # Assign a single representative stratify key per patient
        patient_stratify = {
            patient: tuple(group[stratify_cols].iloc[0])
            for patient, group in patient_groups.items()
        }

        # Prepare stratify data and patient list
        patients = list(patient_groups.keys())
        stratify_labels = [patient_stratify[patient] for patient in patients]

        def safe_train_test_split(patients, labels, test_size, random_state):
            stratify = labels
            while stratify:
                try:
                    return train_test_split(
                        patients,
                        test_size=test_size,
                        stratify=stratify,
                        random_state=random_state,
                    )
                except ValueError:
                    stratify = stratify[:-1]  # Remove the last stratification column
            return train_test_split(
                patients, test_size=test_size, random_state=random_state
            )

        # First split into train+val and test
        train_patients, test_patients = safe_train_test_split(
            patients, stratify_labels, test_size=test_size, random_state=seed
        )

        # Create final DataFrames
        train_df = pd.concat([patient_groups[patient] for patient in train_patients])
        test_df = pd.concat([patient_groups[patient] for patient in test_patients])

        return train_df, test_df

    @staticmethod
    def _safe_train_test_split(patients, labels, test_size, random_state):
        stratify = labels
        logging.info(f"Patients shape: {patients.shape}")
        while stratify:
            try:
                return train_test_split(
                    patients,
                    test_size=test_size,
                    stratify=patients[stratify],
                    random_state=random_state,
                )
            except ValueError as e:
                logging.warning(f"Stratification failed: {e}")
                stratify = stratify[:-1]  # Remove the last stratification column
        return train_test_split(
            patients, test_size=test_size, random_state=random_state
        )

    @staticmethod
    def _get_stratification_key(row, strat_cols: list[str]):
        return "_".join([str(row[col]) for col in strat_cols])

    @staticmethod
    def separate_singletons(
        df: pd.DataFrame, stratified_col: str
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        # Identify groups with only one sample:
        group_counts = df[stratified_col].value_counts()
        singleton_keys = group_counts[group_counts == 1].index

        # Separate out single-sample cases:
        singletons = df[df[stratified_col].isin(singleton_keys)]
        non_singletons = df[~df[stratified_col].isin(singleton_keys)]
        return singletons, non_singletons

    @staticmethod
    def split_data(
        df: pd.DataFrame,
        test_size: float,
        val_size: float,
        seed: int,
        stratify_cols: list[str],
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        logging.info("Splitting data")
        stratification_key_name = "stratification_key"

        df[stratification_key_name] = df.apply(
            lambda x: StratifiedSplitter._get_stratification_key(x, stratify_cols),
            axis=1,
        )
        unique_patients = df[["patient", stratification_key_name]].drop_duplicates()

        # Construct train set
        singletons, non_singletons = StratifiedSplitter.separate_singletons(
            unique_patients, stratification_key_name
        )
        unique_train, unique_temp = train_test_split(
            non_singletons,
            test_size=test_size,
            stratify=non_singletons[stratification_key_name],
            random_state=seed,
        )
        train_df = pd.concat([unique_train, singletons])

        # Construct test and val set
        singletons, non_singletons = StratifiedSplitter.separate_singletons(
            unique_temp, stratification_key_name
        )
        unique_val, unique_test = train_test_split(
            non_singletons,
            test_size=val_size,
            stratify=non_singletons[stratification_key_name],
            random_state=seed,
        )
        train_df = pd.concat([train_df, singletons])

        # Create final dataframes
        train_df = df[df["patient"].isin(train_df["patient"])]
        val_df = df[df["patient"].isin(unique_val["patient"])]
        test_df = df[df["patient"].isin(unique_test["patient"])]

        # Prevent patient leakage
        train_df = train_df.copy()
        val_df = val_df.copy()
        test_df = test_df.copy()
        train_df["split"] = "train"
        val_df["split"] = "val"
        test_df["split"] = "test"

        combined = pd.concat([train_df, val_df, test_df])

        # for each patient, finds the most common split, i.e. where the majority of the samples currently belong
        patient_split_counts = (
            combined.groupby(["patient", "split"]).size().unstack(fill_value=0)
        )
        majority_split = patient_split_counts.idxmax(axis=1)
        combined["final_split"] = combined["patient"].map(majority_split)

        # Prelimninary split moment
        final_train_df = combined[combined["final_split"] == "train"].copy()
        final_val_df = combined[combined["final_split"] == "val"].copy()
        final_test_df = combined[combined["final_split"] == "test"].copy()

        # go through each validation and test and check if there is no singleton per stratification key. If there is, move it to train.
        for split_name in ["val", "test"]:
            if split_name == "val":
                df = final_val_df
            else:
                df = final_test_df

            singletons, non_singletons = StratifiedSplitter.separate_singletons(
                df, stratification_key_name
            )
            if not singletons.empty:
                print(
                    f"Found {len(singletons)} singletons in {split_name} split, moving to train"
                )
                for _, row in singletons.iterrows():
                    to_move = df[df["patient"] == row["patient"]]
                    final_train_df = pd.concat(
                        [final_train_df, to_move], ignore_index=True
                    )
                    df = df[df["patient"] != row["patient"]]

            if split_name == "val":
                final_val_df = df
            else:
                final_test_df = df

        # Ensure that each split has at least one sample from each stratification key. If val or test split has no samples from a stratification key, move one random sample from that respective stratification key from train to val or test.
        unique_keys_in_train = final_train_df[stratification_key_name].unique()
        for split_name in ["val", "test"]:
            if split_name == "val":
                split_df = final_val_df
            else:
                split_df = final_test_df

            for key in unique_keys_in_train:
                if key not in split_df[stratification_key_name].values:
                    # choose one random sample from train with that key
                    sample = final_train_df[
                        final_train_df[stratification_key_name] == key
                    ].sample(n=1, random_state=seed)
                    sample_to_move = final_train_df[
                        final_train_df["patient"] == sample["patient"].values[0]
                    ]
                    split_df = pd.concat([split_df, sample_to_move], ignore_index=True)
                    final_train_df = final_train_df[
                        final_train_df["patient"] != sample["patient"].values[0]
                    ]

            if split_name == "val":
                final_val_df = split_df
            else:
                final_test_df = split_df

        # Ensure that for US and CESM, we have at least 100 samples for test and val splits. If not, move random samples from train to val or test until we reach 100 samples.
        for split_name in ["val", "test"]:
            if split_name == "val":
                split_df = final_val_df
            else:
                split_df = final_test_df

            for modality in ["cesm", "us"]:
                while split_df[split_df["modality"] == modality].shape[0] < 100:
                    sample = final_train_df[
                        final_train_df["modality"] == modality
                    ].sample(n=1, random_state=seed)
                    sample_to_move = final_train_df[
                        final_train_df["patient"] == sample["patient"].values[0]
                    ]
                    split_df = pd.concat([split_df, sample_to_move], ignore_index=True)
                    final_train_df = final_train_df[
                        final_train_df["patient"] != sample["patient"].values[0]
                    ]

            if split_name == "val":
                final_val_df = split_df
            else:
                final_test_df = split_df

        # save to a csv file the patient and the split to which it belongs
        final_train_df[["patient"]].to_csv("train_patients.csv", index=False)
        final_val_df[["patient"]].to_csv("val_patients.csv", index=False)
        final_test_df[["patient"]].to_csv("test_patients.csv", index=False)

        # Optionally drop helper columns if needed.
        final_train_df = final_train_df.drop(columns=["split", "final_split"])
        final_val_df = final_val_df.drop(columns=["split", "final_split"])
        final_test_df = final_test_df.drop(columns=["split", "final_split"])

        # Check for any patients that appear in multiple splits
        train_patients = set(final_train_df["patient"])
        val_patients = set(final_val_df["patient"])
        test_patients = set(final_test_df["patient"])
        common_patients = (
            train_patients.intersection(val_patients)
            .union(train_patients.intersection(test_patients))
            .union(val_patients.intersection(test_patients))
        )
        if common_patients:
            logging.warning(f"Patients in multiple splits: {common_patients}")

        # check if the straitify keys in test and val are present in train
        train_stratify_keys = set(final_train_df[stratification_key_name])
        val_stratify_keys = set(final_val_df[stratification_key_name])
        if not val_stratify_keys.issubset(train_stratify_keys):
            logging.warning(
                f"Val stratify keys not in train. Moving back to train: {val_stratify_keys - train_stratify_keys}"
            )
            # Move those keys back to train
            set_to_move = val_stratify_keys - train_stratify_keys
            for key in set_to_move:
                patients_to_move = final_val_df[
                    final_val_df[stratification_key_name] == key
                ]["patient"].unique()
                for patient in patients_to_move:
                    sample_to_move = final_val_df[final_val_df["patient"] == patient]
                    final_train_df = pd.concat(
                        [final_train_df, sample_to_move], ignore_index=True
                    )
                    final_val_df = final_val_df[final_val_df["patient"] != patient]

        train_stratify_keys = set(final_train_df[stratification_key_name])
        test_stratify_keys = set(final_test_df[stratification_key_name])
        if not test_stratify_keys.issubset(train_stratify_keys):
            logging.warning(
                f"Test stratify keys not in train. Moving back to train: {test_stratify_keys - train_stratify_keys}"
            )
            # Move those keys back to train
            set_to_move = test_stratify_keys - train_stratify_keys
            for key in set_to_move:
                patients_to_move = final_test_df[
                    final_test_df[stratification_key_name] == key
                ]["patient"].unique()
                for patient in patients_to_move:
                    sample_to_move = final_test_df[final_test_df["patient"] == patient]
                    final_train_df = pd.concat(
                        [final_train_df, sample_to_move], ignore_index=True
                    )
                    final_test_df = final_test_df[final_test_df["patient"] != patient]

        # Check that each patient appears in only one split:
        logging.info(
            f"Unique patients - Train: {final_train_df['patient'].nunique()}, Val: {final_val_df['patient'].nunique()}, Test: {final_test_df['patient'].nunique()}"
        )

        logging.info(
            f"Final split sizes - Train: {len(final_train_df)}, Val: {len(final_val_df)}, Test: {len(final_test_df)}"
        )

        # Check distribution of stratification columns
        logging.info("Train set distribution:")
        logging.info(Counter(final_train_df[stratification_key_name]))
        logging.info("Train set birads distribution:")
        logging.info(Counter(final_train_df["birads"]))
        logging.info("Train set modality distribution:")
        logging.info(Counter(final_train_df["modality"]))

        logging.info("Val set distribution:")
        logging.info(Counter(final_val_df[stratification_key_name]))
        logging.info("Val set birads distribution:")
        logging.info(Counter(final_val_df["birads"]))
        logging.info("Val set modality distribution:")
        logging.info(Counter(final_val_df["modality"]))

        logging.info("Test set distribution:")
        logging.info(Counter(final_test_df[stratification_key_name]))
        logging.info("Test set birads distribution:")
        logging.info(Counter(final_test_df["birads"]))
        logging.info("Test set modality distribution:")
        logging.info(Counter(final_test_df["modality"]))

        return final_train_df, final_test_df, final_val_df

    @staticmethod
    def save_memmap(
        df: pd.DataFrame,
        imgs_filename: str,
        images_shape: tuple,
        segs_filename: str | None = None,
    ) -> None:

        if len(df) == 0:
            logging.warning("Dataframe is empty, no data to save")
            return

        n_images = len(df)
        imgs_arr = np.memmap(
            imgs_filename,
            dtype=np.uint8,
            mode="w+",
            shape=(n_images, images_shape[0], images_shape[1]),
        )

        segs_arr = None
        if segs_filename is not None:
            segs_arr = np.memmap(
                segs_filename,
                dtype=np.uint8,
                mode="w+",
                shape=(n_images, images_shape[0], images_shape[1]),
            )

        index = 0

        for row in tqdm(
            df.itertuples(),
            total=n_images,
            desc="Saving images",
            unit="image",
            leave=False,
        ):
            try:
                imgs = load_images_from_npy(row.image_path)
                img = imgs[row.slice]
                imgs_arr[index] = img

                if segs_arr is not None:
                    segs = load_images_from_npy(row.segmentation_path)
                    seg = segs[row.slice]
                    segs_arr[index] = seg

                index += 1
            except Exception as e:
                logging.error(
                    f"Error saving image {row.image_path} on slice {row.slice}: {e}"
                )

        imgs_arr.flush()
        if segs_arr is not None:
            segs_arr.flush()

    @staticmethod
    def create_auxiliary_df(df: pd.DataFrame, is_segmentation: bool) -> pd.DataFrame:
        data = []
        for row in tqdm(
            df.itertuples(),
            total=df.shape[0],
            desc="Creating auxiliary dataframe",
            unit="row",
        ):
            if is_segmentation:
                iters = row.num_segmentations
            else:
                iters = row.num_exam_imgs

            for i in range(iters):
                data.append(
                    AuxiliarySplitInformation(
                        id=row.id,
                        patient=row.patient,
                        dataset=row.dataset,
                        modality=row.modality,
                        birads=row.birads,
                        race=row.race,
                        machine=row.machine,
                        exam_type=row.exam_type,
                        laterality=row.laterality,
                        image_path=row.exam_imgs,
                        segmentation_path=row.segmentations_path,
                        report=row.current_report,
                        full_report=row.full_report,
                        slice=i,
                    ).__dict__
                )

        return pd.DataFrame(data)
