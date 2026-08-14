import glob
import logging
import os
from argparse import ArgumentParser

import pandas as pd

from data_preprocessing.preprocessing_configs import RESIZE_DIMS
from data_split.stratified_splitter import StratifiedSplitter

RG = "rg"
ALL_MODALITIES = ["mg", "tomo", "cesm", "mr", "us"]


def save_splits(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    val_df: pd.DataFrame,
    prefix_name: str,
    save_path: str,
) -> None:
    StratifiedSplitter.save_memmap(
        df=train_df,
        imgs_filename=os.path.join(
            save_path, f"{prefix_name}{StratifiedSplitter.TRAIN_SPLIT_BIN_EXT}"
        ),
        images_shape=RESIZE_DIMS,
    )
    train_df.to_csv(
        os.path.join(
            save_path, f"{prefix_name}{StratifiedSplitter.TRAIN_SPLIT_CSV_EXT}"
        ),
        index=False,
    )

    StratifiedSplitter.save_memmap(
        df=test_df,
        imgs_filename=os.path.join(
            save_path, f"{prefix_name}{StratifiedSplitter.TEST_SPLIT_BIN_EXT}"
        ),
        images_shape=RESIZE_DIMS,
    )
    test_df.to_csv(
        os.path.join(
            save_path, f"{prefix_name}{StratifiedSplitter.TEST_SPLIT_CSV_EXT}"
        ),
        index=False,
    )

    StratifiedSplitter.save_memmap(
        df=val_df,
        imgs_filename=os.path.join(
            save_path, f"{prefix_name}{StratifiedSplitter.VAL_SPLIT_BIN_EXT}"
        ),
        images_shape=RESIZE_DIMS,
    )
    val_df.to_csv(
        os.path.join(save_path, f"{prefix_name}{StratifiedSplitter.VAL_SPLIT_CSV_EXT}"),
        index=False,
    )


def split_and_save(
    df: pd.DataFrame,
    strat_cols: list[str],
    test_size: float,
    val_size: float,
    seed: int,
    save_path: str,
    hold_modality_out: bool = False,
) -> None:
    # Split
    logging.info("Splitting report generation data")
    train, test, val = StratifiedSplitter.split_data(
        df, test_size, val_size, seed, strat_cols
    )

    # Save
    if hold_modality_out:
        logging.info("Saving splits with modality held out.")
        for modality in ALL_MODALITIES:
            logging.info(f"helding modality: {modality}")

            curr_train = train[train["modality"] != modality].copy(deep=True)
            curr_test = test[test["modality"] != modality].copy(deep=True)
            curr_val = val[val["modality"] != modality].copy(deep=True)

            if len(curr_train) == 0 or len(curr_test) == 0 or len(curr_val) == 0:
                logging.warning(
                    f"Skipping held out modality modality {modality} due to empty splits"
                )
                continue

            prefix_name = f"all-{RG}-no-{modality}"
            save_splits(curr_train, curr_test, curr_val, prefix_name, save_path)

    logging.info("Saving splits for all modalities. No modality held out.")
    prefix_name = f"all-{RG}"
    save_splits(train, test, val, prefix_name, save_path)

    # Save per modality
    for modality in ALL_MODALITIES:
        logging.info(f"Splitting and saving for modality: {modality}")

        curr_train = train[train["modality"] == modality].copy(deep=True)
        curr_test = test[test["modality"] == modality].copy(deep=True)
        curr_val = val[val["modality"] == modality].copy(deep=True)

        if len(curr_train) == 0 or len(curr_test) == 0 or len(curr_val) == 0:
            logging.warning(f"Skipping modality {modality} due to empty splits")
            continue

        prefix_name = f"{modality}-{RG}"
        save_splits(curr_train, curr_test, curr_val, prefix_name, save_path)


if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("--processed_data_path", type=str, required=True)
    args.add_argument("--save_path", type=str, required=True)
    args.add_argument("--test_size", type=float, default=0.01)
    args.add_argument("--val_size", type=float, default=0.5)
    args.add_argument("--random_seed", type=int, default=42)
    args.add_argument("--debug", action="store_true", help="Logging debug messages")
    args.add_argument(
        "--hold_modality_out",
        action="store_true",
        help="Whether to hold out each modality when splitting",
    )
    args = args.parse_args()

    logging.basicConfig(
        filename="logs/report_generation_split.log",
        filemode="a",
        format="%(name)s - %(levelname)s - %(message)s",
        level=logging.DEBUG if args.debug else logging.INFO,
    )

    # Create the save path if it does not exist
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # get all the csv files in the processed data path
    csvs = glob.glob(os.path.join(args.processed_data_path, "**/*.csv"), recursive=True)

    # Load all the csv dataframes and concat them all together, ignoring the index
    logging.info("Loading all the csv files")
    agg_df = pd.concat([pd.read_csv(csv) for csv in csvs], ignore_index=True)

    # Filter by birads values which is the focus for RG
    logging.info("Filtering by birads values")
    rg_df = agg_df[
        (agg_df["birads"].notnull())
        & (agg_df["num_exam_imgs"] > 0)
        & (agg_df["current_report"].notnull() | agg_df["segmentations_path"].notnull())
    ].copy(
        deep=True
    )  # Adjustment to the filter: the report can be null, but the segmentation cannot in that case

    rg_df["patient"] = rg_df["patient"].astype(str)

    # Create an auxiliary dataframe with all images and duplicate information
    data_df = StratifiedSplitter.create_auxiliary_df(rg_df, False)

    # The nan reports become empty so we can still use modality and birads
    data_df["report"] = data_df["report"].apply(lambda x: "" if pd.isna(x) else x)

    split_and_save(
        df=data_df,
        strat_cols=["birads", "modality"],
        test_size=args.test_size,
        val_size=args.val_size,
        seed=args.random_seed,
        save_path=args.save_path,
        hold_modality_out=args.hold_modality_out,
    )
