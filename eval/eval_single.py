import json
import logging
import multiprocessing
import os
from argparse import ArgumentParser
from contextlib import nullcontext

import numpy as np
import pandas as pd
import torch
import torchvision.transforms.v2 as T
import wandb
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from breast_datasets.report_generation_dataset import ReportGenerationDataset
from components.plotting.plotting import plot_confidence_interval_table
from components.processing.misc import div_255, repeat_rgb_channels
from data_preprocessing.medical_mappings import modality_mapping
from eval.coverage_evaluator import CoverageEvaluator
from eval.misc import (
    MetricsHolder,
    MetricsListHolder,
    SeedModelResults,
    get_content_from_predictions,
    setup_seed_models,
)
from eval.report_generation_metrics import compute_cls_metrics
from models import get_model_class_by_name


def compute_modality_f1(idxs, predicted_modality, target_modality) -> float:
    metrics_dict = compute_cls_metrics(
        list(predicted_modality[idxs]),
        list(target_modality[idxs]),
        features_approach="macro",
    )
    return metrics_dict["f1 score"]


def compute_birads_f1(idxs, predicted_birads, target_birads) -> float:
    metrics_dict = compute_cls_metrics(
        list(predicted_birads[idxs]),
        list(target_birads[idxs]),
        features_approach="macro",
    )
    return metrics_dict["f1 score"]


def compute_rg_metrics_with_weights(idxs, findings_metrics) -> tuple[float, float]:
    bleus4 = [findings_metrics[idx].get("bleu4", 0.0) for idx in idxs]
    meteors = [findings_metrics[idx].get("meteor", 0.0) for idx in idxs]
    return sum(bleus4) / len(bleus4) if bleus4 else 0.0, sum(meteors) / len(
        meteors
    ) if meteors else 0.0


def compute_findings_coverage(
    idxs, predicted_findings, target_findings
) -> tuple[float, float]:
    cov, eqs = CoverageEvaluator.construct_findings_coverage(
        predicted_reports=predicted_findings[idxs], gt_reports=target_findings[idxs]
    )
    return sum(cov) / len(cov) if cov else 0.0, sum(eqs) / len(eqs) if eqs else 0.0


def compute_model_performance(
    pats,
    pats_to_idx,
    predicted_modality,
    target_modality,
    predicted_birads,
    target_birads,
    predicted_findings,
    target_findings,
    findings_metrics,
) -> MetricsHolder:

    chosen_indexes = []
    for _ in range(len(pats)):
        curr_pat = rng.choice(pats)
        chosen_indexes.extend(pats_to_idx[curr_pat])

    modality_f1 = compute_modality_f1(
        chosen_indexes, predicted_modality, target_modality
    )
    birads_f1 = compute_birads_f1(chosen_indexes, predicted_birads, target_birads)
    findings_bleu4, findings_meteor = compute_rg_metrics_with_weights(
        chosen_indexes, findings_metrics
    )
    findings_coverage, findings_coverage_equality = compute_findings_coverage(
        chosen_indexes, predicted_findings, target_findings
    )

    return MetricsHolder(
        modality_f1=modality_f1,
        birads_f1=birads_f1,
        findings_bleu4=findings_bleu4,
        findings_meteor=findings_meteor,
        findings_coverage=findings_coverage,
        findings_coverage_equality=findings_coverage_equality,
    )


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")

    args = ArgumentParser()
    args.add_argument(
        "--eval_single_config",
        type=str,
        default="configs/eval_single/cesm_baseline.yaml",
    )
    args.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    args.add_argument(
        "--cumulative_metric_file",
        type=str,
        default="eval_single_metrics.json",
        help="File to save cumulative metrics",
    )
    args = args.parse_args()
    seed = args.seed
    metric_file = args.cumulative_metric_file
    args = OmegaConf.load(args.eval_single_config)

    # Set up the wandb run and models
    run = wandb.init(
        project=args.wandb.project, config=OmegaConf.to_container(args, resolve=True)
    )  # type: ignore
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_cls = get_model_class_by_name(args.model.class_name)
    seed_models = setup_seed_models(
        runs=args.model.runs, device=device, eval_mode=True, model_cls=model_cls
    )

    # 1. Set up the data loaders
    base_transformation = T.Compose(
        [
            T.ToImage(),
            T.Lambda(repeat_rgb_channels),
            T.Resize(
                args.test_data.img_resize, interpolation=T.InterpolationMode.BICUBIC
            ),
            T.CenterCrop(args.test_data.img_resize),
            T.Lambda(div_255),
            # No need to normalize here since it is done at inference
        ]
    )

    test_dataset = ReportGenerationDataset(
        imgs_path=args.test_data.imgs_path,
        csv_path=args.test_data.csv_path,
        imgs_shape=tuple(args.test_data.imgs_shape),
        transform=base_transformation,
        return_birads=True,
        return_modalities=True,
        return_exam_type=False,
        return_origin_dataset=True,
        return_patient_id=True,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.trainer.batch_size,
        num_workers=multiprocessing.cpu_count() - 1,
        pin_memory=True,
        persistent_workers=True,
    )

    # 2. Predict entire test set and cache results => We are using greedy decoding, the results are deterministic
    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[args.trainer.precision]
    ctx = (
        nullcontext()
        if device == "cpu"
        else torch.amp.autocast(device_type=device, dtype=ptdtype)
    )

    cached_predictions: list[SeedModelResults] = [
        SeedModelResults() for _ in range(len(seed_models))
    ]

    for seed_idx in range(len(seed_models)):
        model = seed_models[seed_idx]
        test_iter = iter(test_dataloader)

        while True:
            try:
                imgs, report, birads, modalities, _, patients_ids = next(test_iter)
            except StopIteration:
                logging.warning(
                    f"Finished predicting the entire test set with seed model {seed_idx}"
                )
                break

            imgs = imgs.to(device)
            if imgs.dim() == 5:
                imgs = imgs.squeeze(1)  # Remove the singleton dimension if present
            birads_np = birads.detach().numpy().tolist()
            modalities_mapped = [modality_mapping[modality] for modality in modalities]
            report = [
                rep.lower() for rep in report
            ]  # convert to lower case to ensure non case sensitive comparison

            with ctx:
                model_predictions = model.model.predict(
                    input_images=imgs,
                    pre_transform=T.Normalize(
                        mean=args.model.norm_mean, std=args.model.norm_std
                    ),
                    plot_attention=False,
                    short_report=False,
                )

            (
                _,
                predictions_findings,
                predictions_birads,
                prediction_modality,
                _,
            ) = get_content_from_predictions(model_predictions)

            cached_predictions[seed_idx].extend_with_predictions(
                modality_pred=prediction_modality,
                birads_pred=predictions_birads,
                findings_pred=predictions_findings,
            )
            cached_predictions[seed_idx].extend_with_targets(
                modality_true=modalities_mapped,
                birads_true=birads_np,
                findings_true=report,
            )
            cached_predictions[seed_idx].extend_with_patient_ids(
                patient_ids=patients_ids
            )
        cached_predictions[seed_idx].compute_lexical_metrics()

    # 3. Bootstrap iterations with results. Keeping track of scores
    metrics_performance = MetricsListHolder()
    rng = np.random.default_rng(seed)

    # Map each patient to its indices
    pat_to_idx = {}
    for i, g in enumerate(cached_predictions[0].patient_ids):
        pat_to_idx.setdefault(g, []).append(i)
    patients = list(pat_to_idx.keys())

    for _ in tqdm(
        range(args.trainer.bootstrap_iters), desc="Bootstrapping eval", unit="bootstrap"
    ):
        # Curr eval for seeds
        curr_metrics_performance = MetricsListHolder()

        for seed_idx in range(len(seed_models)):
            model = seed_models[seed_idx]

            # Draw one weight per patient
            curr_metrics_performance.extend_with_metrics(
                compute_model_performance(
                    pats=patients,
                    pats_to_idx=pat_to_idx,
                    predicted_modality=np.array(
                        cached_predictions[seed_idx].pred_modality
                    ),
                    target_modality=np.array(
                        cached_predictions[seed_idx].target_modality
                    ),
                    predicted_birads=np.array(cached_predictions[seed_idx].pred_birads),
                    target_birads=np.array(cached_predictions[seed_idx].target_birads),
                    predicted_findings=np.array(
                        cached_predictions[seed_idx].pred_findings
                    ),
                    target_findings=np.array(
                        cached_predictions[seed_idx].target_findings
                    ),
                    findings_metrics=np.array(
                        cached_predictions[seed_idx].pred_findings_lexical_metrics
                    ),
                )
            )

        metrics_performance.extend_with_average(curr_metrics_performance)

    baseline_95_ci = MetricsListHolder.compute_95_ci(metrics_performance)
    plot_confidence_interval_table(baseline_95_ci, "Model 95% CI")

    # 4. Save cumulative metrics to a file
    modality_file = args.test_data.csv_path.split("/")[-1].split("-rg-")[0]
    model_name = args.model.name

    # read existing cumulative metrics if the file exists
    if os.path.exists(metric_file):
        with open(metric_file, "r") as f:
            cumulative_metrics = json.load(f)
    else:
        cumulative_metrics = {}

    if modality_file not in cumulative_metrics:
        cumulative_metrics[modality_file] = {}

    for metric_name, vals in baseline_95_ci.items():
        if metric_name not in cumulative_metrics[modality_file]:
            cumulative_metrics[modality_file][metric_name] = {}
        cumulative_metrics[modality_file][metric_name][model_name] = vals

    with open(metric_file, "w") as f:
        json.dump(cumulative_metrics, f, indent=4)

    # 5. Log the results per seed model to a csv and load to wandb
    for seed_idx in range(len(seed_models)):
        to_save = pd.DataFrame(
            {
                "exam_id": test_dataset.data["id"],
                "patient_id": cached_predictions[seed_idx].patient_ids,
                "true_modality": cached_predictions[seed_idx].target_modality,
                "predicted_modality": cached_predictions[seed_idx].pred_modality,
                "true_birads": cached_predictions[seed_idx].target_birads,
                "predicted_birads": cached_predictions[seed_idx].pred_birads,
                "true_report": cached_predictions[seed_idx].target_findings,
                "predicted_report": cached_predictions[seed_idx].pred_findings,
                "findings_bleu4": [
                    elem["bleu4"]
                    for elem in cached_predictions[
                        seed_idx
                    ].pred_findings_lexical_metrics
                ],
                "findings_meteor": [
                    elem["meteor"]
                    for elem in cached_predictions[
                        seed_idx
                    ].pred_findings_lexical_metrics
                ],
                "findings_coverage": [
                    elem["findings_coverage"]
                    for elem in cached_predictions[
                        seed_idx
                    ].pred_findings_lexical_metrics
                ],
                "findings_effective_equality": [
                    elem["findings_effective_equality"]
                    for elem in cached_predictions[
                        seed_idx
                    ].pred_findings_lexical_metrics
                ],
            }
        )

        csv_path = (
            f"seed_{seed_idx + 1}_predictions_{args.model.name}_{modality_file}.csv"
        )
        to_save.to_csv(csv_path, index=False)
        wandb.save(csv_path)

    wandb.finish()
