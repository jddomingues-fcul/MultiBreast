import logging
import multiprocessing
from argparse import ArgumentParser
from contextlib import nullcontext

import pandas as pd
import torch
import torchvision.transforms.v2 as T
import wandb
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from breast_datasets.report_generation_dataset import ReportGenerationDataset
from components.plotting.error_violin_by_true import plot_error_distance_violin
from components.plotting.tolerance_curve import plot_tolerance_curve
from components.processing.misc import div_255, repeat_rgb_channels
from data_preprocessing.medical_mappings import modality_mapping
from eval.coverage_evaluator import CoverageEvaluator
from eval.misc import get_content_from_predictions
from eval.performance_evaluator import PerformanceEvaluator
from models import get_model_class_by_name

if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")

    args = ArgumentParser()
    args.add_argument("--eval_on_set_config", type=str, required=True)
    args.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    args = args.parse_args()
    args = OmegaConf.load(args.eval_on_set_config)

    # Set up the wandb run and models
    run = wandb.init(
        project=args.wandb.project, config=OmegaConf.to_container(args, resolve=True)
    )  # type: ignore
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_cls = get_model_class_by_name(args.model.class_name)
    model = model_cls.from_pretrained(args.model.run, device=device)

    # Set up the data loaders
    base_transformation = T.Compose(
        [
            T.ToImage(),
            T.Lambda(repeat_rgb_channels),
            T.Resize(
                args.test_data.img_resize, interpolation=T.InterpolationMode.BICUBIC
            ),
            T.CenterCrop(args.test_data.img_resize),
            T.Lambda(div_255),
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
        return_exam_ids=True,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.trainer.batch_size,
        num_workers=multiprocessing.cpu_count() - 1,
        pin_memory=True,
        persistent_workers=True,
    )

    # Keeping track of scores
    findings_coverage = CoverageEvaluator(model_name=args.model.name)
    modality_cls_evaluator = PerformanceEvaluator(
        eval_func="cls", model_name=args.model.name, scope="Modality Performance"
    )
    birads_cls_evaluator = PerformanceEvaluator(
        eval_func="cls", model_name=args.model.name, scope="BI-RADS Performance"
    )
    findings_rg_evaluator = PerformanceEvaluator(
        eval_func="rg", model_name=args.model.name, scope="Findings Performance"
    )

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

    # Evaluate the models
    loader = iter(test_dataloader)
    table_data = []

    birads_gt = []
    birads_pred = []
    modals = []
    exams_ids = []

    while True:
        try:
            imgs, report, birads, modalities, origin_datasets, eids = next(loader)
        except StopIteration:
            logging.warning("Finished predicting the entire test set")
            break

        imgs = imgs.to(device)
        if imgs.dim() == 5:
            imgs = imgs.squeeze(1)  # Remove the singleton dimension if present
        birads_np = birads.detach().numpy().tolist()
        exams_ids.extend(eids)
        birads_gt.extend(birads_np)
        modals.extend(modalities)
        modalities_mapped = [modality_mapping[modality] for modality in modalities]
        report = [
            rep.lower() for rep in report
        ]  # convert to lower case to ensure non case sensitive comparison

        with ctx:
            model_predictions = model.predict(
                input_images=imgs,
                pre_transform=T.Normalize(
                    mean=args.model.norm_mean, std=args.model.norm_std
                ),
                plot_attention=False,
                short_report=False,
            )

        (
            predictions_pretty_report,
            predictions_findings,
            predictions_birads,
            prediction_modality,
            rg_ids,
        ) = get_content_from_predictions(model_predictions)
        birads_pred.extend(predictions_birads)

        # for add info scores
        findings_coverage.add_findings(predictions_findings, report)

        # for performance scores
        modality_cls_evaluator.add_predictions(
            prediction_modality, modalities_mapped, birads_np, modalities
        )
        birads_cls_evaluator.add_predictions(
            predictions_birads, birads_np, birads_np, modalities
        )
        findings_rg_evaluator.add_predictions(
            predictions_findings, report, birads_np, modalities
        )

        # Plot the predictions to wandb as a table with image, target report, predicted report, origin dataset
        for i in range(len(predictions_pretty_report)):
            table_data.append(
                [
                    wandb.Image(imgs[i].cpu()),
                    report[i],
                    predictions_pretty_report[i],
                    f"Modality: {modalities[i]} | BI-RADS: {birads_np[i]} | Origin: {origin_datasets[i]}",
                ]
            )

    columns = ["Image", "Target Report", "Predicted Report", "Info"]
    wandb.log({"Predictions": wandb.Table(data=table_data, columns=columns)})

    # Save the predictions and targets to wandb
    modality_file = args.test_data.csv_path.split("/")[-1].split(".")[0]
    predictions_df = pd.DataFrame(
        {
            "exam_id": exams_ids,
            "true_modality": modality_cls_evaluator.global_results.ground_truth,
            "predicted_modality": modality_cls_evaluator.global_results.predictions,
            "true_birads": birads_cls_evaluator.global_results.ground_truth,
            "predicted_birads": birads_cls_evaluator.global_results.predictions,
            "true_report": findings_rg_evaluator.global_results.ground_truth,
            "predicted_report": findings_rg_evaluator.global_results.predictions,
        }
    )
    csv_path = f"predictions_{args.model.name}_{modality_file}.csv"
    predictions_df.to_csv(csv_path, index=False)
    wandb.save(csv_path)

    # Compute and plot the basic metrics
    modality_cls_evaluator.compute_metrics(with_plot=True)
    birads_cls_evaluator.compute_metrics(with_plot=True)
    findings_rg_evaluator.compute_metrics(with_plot=True)

    # compute and plot the detailed metrics
    modality_cls_evaluator.compute_detailed_metrics(with_plot=True)
    birads_cls_evaluator.compute_detailed_metrics(with_plot=True)
    findings_rg_evaluator.compute_detailed_metrics(with_plot=True)

    # plot add info scores
    findings_coverage.compute_metrics(with_plot=True)

    plot_error_distance_violin(
        true_labels=birads_gt,
        diffs=[pred - true for pred, true in zip(birads_pred, birads_gt)],
        title="BI-RADS Prediction Error Distribution by True Class",
        figsize=(8, 6),
        out_prefix=f"birads_error_violin_plot_{args.model.name}_{args.test_data.csv_path.split('/')[-1].split('.')[0]}",
        ylim=(-7, 7),
    )

    # Tolerance curve
    plot_tolerance_curve(
        y_true=birads_gt,
        y_pred=birads_pred,
        groups=modals,
        filename=f"birads_tolerance_curve_plot_{args.model.name}_{args.test_data.csv_path.split('/')[-1].split('.')[0]}",
        k_max=6,
    )

    wandb.finish()
