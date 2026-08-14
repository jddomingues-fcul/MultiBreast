import multiprocessing
from argparse import ArgumentParser

import torch
import wandb
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torchvision import transforms

from breast_datasets.report_generation_dataset import ReportGenerationDataset
from breast_datasets.stratified_batch_sampler import BootstrapingSampler
from components.plotting.plotting import (
    plot_confidence_interval_table,
    plot_p_values_table,
)
from components.processing.misc import (
    div_255,
    from_numpy,
    repeat_rgb_channels,
    unsqueeze,
)
from eval.misc import MetricsListHolder, predict_with_test_loader, setup_seed_models

if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")

    args = ArgumentParser()
    args.add_argument(
        "--eval_config", type=str, default="configs/eval/cesm_vs_all.yaml"
    )
    args = args.parse_args()

    args = OmegaConf.load(args.eval_config)

    # Set up the wandb run and models
    run = wandb.init(
        project=args.wandb.project, config=OmegaConf.to_container(args, resolve=True)
    )  # type: ignore
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Set up the models
    baseline_models = setup_seed_models(
        runs=args.baseline_model.runs, device=device, eval_mode=True
    )
    challenger_models = setup_seed_models(
        runs=args.challenger_model.runs, device=device, eval_mode=True
    )

    assert len(baseline_models) == len(challenger_models), (
        "The number of baseline and challenger models must be the same."
    )

    # Set up the data loaders
    channel_transformation = (
        unsqueeze if args.test_data.grayscale_channel else repeat_rgb_channels
    )
    base_transformation = transforms.Compose(
        [
            transforms.Lambda(from_numpy),
            transforms.Lambda(div_255),
            transforms.Lambda(channel_transformation),
            transforms.Resize(
                args.test_data.img_resize,
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.CenterCrop(args.test_data.img_resize),
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
    )
    test_targets = test_dataset.get_class_labels()

    # 1. initate the evaluator, and metrics holder for both models
    deltas = MetricsListHolder()
    baseline_metrics_performance = MetricsListHolder()
    challenger_metrics_performance = MetricsListHolder()

    # 2. for each bootstrap iteration
    for _ in range(args.trainer.bootstrap_iters):
        # 3. Current diffs evaluator
        curr_diffs = MetricsListHolder()
        curr_baseline_metrics_performance = MetricsListHolder()
        curr_challenger_metrics_performance = MetricsListHolder()

        # 4. create the sampler for the test set, needs to be stratified and done each step
        val_sampler = BootstrapingSampler(targets=test_targets)
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=args.trainer.batch_size,
            sampler=val_sampler,
            num_workers=multiprocessing.cpu_count() - 1,
            pin_memory=True,
            persistent_workers=True,
        )

        for seed_idx in range(len(baseline_models)):
            # 5. get the models for the respective seed
            baseline_model = baseline_models[seed_idx]
            challenger_model = challenger_models[seed_idx]

            # 6. predict with both models and get the evaluation results
            baseline_results = predict_with_test_loader(
                loader_iterator=iter(test_dataloader),
                model=baseline_model,
                trainer_iters=args.trainer.iters,
                norm_mean=args.baseline_model.norm_mean,
                norm_std=args.baseline_model.norm_std,
                device=device,
            )

            challenger_results = predict_with_test_loader(
                loader_iterator=iter(test_dataloader),
                model=challenger_model,
                trainer_iters=args.trainer.iters,
                norm_mean=args.challenger_model.norm_mean,
                norm_std=args.challenger_model.norm_std,
                device=device,
            )

            # 7. Compute the diffs
            metrics_diffs = baseline_results.compute_diff_with(challenger_results)

            # 8. Add the diffs to the current diffs evaluator, and the metrics performance holders
            curr_diffs.extend_with_metrics(metrics_diffs)
            curr_baseline_metrics_performance.extend_with_metrics(baseline_results)
            curr_challenger_metrics_performance.extend_with_metrics(challenger_results)

        # 9. Add the current diffs to the deltas
        deltas.extend_with_average(curr_diffs)
        baseline_metrics_performance.extend_with_average(
            curr_baseline_metrics_performance
        )
        challenger_metrics_performance.extend_with_average(
            curr_challenger_metrics_performance
        )

    # Compute CI and SE for each model individually
    baseline_95_ci = MetricsListHolder.compute_95_ci(baseline_metrics_performance)
    plot_confidence_interval_table(baseline_95_ci, "Baseline Model 95% CI")
    challenger_95_ci = MetricsListHolder.compute_95_ci(challenger_metrics_performance)
    plot_confidence_interval_table(challenger_95_ci, "Challenger Model 95% CI")

    # Compute the p-values for the deltas
    p_values = MetricsListHolder.compute_p_value(deltas)
    plot_p_values_table(p_values, "P-values for Deltas on Bootstrap Samples")

    wandb.finish()
