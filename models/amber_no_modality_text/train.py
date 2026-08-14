import logging
import multiprocessing
import os
import time
from argparse import ArgumentParser
from contextlib import nullcontext

import torch
import torchvision.transforms.v2 as T
import wandb
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchinfo import summary
from tqdm import tqdm

from breast_datasets.report_generation_dataset import ReportGenerationDataset
from breast_datasets.stratified_batch_sampler import ConstantDiffuseSampler
from components.losses import get_loss_fn
from components.lr_schedulers import get_scheduler
from components.optimizers import get_optimizer
from components.processing.misc import div_255, repeat_rgb_channels, unsqueeze
from components.train.train_utils import (
    estimate_val_loss,
    estimate_val_metrics,
    freeze_encoder_parameters,
    get_batch,
    load_from_checkpoint,
    save_model,
)
from data_preprocessing.medical_mappings import birads_mapping
from models.amber_no_modality_text.amber_no_modality import AmberNoModality

MODEL_NAME = "ckpt.pt"


def compute_loss(
    batch, model: AmberNoModality, loss_fn: nn.Module, ctx, transforms
) -> torch.Tensor:
    im, reports, birads, modalities = batch
    im = im.to(model.compute_device, non_blocking=True)
    birads_text = [birads_mapping[b.item()] for b in birads]
    decoder_input, decoder_input_mask, decoder_target = model.tokenizer.encode_batch(
        reports, birads_text, modalities
    )

    decoder_input, decoder_input_mask, decoder_target = (
        decoder_input.to(model.compute_device, non_blocking=True),
        decoder_input_mask.to(model.compute_device, non_blocking=True),
        decoder_target.to(model.compute_device, non_blocking=True),
    )

    if transforms is not None:
        im = transforms(im)

    with ctx:
        logits = model(
            im, decoder_input=decoder_input, decoder_input_mask=decoder_input_mask
        )
    loss = loss_fn(logits, decoder_target.view(-1))

    return loss


def main(cfg, debug: bool, resume_from: str | None, seed: int, device: str):
    # Set the seed for reproducibility
    torch.manual_seed(seed=seed)

    logging.basicConfig(
        filename=f"logs/{cfg.log_filename}",
        filemode="w",
        format="%(name)s - %(levelname)s - %(message)s",
        level=logging.DEBUG if debug else logging.INFO,
    )

    # for reproducibility
    run = wandb.init(
        project=cfg.wandb.project,
        config=OmegaConf.to_container(cfg, resolve=True),  # type: ignore
    )

    model_dir_save_path = os.path.join(cfg.out_dir, run.id)
    os.makedirs(name=model_dir_save_path, exist_ok=True)
    model_save_path = os.path.join(model_dir_save_path, MODEL_NAME)

    ########################################################################
    # DATA SETUP                                                           #
    ########################################################################
    train_transform = T.Compose(
        [
            T.ToImage(),
            T.Lambda(unsqueeze),
            T.Lambda(repeat_rgb_channels),
            T.Resize(cfg.data.img_resize, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(cfg.data.img_resize),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomApply(
                [
                    T.RandomAffine(
                        degrees=(-30, 30),
                        translate=(0.0, 0.0),
                        scale=(0.9, 1.1),
                        shear=(-10, 10),
                    )
                ],
                p=0.5,
            ),
            T.RandomApply(
                [T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)],
                p=0.5,
            ),
            T.RandomAdjustSharpness(3, p=0.5),
            T.Lambda(div_255),
            T.Normalize(mean=cfg.data.norm_mean, std=cfg.data.norm_std),
        ]
    )

    val_transform = T.Compose(
        [
            T.ToImage(),
            T.Lambda(unsqueeze),
            T.Lambda(repeat_rgb_channels),
            T.Resize(cfg.data.img_resize, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(cfg.data.img_resize),
            T.Lambda(div_255),
            T.Normalize(mean=cfg.data.norm_mean, std=cfg.data.norm_std),
        ]
    )

    # Train data setup
    train_dataset = ReportGenerationDataset(
        imgs_path=cfg.data.train_imgs_path,
        csv_path=cfg.data.train_csv_path,
        imgs_shape=tuple(cfg.data.imgs_shape),
        transform=None,
        return_birads=True,
        return_modalities=True,
        return_exam_type=False,
    )

    weights = train_dataset.make_weights_for_weighted_sampler()
    sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights))
    train_dataloader = DataLoader(
        dataset=train_dataset,
        sampler=sampler,
        batch_size=cfg.trainer.batch_size,
        num_workers=multiprocessing.cpu_count() - 1,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )

    # Validation data setup
    val_dataset = ReportGenerationDataset(
        imgs_path=cfg.data.val_imgs_path,
        csv_path=cfg.data.val_csv_path,
        imgs_shape=tuple(cfg.data.imgs_shape),
        transform=None,
        return_birads=True,
        return_modalities=True,
        return_exam_type=False,
    )

    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=cfg.trainer.eval_batch_size,
        shuffle=False,
        num_workers=multiprocessing.cpu_count() - 1,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )

    constant_set_sampler = ConstantDiffuseSampler(
        targets=val_dataset.get_class_labels(),
        batch_size=cfg.trainer.val_metrics_n_samples,
    )
    constant_val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=cfg.trainer.eval_batch_size,
        sampler=constant_set_sampler,
        num_workers=multiprocessing.cpu_count() - 1,
        pin_memory=True,
        persistent_workers=True,
    )

    train_dataloader_iter = iter(train_dataloader)
    val_dataloader_iter = iter(val_dataloader)

    ########################################################################
    # MODEL & MISC SETUP                                                   #
    ########################################################################
    # Model
    model = AmberNoModality(
        model_configs=cfg.model, tokenizer_configs=cfg.tokenizer, compute_device=device
    )
    model.to(device)
    model.compile()
    model.train()

    # Freeze encoder parameters if specified
    if cfg.model.encoder.freeze_layers:
        freeze_encoder_parameters(model=model, n_layers=cfg.model.encoder.freeze_layers)

    # print model information
    print(summary(model=model))

    # Loss function
    cfg.loss_fn.ignore_index = model.tokenizer.get_pad_token_id()
    loss_fn = get_loss_fn(loss_configs=cfg.loss_fn)

    # Optimizer
    optimizer = get_optimizer(cfg.optimizer)

    non_encoder_parameters = []
    for name, param in model.named_parameters():
        if "encoder" not in name:
            non_encoder_parameters.append(param)

    optimizer = optimizer(
        [
            {
                "params": model.encoder.parameters(),
                "lr": cfg.optimizer.encoder.lr,
                "weight_decay": cfg.optimizer.encoder.weight_decay,
            },
            {
                "params": non_encoder_parameters,
                "lr": cfg.optimizer.decoder.lr,
                "weight_decay": cfg.optimizer.decoder.weight_decay,
            },
        ]
    )

    # Scheduler
    scheduler = get_scheduler(scheduler_configs=cfg.lr_scheduler)(optimizer)

    iter_num = 0  # number of iterations in the lifetime of this process
    best_val_loss = 1e9
    best_modality_f1 = 0.0
    best_birads_f1 = 0.0
    best_findings_bleu = 0.0
    best_coverage = 0.0
    best_coverage_eq = 0.0

    if resume_from is not None:
        print(f"Loading checkpoint from {resume_from}")
        (
            model,
            optimizer,
            scheduler,
            iter_num,
            best_val_loss,
            best_modality_f1,
            best_birads_f1,
            best_findings_bleu,
            best_coverage,
            best_coverage_eq,
        ) = load_from_checkpoint(resume_from, device, model, optimizer, scheduler)

    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[cfg.trainer.precision]
    ctx = (
        nullcontext()
        if device == "cpu"
        else torch.amp.autocast(device_type=device, dtype=ptdtype)
    )
    scaler = torch.amp.GradScaler(enabled=(cfg.trainer.precision == "float16"))

    ########################################################################
    # TRAINING LOOP                                                        #
    ########################################################################
    batch = get_batch(loader_iter=train_dataloader_iter)  # fetch the very first batch

    patience = cfg.early_stopper.patience  # Number of epochs to wait for improvement
    min_delta = (
        cfg.early_stopper.min_delta
    )  # Minimum change in loss to qualify as improvement
    patience_counter = 0  # Counter for epochs without improvement

    while True:
        # evaluate the loss on train/val sets and write checkpoints
        if iter_num % cfg.trainer.eval_interval == 0 and iter_num > 0:
            val_loss = estimate_val_loss(
                eval_iters=cfg.trainer.eval_iters,
                loader_iter=val_dataloader_iter,
                loader=val_dataloader,
                model=model,
                loss_fn=loss_fn,
                compute_loss_fn=compute_loss,
                ctx=ctx,
                transforms=val_transform,
            )

            val_metrics = estimate_val_metrics(
                loader=constant_val_dataloader,
                model=model,
                ctx=ctx,
                transforms=val_transform,
            )

            curr_mod_f1 = val_metrics["modality"]["global"]["f1 score"]
            curr_bi_f1 = val_metrics["birads"]["global"]["f1 score"]
            curr_findings_bleu = val_metrics["findings"]["global"]["bleu4"]
            curr_coverage = val_metrics["findings_coverage"]["findings_coverage"]
            curr_coverage_eq = val_metrics["findings_coverage"][
                "findings_effective_equality"
            ]

            print(
                f"iter {iter_num}: val loss {val_loss:.4f}, modality f1 {curr_mod_f1:.4f}, birads f1 {curr_bi_f1:.4f}, findings bleu {curr_findings_bleu:.4f}, coverage {curr_coverage:.4f}, coverage effective equality {curr_coverage_eq:.4f}"
            )
            wandb.log({"iter": iter_num, "val/loss": val_loss})

            for metric_name, metric_value in val_metrics.items():
                if metric_value is None:
                    continue
                for sub_metric_name, sub_metric_value in metric_value.items():
                    wandb.log(
                        {f"val/{metric_name}/{sub_metric_name}": sub_metric_value}
                    )

            if (val_loss < best_val_loss - min_delta) or (
                best_modality_f1 + best_birads_f1 + best_findings_bleu
                < curr_mod_f1
                + curr_bi_f1
                + curr_findings_bleu
                + curr_coverage
                + curr_coverage_eq
                - min_delta
            ):
                best_modality_f1 = curr_mod_f1
                best_birads_f1 = curr_bi_f1
                best_findings_bleu = curr_findings_bleu
                best_coverage = curr_coverage
                best_coverage_eq = curr_coverage_eq

                best_val_loss = val_loss
                patience_counter = 0  # Reset patience counter

                if iter_num >= 0:
                    save_model(
                        model,
                        optimizer,
                        scheduler,
                        iter_num,
                        cfg,
                        val_loss,
                        best_val_loss,
                        best_findings_bleu,
                        best_modality_f1,
                        best_birads_f1,
                        best_coverage,
                        best_coverage_eq,
                        model_save_path,
                    )
            else:
                patience_counter += 1
                print(
                    f"No improvement in val loss nor the metrics for {patience_counter} evaluations."
                )

            # Stop training if patience is exceeded
            if patience_counter >= patience:
                print("Early stopping triggered. Training stopped.")
                break

        # forward backward update, with gradient accumulation to simulate larger batch size
        # and using the GradScaler if data type is float16
        for _ in tqdm(
            range(cfg.trainer.accumulate_grad_batches),
            desc="Gradient Accumulation",
            disable=cfg.trainer.accumulate_grad_batches == 1,
        ):
            with ctx:
                # with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as p:
                loss = compute_loss(
                    batch=batch,
                    model=model,
                    loss_fn=loss_fn,
                    ctx=ctx,
                    transforms=train_transform,
                )
                loss = (
                    loss / cfg.trainer.accumulate_grad_batches
                )  # scale the loss to account for gradient accumulation
                # break

            batch = get_batch(loader_iter=train_dataloader_iter)
            if batch is None:
                # reset the iterator and get a new batch
                train_dataloader_iter = iter(train_dataloader)
                batch = get_batch(loader_iter=train_dataloader_iter)

            # backward pass, with gradient scaling if training in fp16
            scaler.scale(loss).backward()

        # clip the gradient
        if cfg.trainer.gradient_clip_val > 0.0:
            scaler.unscale_(optimizer)
            pre_clip_norm = torch.nn.utils.clip_grad_norm_(
                parameters=model.parameters(),
                max_norm=cfg.trainer.gradient_clip_val,
                foreach=True,
            )
            wandb.log({"manual_gradient_tracking/pre_clip_norm": pre_clip_norm})

        # step the optimizer and scaler if training in fp16
        scaler.step(optimizer)
        scaler.update()

        # step the scheduler
        scheduler.step()

        # flush the gradients as soon as we can, no need for this memory anymore
        optimizer.zero_grad(set_to_none=True)

        # logging
        if iter_num % cfg.trainer.log_interval == 0:
            # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
            lossf = loss.item() * cfg.trainer.accumulate_grad_batches
            wandb.log(
                {
                    "iter": iter_num,
                    "train/loss": lossf,
                    "lr": scheduler.get_last_lr()[0],
                }
            )
            print(f"iter {iter_num}: train loss {lossf:.4f}")

        iter_num += 1

        # termination conditions
        if iter_num > cfg.trainer.max_iters:
            break

        # Compute elapsed time
        elapsed_time = time.time() - run.start_time
        if elapsed_time > cfg.timer.duration:
            print("Training stopped due to exceeding time limit.")
            break

    # Save the final model to wandb
    wandb.save(glob_str=model_save_path, base_path=model_dir_save_path)
    wandb.finish()

    # clean up after training
    del train_dataset
    del train_dataloader
    del val_dataset
    del val_dataloader
    del model
    del loss_fn
    del optimizer
    del scheduler
    del batch

    # clean cuda cache after each training seed
    if device == "cuda":
        torch.cuda.empty_cache()

    print("Training finished.")


if __name__ == "__main__":
    # Set the precision to medium to maximize the performance
    torch.set_float32_matmul_precision("medium")

    args = ArgumentParser()
    args.add_argument(
        "--yaml_config",
        type=str,
        default="configs/train/amber/baseline.yaml",
        help="Path to the YAML configuration file",
    )
    args.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    args.add_argument(
        "--resume_from", type=str, default=None, help="Path to resume checkpoint from"
    )
    args.add_argument("--debug", action="store_true", default=False)
    args = args.parse_args()

    cfg = OmegaConf.load(file_=args.yaml_config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # more room for compilation
    torch._dynamo.config.cache_size_limit = 512
    torch._dynamo.config.accumulated_cache_size_limit = 2048

    print(f"Running training with seed {args.seed}...")
    main(
        cfg=cfg,
        debug=args.debug,
        resume_from=args.resume_from,
        seed=args.seed,
        device=device,
    )
