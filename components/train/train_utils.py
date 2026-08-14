import logging

import torch
from omegaconf import DictConfig, ListConfig
from torch import nn

from data_preprocessing.medical_mappings import (
    UNKNOWN,
    birads_mapping,
    modality_mapping,
    modality_mapping_reverse,
)
from eval.coverage_evaluator import CoverageEvaluator
from eval.misc import get_content_from_predictions
from eval.performance_evaluator import PerformanceEvaluator


def save_model(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    iter_num: int,
    cfg: DictConfig | ListConfig,
    val_loss: float,
    best_val_loss: float,
    best_findings_bleu: float,
    best_modality_f1: float,
    best_birads_f1: float,
    best_coverage: float,
    best_coverage_eq: float,
    model_save_path: str,
):
    """Saves the model checkpoint with the current state of the model, optimizer, scheduler, and other relevant information.

    Args:
        model (nn.Module): Model to be saved.
        optimizer (torch.optim.Optimizer): Optimizer state to be saved.
        scheduler (torch.optim.lr_scheduler.LRScheduler): Scheduler state to be saved.
        iter_num (int): Current iteration number.
        cfg (DictConfig | ListConfig): Configuration object containing model parameters and other settings.
        val_loss (float): Current validation loss.
        best_val_loss (float): Best validation loss achieved so far.
        best_findings_bleu (float): Best BLEU score for findings achieved so far.
        best_modality_f1 (float): Best F1 score for modality classification achieved so far.
        best_birads_f1 (float): Best F1 score for BIRADS classification achieved so far.
        best_coverage (float): Best coverage score for findings achieved so far.
        best_coverage_eq (float): Best effective equality coverage score for findings achieved so far.
        model_save_path (str): Path where the model checkpoint will be saved.
    """

    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "model_args": cfg.model,
        "iter_num": iter_num,
        "best_val_loss": best_val_loss,
        "config": cfg,
        "scheduler": scheduler.state_dict(),
        "best_findings_bleu": best_findings_bleu,
        "best_modality_f1": best_modality_f1,
        "best_birads_f1": best_birads_f1,
        "best_coverage": best_coverage,
        "best_coverage_eq": best_coverage_eq,
    }
    print(
        f"saving checkpoint to {cfg.out_dir} at iter {iter_num}, val loss {val_loss:.4f}, modality f1 {best_modality_f1:.4f}, birads f1 {best_birads_f1:.4f}, findings bleu {best_findings_bleu:.4f}"
    )

    torch.save(checkpoint, model_save_path)


def load_from_checkpoint(
    checkpoint_path: str,
    device: str,
    created_model: nn.Module,
    created_optim: torch.optim.Optimizer,
    created_scheduler: torch.optim.lr_scheduler.LRScheduler,
) -> tuple:
    """Loads model from a checkpoint.

    Args:
        checkpoint_path (str): Path to the checkpoint file.
        device (str): Device to load the model onto (e.g., 'cpu' or 'cuda').
        created_model (nn.Module): Model to be loaded.
        created_tokenizer (BreastCancerTokenizer): Tokenizer to be loaded.
        created_optim (torch.optim.Optimizer): Optimizer to be loaded.
        created_scheduler (torch.optim.lr_scheduler.LRScheduler): Scheduler to be loaded.

    Returns:
        tuple: Contains the loaded model, optimizer, scheduler, iteration number, best validation loss, best modality F1 score, best BIRADS F1 score, and best findings BLEU score.
    """
    checkpoint = torch.load(f=checkpoint_path, map_location=device, weights_only=False)

    created_model.load_state_dict(state_dict=checkpoint["model"])
    created_optim.load_state_dict(state_dict=checkpoint["optimizer"])
    created_scheduler.load_state_dict(state_dict=checkpoint["scheduler"])

    iter_num = checkpoint["iter_num"]
    best_val_loss = checkpoint.get("best_val_loss", 1e9)
    best_modality_f1 = checkpoint.get("best_modality_f1", 0.0)
    best_birads_f1 = checkpoint.get("best_birads_f1", 0.0)
    best_findings_bleu = checkpoint.get("best_findings_bleu", 0.0)
    best_coverage = checkpoint.get("best_coverage", 0.0)
    best_coverage_eq = checkpoint.get("best_coverage_eq", 0.0)

    return (
        created_model,
        created_optim,
        created_scheduler,
        iter_num,
        best_val_loss,
        best_modality_f1,
        best_birads_f1,
        best_findings_bleu,
        best_coverage,
        best_coverage_eq,
    )


def freeze_encoder_parameters(model: nn.Module, n_layers: int | bool):
    """Freezes the parameters of the encoder for the first n_layers.

    Args:
        model (nn.Module): Model whose encoder parameters will be frozen.
        n_layers (int | bool): Number of layers to freeze. If True, all layers are frozen.
    """
    logging.info(f"Freezing encoder parameters. Layers to freeze: {n_layers}")

    # convert n_layers bool to int. If it is bool, it will be True, to freeze all layers
    if isinstance(n_layers, bool) and n_layers:
        for n, _ in model.named_parameters():
            if "encoder.blocks" in n:
                n_layers = int(n.split(".")[2])

    # freeze encoder parameters, first n_layer
    for n, p in model.named_parameters():
        if "encoder.blocks" in n:
            layer_num = int(n.split(".")[2])
            if layer_num < n_layers:
                logging.info(f"Freezing {n}")
                p.requires_grad = False
            else:
                p.requires_grad = True
        else:
            p.requires_grad = True


def get_batch(loader_iter):
    """Retrieves the next batch from the data loader iterator.

    Args:
        loader_iter (_type_): Iterator for the data loader.

    Returns:
        _type_: Returns the next batch from the iterator, or None if the iterator is exhausted.
    """
    try:
        batch = next(loader_iter)
        return batch
    except StopIteration:
        return None


@torch.no_grad()
def estimate_val_loss_cls(
    eval_iters: int,
    loader_iter,
    loader,
    model: nn.Module,
    loss_fn: nn.Module,
    compute_loss_fn,
    ctx,
    transforms=None,
) -> float:
    model.eval()

    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        batch = get_batch(loader_iter)
        if batch is None:
            # reset the iterator and get a new batch
            loader_iter = iter(loader)
            batch = get_batch(loader_iter)

        loss = compute_loss_fn(
            batch=batch, model=model, loss_fn=loss_fn, ctx=ctx, transforms=transforms
        )
        losses[k] = loss.item()

    model.train()
    return losses.mean().item()


@torch.no_grad()
def estimate_val_loss(
    eval_iters: int,
    loader_iter,
    loader,
    model: nn.Module,
    loss_fn: nn.Module,
    compute_loss_fn,
    ctx,
    transforms=None,
) -> float:
    """Estimates the validation loss over a number of iterations.

    Args:
        eval_iters (int): Number of iterations to evaluate the validation loss.
        loader_iter (_type_): Iterator for the data loader.
        loader (_type_): Data loader containing the validation data.
        model (nn.Module): Model used for computing the loss.
        loss_fn (nn.Module): Loss function used to compute the loss.
        device (str): Device to which the data and model should be moved (e.g., 'cpu' or 'cuda').
        ctx (_type_): Context manager for handling the computation.

    Returns:
        float: Returns the average validation loss over the specified number of iterations.
    """
    model.eval()
    model.tokenizer.eval()

    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        batch = get_batch(loader_iter)
        if batch is None:
            # reset the iterator and get a new batch
            loader_iter = iter(loader)
            batch = get_batch(loader_iter)

        loss = compute_loss_fn(
            batch=batch, model=model, loss_fn=loss_fn, ctx=ctx, transforms=transforms
        )
        losses[k] = loss.item()

    model.train()
    model.tokenizer.train()
    return losses.mean().item()


@torch.no_grad()
def estimate_val_loss_with_batch(
    batch, model: nn.Module, loss_fn: nn.Module, compute_loss_fn, ctx, transforms
) -> float:
    """Estimates the validation loss over a batch

    Args:
        batch (_type_): Batch of data containing images, reports, birads, and modalities.
        model (nn.Module): Model used for computing the loss.
        loss_fn (nn.Module): Loss function used to compute the loss.
        ctx (_type_): Context manager for handling the computation.

    Returns:
        float: Returns the average validation loss over the specified number of iterations.
    """
    model.eval()
    model.tokenizer.eval()
    loss = compute_loss_fn(
        batch=batch, model=model, loss_fn=loss_fn, ctx=ctx, transforms=transforms
    )
    model.train()
    model.tokenizer.train()
    return loss.item()


@torch.no_grad()
def estimate_val_metrics_cls(
    loader,
    model: nn.Module,
    ctx,
    transforms=None,
) -> dict:
    model.eval()

    birads_cls_evaluator = PerformanceEvaluator(
        eval_func="cls", model_name="", scope="BI-RADS val Performance"
    )
    loader_iter = iter(loader)

    while True:
        batch = get_batch(loader_iter)
        if batch is None:
            break  # Exit the loop if there are no more batches

        imgs, report, birads, modalities = batch
        birads_np = birads.detach().numpy().tolist()
        imgs = imgs.to(model.compute_device, non_blocking=True)
        # birads = birads.to(model.compute_device, non_blocking=True)

        if transforms is not None:
            imgs = transforms(imgs)

        with ctx:
            predictions = model.predict(
                input_images=imgs, pre_transform=None, plot_attention=False
            )

        predicted_birads = [elem["predicted_class"] for elem in predictions]

        for m, pb, b in zip(modalities, predicted_birads, birads_np):
            print(f"modality: {m} birads: {b} predicted birads: {pb}")

        birads_cls_evaluator.add_predictions(
            predictions=predicted_birads,
            gt=birads_np,
            birads=birads_np,
            modalities=modalities,
        )

        # Compute and plot the basic metrics
        bm = birads_cls_evaluator.compute_metrics(with_plot=False)

        results = {"birads": bm}

        model.train()
        return results


@torch.no_grad()
def estimate_val_metrics(
    loader,
    model: nn.Module,
    ctx,
    transforms=None,
) -> dict:
    """Estimates the validation metrics for a given batch.

    Args:
        loader_iter (_type_): Iterator for the data loader.
        loader (_type_): Data loader containing the validation data.
        model (nn.Module): Model used for computing the metrics.
        batch (_type_): Batch of data containing images, reports, birads, and modalities.
        device (str): Device to which the data and model should be moved (e.g., 'cpu' or 'cuda').

    Returns:
        dict: Returns a dictionary containing the computed metrics for modality classification, BIRADS classification, and findings generation.
    """
    model.eval()
    model.tokenizer.eval()

    modality_cls_evaluator = PerformanceEvaluator(
        eval_func="cls", model_name="", scope="Modality val Performance"
    )
    birads_cls_evaluator = PerformanceEvaluator(
        eval_func="cls", model_name="", scope="BI-RADS val Performance"
    )
    findings_rg_evaluator = PerformanceEvaluator(
        eval_func="rg", model_name="", scope="Findings val Performance"
    )
    info_coverage = CoverageEvaluator(model_name="Model")
    loader_iter = iter(loader)

    while True:
        batch = get_batch(loader_iter)
        if batch is None:
            break  # Exit the loop if there are no more batches

        imgs, report, birads, modalities = batch
        imgs = imgs.to(model.compute_device, non_blocking=True)
        if transforms is not None:
            imgs = transforms(imgs)

        birads_np = birads.detach().numpy().tolist()
        birads_text = [birads_mapping[b.item()] for b in birads]
        modalities_mapped = [modality_mapping[modality] for modality in modalities]
        report = [
            rep.lower() for rep in report
        ]  # convert to lower case to ensure non case sensitive comparison

        with ctx:
            predictions = model.predict(
                input_images=imgs,
                pre_transform=None,
                plot_attention=False,
                short_report=False,
            )  # the image batch is already preprocessed, no need to mess with it otherwise we will have a bug

        _, predictions_findings, predictions_birads, prediction_modality, _ = (
            get_content_from_predictions(predictions=predictions)
        )

        for m, pm, b, pb, f, pf in zip(
            modalities,
            prediction_modality,
            birads_text,
            predictions_birads,
            report,
            predictions_findings,
        ):
            print(
                f"modality: {m}            prediction: {modality_mapping_reverse.get(pm, UNKNOWN)}"
            )
            print(
                f"birads: {b}              prediction: {birads_mapping.get(pb, UNKNOWN)}"
            )
            print(f"findings: {f}             prediction: {pf}\n")

        # for add info scores
        info_coverage.add_findings(
            predicted_reports=predictions_findings, gt_reports=report
        )

        # for performance scores
        modality_cls_evaluator.add_predictions(
            predictions=prediction_modality,
            gt=modalities_mapped,
            birads=birads_np,
            modalities=modalities,
        )
        birads_cls_evaluator.add_predictions(
            predictions=predictions_birads,
            gt=birads_np,
            birads=birads_np,
            modalities=modalities,
        )
        findings_rg_evaluator.add_predictions(
            predictions=predictions_findings,
            gt=report,
            birads=birads_np,
            modalities=modalities,
        )

    # Compute and plot the basic metrics
    mm = modality_cls_evaluator.compute_metrics(with_plot=False)
    bm = birads_cls_evaluator.compute_metrics(with_plot=False)
    fm = findings_rg_evaluator.compute_metrics(with_plot=False)
    finds_cov = info_coverage.compute_metrics(with_plot=False)

    results = {
        "modality": mm,
        "birads": bm,
        "findings": fm,
        "findings_coverage": finds_cov,
    }

    model.train()
    model.tokenizer.train()
    return results


@torch.no_grad()
def estimate_val_metrics_with_batch(
    batch,
    model: nn.Module,
    ctx,
    transforms=None,
) -> dict:
    """Estimates the validation metrics for a given batch.

    Args:
        batch (_type_): Batch of data containing images, reports, birads, and modalities.
        model (nn.Module): Model used for computing the metrics.
        batch (_type_): Batch of data containing images, reports, birads, and modalities.
        device (str): Device to which the data and model should be moved (e.g., 'cpu' or 'cuda').

    Returns:
        dict: Returns a dictionary containing the computed metrics for modality classification, BIRADS classification, and findings generation.
    """
    model.eval()
    model.tokenizer.eval()

    modality_cls_evaluator = PerformanceEvaluator(
        eval_func="cls", model_name="", scope="Modality val Performance"
    )
    birads_cls_evaluator = PerformanceEvaluator(
        eval_func="cls", model_name="", scope="BI-RADS val Performance"
    )
    findings_rg_evaluator = PerformanceEvaluator(
        eval_func="rg", model_name="", scope="Findings val Performance"
    )
    info_coverage = CoverageEvaluator(model_name="Model")

    imgs, report, birads, modalities = batch
    imgs = imgs.to(model.compute_device, non_blocking=True)
    if transforms is not None:
        imgs = transforms(imgs)
    birads_np = birads.detach().numpy().tolist()
    birads_text = [birads_mapping[b.item()] for b in birads]
    modalities_mapped = [modality_mapping[modality] for modality in modalities]
    report = [
        rep.lower() for rep in report
    ]  # convert to lower case to ensure non case sensitive comparison

    with ctx:
        predictions = model.predict(
            input_images=imgs,
            pre_transform=None,
            plot_attention=False,
            short_report=False,
        )  # the image batch is already preprocessed, no need to mess with it otherwise we will have a bug

    _, predictions_findings, predictions_birads, prediction_modality, _ = (
        get_content_from_predictions(predictions=predictions)
    )

    for m, pm, b, pb, f, pf in zip(
        modalities,
        prediction_modality,
        birads_text,
        predictions_birads,
        report,
        predictions_findings,
    ):
        print(
            f"modality: {m}            prediction: {modality_mapping_reverse.get(pm, UNKNOWN)}"
        )
        print(f"birads: {b}              prediction: {birads_mapping.get(pb, UNKNOWN)}")
        print(f"findings: {f}             prediction: {pf}\n")

    # for add info scores
    info_coverage.add_findings(
        predicted_reports=predictions_findings, gt_reports=report
    )

    # for performance scores
    modality_cls_evaluator.add_predictions(
        predictions=prediction_modality,
        gt=modalities_mapped,
        birads=birads_np,
        modalities=modalities,
    )
    birads_cls_evaluator.add_predictions(
        predictions=predictions_birads,
        gt=birads_np,
        birads=birads_np,
        modalities=modalities,
    )
    findings_rg_evaluator.add_predictions(
        predictions=predictions_findings,
        gt=report,
        birads=birads_np,
        modalities=modalities,
    )

    # Compute and plot the basic metrics
    mm = modality_cls_evaluator.compute_metrics(with_plot=False)
    bm = birads_cls_evaluator.compute_metrics(with_plot=False)
    fm = findings_rg_evaluator.compute_metrics(with_plot=False)
    finds_cov = info_coverage.compute_metrics(with_plot=False)

    results = {
        "modality": mm,
        "birads": bm,
        "findings": fm,
        "findings_coverage": finds_cov,
    }

    model.train()
    model.tokenizer.train()
    return results
