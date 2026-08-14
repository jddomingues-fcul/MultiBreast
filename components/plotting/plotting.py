import os
import re
from textwrap import fill

import cv2
import matplotlib
import numpy as np
import seaborn as sns
import torch
import wandb
from matplotlib import pyplot as plt
from torch import nn

from components.analyzer.cka import get_cka_matrix

matplotlib.use("Agg")  # Close the figure to free memory


def plot_samples(
    x,
    y,
    preds,
    n_samples: int = 3,
    text_preds: bool = False,
    title: str = "Samples",
    samples_ids=None,
    original_dataset: str = "Not defined",
):
    n_samples = min(n_samples, len(x))
    fig, axs = plt.subplots(n_samples, 3, figsize=(15, 5 * n_samples))
    samples_ids_str = ""
    if n_samples == 1:
        axs = np.expand_dims(axs, axis=0)
    for i in range(n_samples):
        if samples_ids is not None:
            samples_ids_str = samples_ids_str + samples_ids[i] + " | "

        og_image = x[i]
        og_img = og_image[
            0
        ]  # if is rgb, we just take one channel because it's the same
        axs[i, 0].imshow(og_img, cmap="gray")
        axs[i, 0].set_title(f"Image (Source Dataset: {original_dataset})")
        axs[i, 0].axis("off")

        if text_preds:
            wrapped_report = fill(y[i], width=40)
            wrapped_predicted_report = fill(preds[i], width=40)
            axs[i, 1].text(
                0.5, 0.5, wrapped_report, wrap=True, ha="center", va="center"
            )
            axs[i, 2].text(
                0.5, 0.5, wrapped_predicted_report, wrap=True, ha="center", va="center"
            )
        else:
            axs[i, 1].imshow(y[i], cmap="gray")
            axs[i, 2].imshow(preds[i], cmap="gray")

        axs[i, 1].set_title("Ground Truth")
        axs[i, 1].axis("off")
        axs[i, 2].set_title("Prediction")
        axs[i, 2].axis("off")

    plt.tight_layout()
    plt.show()
    wandb.log({slugify(f"{title} -- {samples_ids_str}"): wandb.Image(plt)})
    plt.close()


def imshow(
    inp: torch.Tensor,
    mean: list | float,
    std: list | float,
    title: str | None = None,
):
    if len(inp) == 3:
        inp = inp.numpy().transpose((1, 2, 0))
        mean = np.array(mean)
        std = np.array(std)
    inp = std * inp + mean  # unnormalize the image
    inp = np.clip(inp, 0, 1)  # clip to make values between 0 and 1
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.axis("off")


def display_tensor(tensor: torch.Tensor, ax, title: str = ""):
    if len(tensor.shape) == 3 and tensor.shape[2] == 3:
        ax.imshow(tensor)
    else:
        ax.matshow(tensor)
    plt.title(title)
    plt.axis("off")


def plot_cls_patches_similarity(
    image: torch.Tensor,
    featuremap_size: int,
    mean: list | float,
    std: list | float,
    patches_embeddings: torch.Tensor,
    cls_embedding: torch.Tensor,
    wandb_logger,
):
    plt.rcParams["figure.figsize"] = [5, 5]
    plt.subplot(1, 2, 1)

    features = patches_embeddings.squeeze(0).detach().cpu()
    cls_token = cls_embedding.squeeze(0).detach().cpu()
    cls_token_sims = (
        (features * cls_token).sum(-1).reshape((featuremap_size, featuremap_size))
    )

    imshow(image, mean, std, "Original")

    ax = plt.subplot(1, 2, 2)
    display_tensor(cls_token_sims, ax, "Similarities\nof patch tokens\nwith CLS token")

    wandb_logger.log({"cls_patches_similarities": wandb.Image(plt)})


def save_attention(
    attention_scores: torch.Tensor,
    image: torch.Tensor,
    decoder_out,
    inferred_report: str,
):
    # mha is the crossattention layer. masked_mha is the self-attention layer
    data = attention_scores.mean(axis=0).detach().cpu().numpy()
    im = np.squeeze(image.cpu().permute(1, 2, 0).numpy().astype(np.uint8))
    fig = plt.figure(figsize=(40, 80))
    iters = data.shape[0]

    for i in range(iters):
        attn_token = np.array(data[i, :])
        res_shape = np.sqrt(attn_token.shape[0]).astype(int)
        attn_token = attn_token.reshape((res_shape, res_shape))

        ax = fig.add_subplot(25, 10, (i + 1))
        ax.set_title(decoder_out[i])

        # Plot the image to align with the attn_token
        img = ax.imshow(im, cmap="gray")

        # Add the attn_token as a heatmap and plot the colorbar
        ax.imshow(attn_token, alpha=0.5, extent=img.get_extent())

    plt.close("all")
    plt.title(inferred_report)

    # Log the figure to wandb
    attn_heatmap = wandb.Image(fig)
    wandb.log(
        {"Attention (cross-attention, last layer, mean over heads)": attn_heatmap}
    )


def plot_logits_probs(
    logits_probs, type, top_k, extra_info_plot_text, report_id, tokenizer
):
    top_k_logits_probs, top_k_idx = torch.topk(logits_probs, top_k, dim=-1)
    top_k_vocab = dict()
    for i, (logit_pro, k_idx) in enumerate(zip(top_k_logits_probs[0], top_k_idx[0])):
        top_k_vocab[
            tokenizer.tokenizer.decode([k_idx.item()], skip_special_tokens=False)
        ] = logit_pro.item()

    # plot the top_k_vocab as a histogram and save it to wandb
    plt.figure(figsize=(10, 5))
    plt.bar(top_k_vocab.keys(), top_k_vocab.values())
    plt.xlabel("Tokens")
    plt.ylabel(f"{type}")
    plt.title(f"Top {top_k} {type} Logits: {extra_info_plot_text}")
    plt.xticks(rotation=90)
    plt.tight_layout()

    hist = wandb.Image(plt)
    plt.close()
    wandb.log({f"infer/reports/{report_id}/pred_token_{type}_hist": hist})


def plot_token_token_attention(
    model, tokens: list[str], report_id: str, predicted_token: str, len_tokens: int
):
    avg_cross_attention_scores = torch.stack(
        [
            layer.attn.attention_scores.detach().squeeze(0).cpu().mean(0)
            for layer in model.decoder.transformer.h
        ]
    ).mean(0)
    plot_tokens_attention(
        attention=avg_cross_attention_scores,
        tokens=tokens,
        report_id=report_id,
        token=predicted_token,
        len_tokens=len_tokens,
    )


def plot_tokens_attention(
    attention: torch.Tensor,
    tokens: list[str],
    report_id: str,
    token: str,
    len_tokens: int,
):
    plt.figure(figsize=(10, 10))
    plt.imshow(attention, cmap="viridis")
    plt.xticks(range(len(tokens)), tokens, rotation=90)
    plt.yticks(range(len(tokens)), tokens)
    plt.title(f"Average attention when predicting: {token}")
    plt.colorbar()
    plt.tight_layout()

    # save the plot to local
    if not os.path.exists("plots"):
        os.makedirs("plots")
    plt.savefig(
        f"plots/report_{report_id}_past_tokens_attention_to_each_other_when_predicting_token_number_{len_tokens}.png"
    )
    plt.close()


def plot_tokens_img_attention(
    attention: np.ndarray,
    token: str,
    image: torch.Tensor,
    report_id: str,
    len_tokens: int,
):
    no_cls_attn = attention[1:] if attention.shape[0] % 2 != 0 else attention
    if image.ndim == 4:
        image = image.squeeze(0)  # remove batch dimension if exists
    n_imgs_patches = no_cls_attn.shape[-1]
    n_height_patches = int(np.sqrt(n_imgs_patches))
    n_width_patches = n_height_patches

    # reshape attention to match the height and width of the image
    no_cls_attn = no_cls_attn.reshape(n_height_patches, n_width_patches)

    # resize the attention to match the image size
    no_cls_attn = cv2.resize(no_cls_attn, (image.shape[1], image.shape[2]))

    # normalize the attention
    attention_resized = no_cls_attn / no_cls_attn.max()

    plt.figure(figsize=(10, 10))
    plt.imshow(image[0].cpu(), cmap="gray")
    plt.imshow(attention_resized, cmap="viridis", alpha=0.6)
    plt.title(f"Average attention when predicting: {token}")
    plt.colorbar()
    plt.tight_layout()

    # save the plot to local
    if not os.path.exists("plots"):
        os.makedirs("plots")

    plt.savefig(
        f"plots/report_{report_id}_past_tokens_attention_to_image_when_predicting_token_number_{len_tokens}.png"
    )
    plt.close()

    cv2.imwrite(
        f"plots/RAW_report_{report_id}_past_tokens_attention_to_image_when_predicting_token_number_{len_tokens}.png",
        attention_resized * 255,
    )


def plot_tokens_attention_to_image(
    model, predicted_token: str, token_idx: int, report_id: str, image: torch.Tensor
):
    avg_cross_attention_scores = torch.stack(
        [
            layer.cross_attention.attention_scores.detach().squeeze(0).cpu().mean(0)
            for layer in model.decoder.transformer.h
        ]
    ).mean(0)
    avg_cross_attention_scores = (
        avg_cross_attention_scores.mean(0).detach().cpu().numpy()
    )  # We average the attention scores of all tokens to the image
    plot_tokens_img_attention(
        attention=avg_cross_attention_scores,
        token=predicted_token,
        image=image,
        report_id=report_id,
        len_tokens=token_idx,
    )


def plot_confidence_scores(
    trues_scores: list[float], false_scores: list[float], plot_title: str
):
    plt.figure(figsize=(10, 5))
    plt.hist(trues_scores, bins=20, alpha=0.5, label="True", color="green")
    plt.hist(false_scores, bins=20, alpha=0.5, label="False", color="red")
    plt.xlabel("Confidence Scores")
    plt.ylabel("Frequency")
    plt.title(plot_title)
    plt.legend()
    plt.tight_layout()

    hist = wandb.Image(plt)
    plt.close()
    wandb.log({slugify(plot_title): hist})


def plot_findings_scores(
    coverages: list[float], effective_eqs: list[float], plot_title: str
):
    coverage_avg = (
        sum(coverages) / len(coverages) if len(coverages) > 0 else 0
    )  # avoid division by zero if there are no coverages
    effective_eqs_avg = (
        sum(effective_eqs) / len(effective_eqs) if len(effective_eqs) > 0 else 0
    )  # avoid division by zero if there are no effective_eqs

    # log the average coverage and effective_eqs as wandb table
    wb_table = wandb.Table(
        data=[[coverage_avg, effective_eqs_avg]], columns=["Coverage", "Effective EQs"]
    )
    wandb.log({plot_title: wb_table})


def plot_basic_performance_table(results_map: dict, title: str):
    result_table = wandb.Table(columns=["metric", "value"])
    for key, value in results_map.items():
        result_table.add_data(key, value)
    wandb.log({slugify(title): result_table})


def slugify(title: str) -> str:
    # 1) spaces and hyphens to underscore
    title = re.sub(r"[ \-]", "_", title)

    # 2) remove any character that is not alphanumeric or underscore
    title = re.sub(r"[^\w]", "", title)

    return title


def plot_confidence_interval_table(metrics_dict: dict, title: str):
    """
    Plot a table with confidence intervals for the given metrics.
    :param metrics_dict: Dictionary with metrics and their confidence intervals.
    :param title: Title for the table.
    """
    ci_table = wandb.Table(
        columns=["Metric", "Lower CI (95%)", "Mean Performance", "Upper CI (95%)"]
    )
    for metric, vals in metrics_dict.items():
        lower_ci, mean_performance, upper_ci = vals
        ci_table.add_data(metric, lower_ci, mean_performance, upper_ci)
    wandb.log({slugify(title): ci_table})


def plot_p_values_table(p_values: dict, title: str):
    """
    Plot a table with p-values for the given metrics.
    :param p_values: Dictionary with metrics and their p-values.
    :param title: Title for the table.
    """
    p_values_table = wandb.Table(columns=["Metric", "P-value"])
    for metric, p_value in p_values.items():
        p_values_table.add_data(metric, p_value)
    wandb.log({slugify(title): p_values_table})


def plot_last_token_cross_attn_cka(
    model_decoder: nn.Module, report_id: str, len_tokens: int
):
    arr = []
    for i in range(len(model_decoder.transformer.h)):
        arr.append(
            model_decoder.transformer.h[i].cross_attention.attention_scores.squeeze(0)
        )

    arr = torch.stack(arr)  # (num_layers, num_heads, seq_len, img_tokens_len)
    feats = arr[:, :, -1, :]

    mat = get_cka_matrix(feats, feats)
    mat = mat - torch.diag(
        torch.diag(mat)
    )  # set the diagonal to zero for better visualization

    plt.figure(figsize=(10, 10))
    plt.imshow(mat.detach().cpu().numpy(), cmap="inferno")
    plt.title(f"CKA Matrix of Cross-Attention Features for token idx {len_tokens}")
    plt.colorbar()
    plt.tight_layout()

    # save the plot to local
    if not os.path.exists("plots"):
        os.makedirs("plots")

    plt.savefig(
        f"plots/report_{report_id}_cross_attention_cka_for_token_idx_{len_tokens}.png"
    )
    plt.close()


def plot_last_token_self_attn_cka(
    model_decoder: nn.Module, report_id: str, len_tokens: int
):
    arr = []
    for i in range(len(model_decoder.transformer.h)):
        arr.append(model_decoder.transformer.h[i].attn.attention_scores.squeeze(0))

    arr = torch.stack(arr)  # (num_layers, num_heads, seq_len, img_tokens_len)
    feats = arr[:, :, -1, :]

    mat = get_cka_matrix(feats, feats)
    mat = mat - torch.diag(
        torch.diag(mat)
    )  # set the diagonal to zero for better visualization

    plt.figure(figsize=(10, 10))
    plt.imshow(mat.detach().cpu().numpy(), cmap="inferno")
    plt.title(f"CKA Matrix of Self-Attention Features for token idx {len_tokens}")
    plt.colorbar()
    plt.tight_layout()

    # save the plot to local
    if not os.path.exists("plots"):
        os.makedirs("plots")

    plt.savefig(
        f"plots/report_{report_id}_self_attention_cka_for_token_idx_{len_tokens}.png"
    )
    plt.close()


def plot_last_token_self_attn_heads_cosine_similarity(
    model_decoder: nn.Module,
    report_id: str,
    len_tokens: int,
    eps: float = 1e-10,
    dim: int = -1,
):
    arr = []
    for i in range(len(model_decoder.transformer.h)):
        head_rep_last_token = model_decoder.transformer.h[
            i
        ].attn.attention_scores.squeeze(0)[:, -1, :]
        layer_mean = head_rep_last_token.mean(0)
        de_biased_rep = head_rep_last_token - layer_mean.unsqueeze(0)
        arr.append(de_biased_rep)

    arr = torch.cat(arr)  # (num_layers, num_heads, seq_len, img_tokens_len)
    cosine_map = torch.nn.functional.cosine_similarity(
        arr.unsqueeze(0), arr.unsqueeze(1), dim=dim, eps=eps
    )

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(cosine_map.detach().cpu().numpy(), cmap="inferno", robust=True, ax=ax)
    ax.set_title(
        f"Per-layer centered cosine similarity matrix across all heads (Self-Attention) Token idx {len_tokens}"
    )
    ax.set_xlabel("layer/head index")
    ax.set_ylabel("layer/head index")
    plt.tight_layout()

    # save the plot to local
    os.makedirs("plots", exist_ok=True)

    plt.savefig(
        f"plots/report_{report_id}_per_layer_centered_cosine_sim_matrix_across_all_self_attn_heads_tok_idx_{len_tokens}.png"
    )
    plt.close()


def plot_last_token_cross_attn_heads_cosine_similarity(
    model_decoder: nn.Module,
    report_id: str,
    len_tokens: int,
    eps: float = 1e-10,
    dim: int = -1,
):
    arr = []
    for i in range(len(model_decoder.transformer.h)):
        head_rep_last_token = model_decoder.transformer.h[
            i
        ].cross_attention.attention_scores.squeeze(0)[:, -1, :]
        layer_mean = head_rep_last_token.mean(0)
        de_biased_rep = head_rep_last_token - layer_mean.unsqueeze(0)
        arr.append(de_biased_rep)

    arr = torch.cat(arr)  # (num_layers, num_heads, seq_len, img_tokens_len)
    cosine_map = torch.nn.functional.cosine_similarity(
        arr.unsqueeze(0), arr.unsqueeze(1), dim=dim, eps=eps
    )

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(cosine_map.detach().cpu().numpy(), cmap="inferno", robust=True, ax=ax)
    ax.set_title(
        f"Per-layer centered cosine similarity matrix across all heads (Cross-Attention) Token idx {len_tokens}"
    )
    ax.set_xlabel("layer/head index")
    ax.set_ylabel("layer/head index")
    plt.tight_layout()

    # save the plot to local
    os.makedirs("plots", exist_ok=True)

    plt.savefig(
        f"plots/report_{report_id}_per_layer_centered_cosine_sim_matrix_across_all_cross_attn_heads_tok_idx_{len_tokens}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def plot_mse_last_token_cross_attn_heads(
    model_decoder: nn.Module, image: torch.Tensor, report_id: str, len_tokens: int
):
    # attention: np.ndarray, token: str

    arr = []
    n_layers = len(model_decoder.transformer.h)
    for i in range(n_layers):
        head_rep_last_token = model_decoder.transformer.h[
            i
        ].cross_attention.attention_scores.squeeze(0)[:, -1, :]

        no_cls_attn = (
            head_rep_last_token[:, 1:]
            if head_rep_last_token.shape[1] % 2 != 0
            else head_rep_last_token
        )
        n_imgs_patches = no_cls_attn.shape[-1]
        n_height_patches = int(np.sqrt(n_imgs_patches))
        n_width_patches = n_height_patches

        # reshape attention to match the height and width of the image
        no_cls_attn = no_cls_attn.reshape(
            (-1, n_height_patches, n_width_patches)
        )  # (num_heads, height_patches, width_patches)

        # normalize the attention
        attention_resized = no_cls_attn / no_cls_attn.max()
        arr.append(attention_resized)

    arr = torch.cat(arr)  # (num_layers, num_heads, seq_len, img_tokens_len)

    mse_matrix = torch.zeros(
        (n_layers, n_layers), device=image.device, dtype=image.dtype
    )
    for i in range(n_layers):
        for j in range(n_layers):
            mse_matrix[i, j] = torch.nn.functional.mse_loss(arr[i], arr[j]).item()

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(mse_matrix.detach().cpu().numpy(), cmap="inferno", robust=True, ax=ax)
    ax.set_title(
        f"MSE across all layers per cross-attention heads Token idx {len_tokens}"
    )
    ax.set_xlabel("layer/head index")
    ax.set_ylabel("layer/head index")
    plt.tight_layout()

    # save the plot to local
    os.makedirs("plots", exist_ok=True)

    plt.savefig(
        f"plots/report_{report_id}_mse_for_similarity_cross_attn_all_layers_token_idx{len_tokens}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def plot_mse_last_token_self_attn_heads(
    model_decoder: nn.Module, image: torch.Tensor, report_id: str, len_tokens: int
):
    # attention: np.ndarray, token: str

    arr = []
    n_layers = len(model_decoder.transformer.h)
    for i in range(n_layers):
        head_rep_last_token = model_decoder.transformer.h[
            i
        ].attn.attention_scores.squeeze(0)[:, -1, :]

        # normalize the attention
        attention_scaled = head_rep_last_token / head_rep_last_token.max()
        arr.append(attention_scaled)

    arr = torch.cat(arr)  # (num_layers, num_heads, seq_len, img_tokens_len)

    mse_matrix = torch.zeros(
        (n_layers, n_layers), device=image.device, dtype=image.dtype
    )
    for i in range(n_layers):
        for j in range(n_layers):
            mse_matrix[i, j] = torch.nn.functional.mse_loss(arr[i], arr[j]).item()

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(mse_matrix.detach().cpu().numpy(), cmap="inferno", robust=True, ax=ax)
    ax.set_title(
        f"MSE across all layers per self-attention heads Token idx {len_tokens}"
    )
    ax.set_xlabel("layer/head index")
    ax.set_ylabel("layer/head index")
    plt.tight_layout()

    # save the plot to local
    os.makedirs("plots", exist_ok=True)

    plt.savefig(
        f"plots/report_{report_id}_mse_for_similarity_self_attn_all_layers_token_idx{len_tokens}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
