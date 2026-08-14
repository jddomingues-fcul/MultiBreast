import logging
import uuid

import torch
import torch.nn as nn
from omegaconf import DictConfig
from tqdm import tqdm

from components.analyzer.utils import compute_toks_entropy
from components.layers.projector import ImageTextProjector
from components.plotting.plotting import (
    plot_last_token_cross_attn_heads_cosine_similarity,
    plot_last_token_self_attn_heads_cosine_similarity,
    plot_mse_last_token_cross_attn_heads,
    plot_mse_last_token_self_attn_heads,
    plot_token_token_attention,
    plot_tokens_attention_to_image,
)
from data_preprocessing.medical_mappings import birads_mapping, modality_mapping
from models.amber.gpt2 import GPT2
from models.amber.pre_trained_tokenizer import PreTrainedTokenizer
from models.amber.raddinov2 import get_raddinov2_model


class Amber(nn.Module):
    def __init__(
        self,
        model_configs: DictConfig,
        tokenizer_configs: DictConfig,
        compute_device: str = "cpu",
        load_rad_dino: bool = True,
    ) -> None:
        super(Amber, self).__init__()

        self.compute_device = compute_device

        self.tokenizer = PreTrainedTokenizer(
            block_size=tokenizer_configs.block_size,
            pre_trained_tokenizer=tokenizer_configs.pre_trained_tokenizer,
            shuffle_findings=tokenizer_configs.shuffle_findings,
        )

        self.encoder = get_raddinov2_model(
            model_configs.encoder.hub,
            model_configs.encoder.uri,
            load_rad_dino=load_rad_dino,
        )

        self.img_projector = ImageTextProjector(
            embed_dim=self.encoder.embed_dim,
            expansion_factor=model_configs.projector.expansion_factor,
        )

        self.decoder = GPT2(
            model_configs=model_configs,
            vocab_size=self.tokenizer.get_vocab_size(),
            init_weights=model_configs.init_weights,
        )

        self.block_size = model_configs.decoder.block_size
        self.features_approach = model_configs.encoder.features_approach
        self.n_layers = model_configs.decoder.n_layers

        if self.features_approach == "film_cls_patches":
            self.film_scale = nn.Linear(
                self.encoder.embed_dim, self.encoder.embed_dim, bias=False
            )
            self.film_bias = nn.Linear(
                self.encoder.embed_dim, self.encoder.embed_dim, bias=False
            )

        if model_configs.init_weights:
            self.img_projector.apply(self.__init_weights)

    def __init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(
        self,
        img_or_projected: torch.Tensor,
        decoder_input: torch.Tensor,
        decoder_input_mask: torch.Tensor,
        flat_logits: bool = True,
        process_image: bool = True,
    ) -> torch.Tensor:
        if process_image:
            projected_img_tokens = self.process_image(img_or_projected)
        else:
            projected_img_tokens = img_or_projected

        logits = self.decoder(
            idx=decoder_input,
            encoder_memory=projected_img_tokens,
            decoder_input_mask=decoder_input_mask,
        )

        if flat_logits:
            logits = logits.view(-1, self.tokenizer.get_vocab_size())

        return logits

    def process_image(self, img: torch.Tensor, project: bool = True) -> torch.Tensor:
        features_dict = self.encoder.forward_features(img)  # type: ignore

        if self.features_approach == "patch_tokens":
            features = features_dict["x_norm_patchtokens"]
        elif self.features_approach == "cls_token":
            features = features_dict["x_norm_clstoken"]
        elif self.features_approach == "combined":
            features = features_dict["x_norm_patchtokens"]
            cls_token = features_dict["x_norm_clstoken"]
            features = features * (cls_token.unsqueeze(1) * features).sum(
                -1, keepdims=True
            )
            features = self.encoder.norm(features)  # type: ignore # Normalize the features
        elif self.features_approach == "cls_with_patches":
            features = features_dict["x_norm_patchtokens"]
            cls_token = features_dict["x_norm_clstoken"]
            features = torch.cat(
                (cls_token.unsqueeze(1), features), dim=1
            )  # CLS: (768), Patches: (1400, 768), Features = (1401, 768)
        elif self.features_approach == "film_cls_patches":
            features = features_dict["x_norm_patchtokens"]
            cls_token = features_dict["x_norm_clstoken"].unsqueeze(1)
            fs = self.film_scale(cls_token)
            fb = self.film_bias(cls_token)
            features = features * fs + fb
        else:
            raise ValueError(f"Unknown features approach: {self.features_approach}")

        if project:
            return self.img_projector(features)

        return features

    @torch.inference_mode()
    def predict(
        self,
        input_images: torch.Tensor,
        pre_transform=None,
        plot_attention: bool = False,
        short_report: bool = False,
    ) -> list:
        was_training = self.training
        if self.training:
            logging.warning(
                "The model is in training mode. Switching to eval model for inference."
            )
            self.eval()
            self.tokenizer.eval()

        # Pre-transform the input images if needed
        if pre_transform is not None:
            input_images = pre_transform(input_images)

        # Go through each image and infer the report
        inferred_results = []

        c_mask = torch.tril(
            torch.ones(
                self.block_size,
                self.block_size,
                dtype=torch.bool,
                device=self.compute_device,
            ),
            diagonal=0,
        )
        for i in tqdm(
            range(input_images.size(0)),
            desc="Inferring reports",
            total=input_images.size(0),
        ):
            projected_imgs = self.process_image(input_images[i].unsqueeze(0))
            inferred_tokens, report_id = self.infer_report(
                image=input_images[i].unsqueeze(0),
                projected_img_tokens=projected_imgs,
                causal_mask=c_mask,
                plot_attention=plot_attention,
                short_report=short_report,
            )

            decoded_target_result = self.tokenizer.decode(inferred_tokens)
            additional_results = self.tokenizer.pretty_decode(
                inferred_tokens, return_structured_predictions=True
            )

            inferred_results.append(
                {
                    "n_tokens": len(inferred_tokens),
                    "decoded_report": decoded_target_result,
                    "report_id": report_id,
                    **additional_results,
                }
            )

        if was_training:
            logging.warning("Switching back to training mode.")
            self.train()
            self.tokenizer.train()

        return inferred_results

    @torch.inference_mode()
    def infer_report(
        self,
        image: torch.Tensor,
        projected_img_tokens: torch.Tensor,
        causal_mask: torch.Tensor,
        plot_attention: bool = False,
        short_report: bool = False,
    ):

        report_id = str(uuid.uuid4())
        tokens_entropy = []
        idx = torch.zeros(
            (1, self.block_size + 30), dtype=torch.long, device=self.compute_device
        )  # +30 to be safe and ensure birads is predicted
        idx[0, 0] = self.tokenizer.get_sos_token_id()
        idx[0, 1] = self.tokenizer.get_modality_start_token_id()
        t = 2

        # First predict modality
        idx, t = self.infer_structured_report(
            list_of_options=list(modality_mapping.keys())
            + [self.tokenizer._MODALITY_END],
            idx=idx,
            time_step=t,
            image=image,
            projected_img_tokens=projected_img_tokens,
            causal_mask=causal_mask,
            report_id=report_id,
            stop_token=self.tokenizer.get_modality_end_token_id(),
            plot_attention=plot_attention,
            tokens_entropy=tokens_entropy,
        )

        # If we are generating a short report, we stop here. Useful on cases to analyse the attention of the model when predicting the birads and modality and huge chunk of data
        if short_report:
            # We now predict the birads
            idx[0, t] = self.tokenizer.get_birads_start_token_id()
            t += 1
            idx, t = self.infer_structured_report(
                list_of_options=list(birads_mapping.values())
                + [self.tokenizer._BIRADS_END],
                idx=idx,
                time_step=t,
                image=image,
                projected_img_tokens=projected_img_tokens,
                causal_mask=causal_mask,
                report_id=report_id,
                stop_token=self.tokenizer.get_birads_end_token_id(),
                plot_attention=plot_attention,
                tokens_entropy=tokens_entropy,
            )

            if idx[0, t - 1] != self.tokenizer.get_eos_token_id():
                idx[0, t] = self.tokenizer.get_eos_token_id()
                t += 1

            return idx[0, :t].detach().cpu().numpy().tolist(), report_id

        # 2 - Predict findings freely, stopping at birads prediction
        while (
            t < self.block_size
        ):  # Worst case scenario we predict as many finding as the block size, and then have to cut initial context
            out_tok_idx = self.infer_next(
                list_of_options=[
                    self.tokenizer._FINDING_START,
                    self.tokenizer._BIRADS_START,
                ],
                idx=idx,
                time_step=t,
                projected_img_tokens=projected_img_tokens,
                causal_mask=causal_mask,
                tokens_entropy=tokens_entropy,
            )

            if out_tok_idx == self.tokenizer.get_birads_start_token_id():
                idx[0, t] = self.tokenizer.get_birads_start_token_id()
                t += 1
                break
            else:
                idx[0, t] = self.tokenizer.get_finding_start_token_id()
                t += 1

            # Within findings, we predict until we find the finding end token
            while t < self.block_size:
                curr_idx = (
                    idx[:, -self.block_size : t] if t > self.block_size else idx[:, :t]
                )
                curr_causal_mask = causal_mask[
                    : curr_idx.size(1), : curr_idx.size(1)
                ].unsqueeze(0)
                logits = self(
                    projected_img_tokens,
                    curr_idx,
                    curr_causal_mask,
                    process_image=False,
                )  # use the pre-computed projected image tokens

                logits = logits[-1, :]
                probs = torch.log_softmax(logits, dim=-1)
                idx_next = probs.argmax(dim=-1).reshape(-1, 1)  # greedy decoding
                tokens_entropy.append(compute_toks_entropy(probs))

                if plot_attention:
                    decoded_idx = self.tokenizer.decode(
                        idx_next[0].detach().cpu().numpy().tolist(),
                        skip_special_tokens=False,
                    )
                    current_tokens = [
                        self.tokenizer.decode([dec_inpt], skip_special_tokens=False)
                        for dec_inpt in idx[0, :t].detach().cpu().numpy().tolist()
                    ]
                    plot_token_token_attention(
                        model=self,
                        tokens=current_tokens,
                        report_id=report_id,
                        predicted_token=decoded_idx,
                        len_tokens=t,
                    )
                    plot_tokens_attention_to_image(
                        model=self,
                        predicted_token=decoded_idx,
                        token_idx=t,
                        report_id=report_id,
                        image=image,
                    )

                    plot_last_token_cross_attn_heads_cosine_similarity(
                        self.decoder, report_id=report_id, len_tokens=t
                    )
                    plot_last_token_self_attn_heads_cosine_similarity(
                        self.decoder, report_id=report_id, len_tokens=t
                    )
                    plot_mse_last_token_cross_attn_heads(
                        self.decoder, image=image, report_id=report_id, len_tokens=t
                    )
                    plot_mse_last_token_self_attn_heads(
                        self.decoder, image=image, report_id=report_id, len_tokens=t
                    )

                if idx_next[0, 0] in self.tokenizer.ADDED_SPECIAL_TOKENS_IDS:
                    idx[0, t] = self.tokenizer.get_finding_end_token_id()
                    t += 1
                    break
                else:
                    idx[0, t] = idx_next[0, 0]
                    t += 1

            if idx_next[0, 0] != self.tokenizer.get_finding_end_token_id():
                break

        if idx[0, t - 1] != self.tokenizer.get_birads_start_token_id():
            idx[0, t] = self.tokenizer.get_birads_start_token_id()
            t += 1

        # Finally predict the birads
        idx, t = self.infer_structured_report(
            list_of_options=list(birads_mapping.values())
            + [self.tokenizer._BIRADS_END],
            idx=idx,
            time_step=t,
            image=image,
            projected_img_tokens=projected_img_tokens,
            causal_mask=causal_mask,
            report_id=report_id,
            stop_token=self.tokenizer.get_birads_end_token_id(),
            plot_attention=plot_attention,
            tokens_entropy=tokens_entropy,
        )

        if idx[0, t - 1] != self.tokenizer.get_eos_token_id():
            idx[0, t] = self.tokenizer.get_eos_token_id()
            t += 1

        if plot_attention:
            # Save the entropy values
            import pandas as pd

            entropy_df = pd.DataFrame(tokens_entropy, columns=["tokens_entropy"])
            entropy_df["predicted_token"] = [
                self.tokenizer.decode([idx_val], skip_special_tokens=False)
                for idx_val in idx[0, 2 : t - 1].detach().cpu().numpy().tolist()
            ]
            entropy_df.to_csv(f"plots/entropy_report_{report_id}.csv", index=False)

        return idx[0, :t].detach().cpu().numpy().tolist(), report_id

    @torch.inference_mode()
    def infer_structured_report(
        self,
        list_of_options: list[str],
        idx: torch.Tensor,
        time_step: int,
        image: torch.Tensor,
        projected_img_tokens: torch.Tensor,
        causal_mask: torch.Tensor,
        report_id: str,
        stop_token: int,
        plot_attention: bool = False,
        tokens_entropy: list = [],
    ) -> tuple[torch.Tensor, int]:

        # Compute the allowed tokens for the model to predict
        limit_tokens = 0
        allowed_tokens = []
        max_structured_tokens = 1
        for opt in list_of_options:
            curr_toks = self.tokenizer.tokenizer.encode(
                opt, add_special_tokens=False
            ).ids
            if len(curr_toks) > max_structured_tokens:
                max_structured_tokens = len(curr_toks)
            allowed_tokens.extend(curr_toks)

        # Loop and predict the structured report
        while limit_tokens < (
            max_structured_tokens + 1
        ):  # +1 for the initiative of the model to predict the stop token
            # Cap the context if needed and construct the mask
            curr_idx = (
                idx[:, -self.block_size : time_step]
                if time_step > self.block_size
                else idx[:, :time_step]
            )
            curr_causal_mask = causal_mask[
                : curr_idx.size(1), : curr_idx.size(1)
            ].unsqueeze(0)
            logits = self(
                projected_img_tokens, curr_idx, curr_causal_mask, process_image=False
            )  # use the pre-computed projected image tokens
            logits = logits[-1, :]

            logits_mask = torch.full_like(
                logits, -float("Inf"), device=self.compute_device
            )
            logits_mask[allowed_tokens] = logits[allowed_tokens]
            probs = torch.log_softmax(logits_mask, dim=-1)
            idx_next = probs.argmax(dim=-1).reshape(-1, 1)
            tokens_entropy.append(compute_toks_entropy(probs))
            del logits_mask

            # Plot the attention scores from the tokens to the image
            if plot_attention:
                decoded_idx = self.tokenizer.decode(
                    idx_next[0].detach().cpu().numpy().tolist(),
                    skip_special_tokens=False,
                )
                plot_tokens_attention_to_image(
                    model=self,
                    predicted_token=decoded_idx,
                    token_idx=time_step,
                    report_id=report_id,
                    image=image,
                )

                plot_last_token_cross_attn_heads_cosine_similarity(
                    self.decoder, report_id=report_id, len_tokens=time_step
                )
                plot_last_token_self_attn_heads_cosine_similarity(
                    self.decoder, report_id=report_id, len_tokens=time_step
                )
                plot_mse_last_token_cross_attn_heads(
                    self.decoder, image=image, report_id=report_id, len_tokens=time_step
                )
                plot_mse_last_token_self_attn_heads(
                    self.decoder, image=image, report_id=report_id, len_tokens=time_step
                )

            idx[0, time_step] = idx_next[0, 0]
            time_step += 1
            limit_tokens += 1

            if stop_token == idx_next[0, 0]:
                break

        if idx[0, time_step - 1] != stop_token:
            idx[0, time_step] = stop_token
            time_step += 1

        return idx, time_step

    @torch.inference_mode()
    def infer_next(
        self,
        list_of_options: list[str],
        idx: torch.Tensor,
        time_step: int,
        projected_img_tokens: torch.Tensor,
        causal_mask: torch.Tensor,
        tokens_entropy: list = [],
    ):

        # Compute the allowed tokens for the model to predict
        allowed_tokens = []
        for opt in list_of_options:
            curr_toks = self.tokenizer.tokenizer.encode(
                opt, add_special_tokens=False
            ).ids
            allowed_tokens.extend(curr_toks)

        # Cap the context if needed and construct the mask
        curr_idx = (
            idx[:, -self.block_size : time_step]
            if time_step > self.block_size
            else idx[:, :time_step]
        )
        curr_causal_mask = causal_mask[
            : curr_idx.size(1), : curr_idx.size(1)
        ].unsqueeze(0)
        logits = self(
            projected_img_tokens, curr_idx, curr_causal_mask, process_image=False
        )  # use the pre-computed projected image tokens
        logits = logits[-1, :]

        # Condition the model to predict only the allowed tokens
        mask = torch.full_like(logits, -float("Inf"), device=self.compute_device)
        mask[allowed_tokens] = logits[allowed_tokens]
        probs = torch.log_softmax(mask, dim=-1)
        tokens_entropy.append(compute_toks_entropy(probs))
        del mask

        idx_next = probs.argmax(dim=-1).reshape(-1, 1)
        return idx_next[0, 0].item()

    @staticmethod
    def from_pretrained(
        chkpt_path: str, device: str = "cpu", eval_mode: bool = True
    ) -> "Amber":
        model_configs = torch.load(chkpt_path, map_location=device, weights_only=False)

        model_configs["model"] = {
            k.replace("_orig_mod.", ""): v for k, v in model_configs["model"].items()
        }

        model = Amber(
            model_configs=model_configs["config"]["model"],
            tokenizer_configs=model_configs["config"]["tokenizer"],
            compute_device=device,
            load_rad_dino=False,  # important
        )

        missing, unexpected = model.load_state_dict(
            model_configs["model"], strict=False
        )

        print("Missing checkpoint keys:", missing)
        print("Unexpected checkpoint keys:", unexpected)

        model = model.to(device)

        if eval_mode:
            model.eval()
            model.tokenizer.eval()

        return model
