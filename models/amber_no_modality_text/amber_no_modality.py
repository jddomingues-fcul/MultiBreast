import uuid

import torch
from omegaconf import DictConfig

from components.analyzer.utils import compute_toks_entropy
from components.plotting.plotting import (
    plot_last_token_cross_attn_heads_cosine_similarity,
    plot_last_token_self_attn_heads_cosine_similarity,
    plot_mse_last_token_cross_attn_heads,
    plot_mse_last_token_self_attn_heads,
    plot_token_token_attention,
    plot_tokens_attention_to_image,
)
from data_preprocessing.medical_mappings import birads_mapping
from models.amber.amber import Amber
from models.amber_no_modality_text.pre_trained_tokenizer_no_modality import (
    PreTrainedTokenizerNoModality,
)


class AmberNoModality(Amber):
    def __init__(
        self,
        model_configs: DictConfig,
        tokenizer_configs: DictConfig,
        compute_device: str = "cpu",
    ) -> None:
        super().__init__(
            model_configs=model_configs,
            tokenizer_configs=tokenizer_configs,
            compute_device=compute_device,
        )

        # TOKENIZER
        self.tokenizer = PreTrainedTokenizerNoModality(
            block_size=tokenizer_configs.block_size,
            pre_trained_tokenizer=tokenizer_configs.pre_trained_tokenizer,
            shuffle_findings=tokenizer_configs.shuffle_findings,
        )

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
        t = 1

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

    @staticmethod
    def from_pretrained(
        chkpt_path: str, device: str = "cpu", eval_mode: bool = True
    ) -> "Amber":
        model_configs = torch.load(chkpt_path, map_location=device, weights_only=False)
        model_configs["model"] = {
            k.replace("_orig_mod.", ""): v for k, v in model_configs["model"].items()
        }  # Remove the _orig_mod. prefix from the keys
        model = AmberNoModality(
            model_configs=model_configs["config"]["model"],
            tokenizer_configs=model_configs["config"]["tokenizer"],
            compute_device=device,
        )
        model.load_state_dict(model_configs["model"])
        model = model.to(device)

        if eval_mode:
            model.eval()
            model.tokenizer.eval()

        return model
