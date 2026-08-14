import logging
import uuid

import torch
from omegaconf import DictConfig
from torch import nn
from tqdm import tqdm

from data_preprocessing.medical_mappings import birads_mapping


class AmberCLS(nn.Module):
    def __init__(self, model_configs: DictConfig, compute_device: str = "cpu") -> None:
        super().__init__()

        self.compute_device = compute_device

        # ENCODER
        self.encoder = torch.hub.load(
            model_configs.encoder.hub, model_configs.encoder.uri
        )
        self.features_approach = model_configs.encoder.features_approach

        # Linear CLASSIFIER
        num_classes = model_configs.mlp.num_classes
        input_dim = model_configs.mlp.input_dim
        self.net = nn.Linear(input_dim, num_classes)
        self.net.weight.data.normal_(mean=0.0, std=0.01)
        self.net.bias.data.zero_()

    def forward(
        self,
        img_or_projected: torch.Tensor,
        flat_logits: bool = True,
        process_image: bool = True,
    ) -> torch.Tensor:
        if process_image:
            projected_img_tokens = self.process_image(img_or_projected)
        else:
            projected_img_tokens = img_or_projected

        logits = self.net(projected_img_tokens)

        if flat_logits:
            logits = logits.view(-1, logits.size(-1))

        return logits

    def process_image(self, img: torch.Tensor) -> torch.Tensor:
        features_dict = self.encoder.forward_features(img)

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

        return features

    @torch.inference_mode()
    def predict(
        self,
        input_images: torch.Tensor,
        pre_transform=None,
        plot_attention: bool = False,
    ) -> list:
        was_training = self.training
        if self.training:
            logging.warning(
                "The model is in training mode. Switching to eval model for inference."
            )
            self.eval()

        # Pre-transform the input images if needed
        if pre_transform is not None:
            input_images = pre_transform(input_images)

        # Go through each image and infer the report
        inferred_results = []

        for i in tqdm(
            range(input_images.size(0)),
            desc="Inferring BI-RADS",
            total=input_images.size(0),
        ):
            projected_imgs = self.encoder(input_images[i].unsqueeze(0))
            report_id = str(uuid.uuid4())

            logits = self(
                img_or_projected=projected_imgs,
                flat_logits=True,
                process_image=False,
            )

            logits = logits[-1, :]
            probs = torch.softmax(logits, dim=-1)
            predicted_class = probs.argmax(dim=-1).item()
            predicted_class_str = birads_mapping.get(predicted_class, "unknown")

            if plot_attention:
                logging.info(
                    f"Predicted class for report {report_id}: {probs.cpu().numpy().tolist()}"
                )

            inferred_results.append(
                {
                    "predicted_class": predicted_class,
                    "predicted_class_str": predicted_class_str,
                    "class_probs": probs.cpu().numpy().tolist(),
                    "report_id": report_id,
                }
            )

        if was_training:
            logging.warning("Switching back to training mode.")
            self.train()
            self.tokenizer.train()

        return inferred_results

    @staticmethod
    def from_pretrained(
        chkpt_path: str, device: str = "cpu", eval_mode: bool = True
    ) -> "AmberCLS":
        model_configs = torch.load(chkpt_path, map_location=device, weights_only=False)
        model_configs["model"] = {
            k.replace("_orig_mod.", ""): v for k, v in model_configs["model"].items()
        }
        model = AmberCLS(
            model_configs=model_configs["config"]["model"], compute_device=device
        )
        model.load_state_dict(model_configs["model"])
        model = model.to(device)

        if eval_mode:
            model.eval()

        return model
