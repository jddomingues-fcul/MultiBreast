import re

import torch
from torch import nn
from transformers import AutoModel


def convert_raddino_to_dinov2_keys(keys) -> list:
    keys = [re.sub(r"^embeddings.", "", key) for key in keys]
    keys = [key.replace("position_embeddings", "pos_embed") for key in keys]
    keys = [key.replace("patch_embeddings.", "patch_embed.") for key in keys]
    keys = [key.replace("projection.", "proj.") for key in keys]
    keys = [key.replace("encoder.layer.", "blocks.") for key in keys]
    keys = [key.replace("attention.attention.", "attn.") for key in keys]
    keys = [key.replace("attention.", "attn.") for key in keys]
    keys = [key.replace(".query.", ".qkv.") for key in keys]
    keys = [key.replace(".key.", ".qkv.") for key in keys]
    keys = [key.replace(".value.", ".qkv.") for key in keys]
    keys = [key.replace(".output.dense.", ".proj.") for key in keys]
    keys = [key.replace(".layer_scale1.", ".ls1.") for key in keys]
    keys = [key.replace(".layer_scale2.", ".ls2.") for key in keys]
    keys = [key.replace("lambda1", "gamma") for key in keys]
    keys = [key.replace("layernorm.", "norm.") for key in keys]
    return keys


def get_raddinov2_model(hub: str, uri: str, load_rad_dino: bool = True) -> nn.Module:
    encoder = torch.hub.load(hub, uri)

    if not load_rad_dino:
        return encoder

    medical_model = AutoModel.from_pretrained(
        "microsoft/rad-dino-maira-2",
        trust_remote_code=True,
    )

    medical_model_sd_keys = medical_model.state_dict().keys()
    medical_model_sd_keys = convert_raddino_to_dinov2_keys(medical_model_sd_keys)

    new_rad_dino_sd = {}
    for key, value in zip(medical_model_sd_keys, medical_model.state_dict().values()):
        if key not in new_rad_dino_sd:
            new_rad_dino_sd[key] = value
        else:
            new_rad_dino_sd[key] = torch.cat((new_rad_dino_sd[key], value), dim=0)

    missing, unexpected = encoder.load_state_dict(new_rad_dino_sd, strict=False)

    print("Missing encoder keys:", missing)
    print("Unexpected encoder keys:", unexpected)

    return encoder
