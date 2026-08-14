import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.v2 as T

from components.processing.misc import div_255, repeat_rgb_channels, unsqueeze
from models.amber.amber import Amber

if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    chkpt_path = "out/y7xvy11a/ckpt.pt"
    model = Amber.from_pretrained(chkpt_path=chkpt_path, device=device, eval_mode=True)

    modality = "us"
    slice_idx = 0  # for mr, use slice_idx=1000, for others use slice_idx=0

    data_path = f"../data/report_generation_split/{modality}-rg-test.csv"
    data = pd.read_csv(data_path)
    just_mrs = data[
        (data["modality"] == modality) & (data["birads"] == "negative")
    ]
    sample = just_mrs.iloc[slice_idx]
    img_path = sample["image_path"]
    seg_path = sample["segmentation_path"]
    slice = sample["slice"]

    img = np.load(img_path)[slice, :, :]

    base_transformation = T.Compose(
        [
            T.ToImage(),
            T.Lambda(unsqueeze),
            T.Lambda(repeat_rgb_channels),
            T.Resize(560, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(560),
            T.Lambda(div_255),
        ]
    )

    img = base_transformation(img)
    img = img.to(device)

    result = model.predict(
        input_images=img,
        pre_transform=T.Normalize(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        ),
        plot_attention=True,
        short_report=False,
    )[0]
    pretty_report = result["pretty_report"]
    pretty_report = pretty_report.replace(model.tokenizer._SEMICOL, "\n")

    print("==================== PREDICTED RESULT ====================")
    print(pretty_report)
    print("\n")

    print("==================== TARGET RESULT ====================")
    print(f"Modality: {modality}")
    print(f"Birads: {sample['birads']}")
    target_report = sample["report"]
    if pd.isna(target_report):
        target_report = "No report available"
    else:
        target_report = target_report.lower()
    target_report = target_report.replace(model.tokenizer._SEMICOL, "\n-")
    print(f"Findings:\n- {target_report}")

    img = img.cpu().numpy()
    img = img[0].transpose(1, 2, 0)
    img *= 255
    img = img.astype(np.uint8)
    cv2.imwrite("plots/original_image.png", img)
