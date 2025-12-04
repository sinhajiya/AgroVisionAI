
from PIL import Image
from typing import Dict, Any
import torch

from models.clip_inference import predict_with_clip
from models.blip_caption import caption_image


def extract_visual_features(
    image_path: str,
    data_dir: str,
    top_k_clip: int = 5,
    device: str = "cpu"
) -> Dict[str, Any]:
 
    img = Image.open(image_path).convert("RGB")

    clip_results = predict_with_clip(
        image_path=image_path,
        data_dir=data_dir,
        top_k=top_k_clip
    )
    blip_caption = caption_image(
        img,
        device=device,
        max_length=80,
        num_beams=1,
        do_sample=False
    )

    return {
        "blip_caption": blip_caption,
        "clip_predictions": clip_results
    }


if __name__ == "__main__":

    import sys
    from pathlib import Path

    image_path = Path("/home/iiserb/Documents/agro/dataset/plantvillage/PlantVillage/Potato___Late_blight/0b2bdc8e-90fd-4bb4-bedb-485502fe8a96___RS_LB 4906.JPG")
    data_dir = "./dataset/archive(1)/plantvillage dataset/color"

    if not image_path.exists():
        print("Image not found:", image_path)
        sys.exit(1)

    out = extract_visual_features(str(image_path), data_dir)
    print("\nBLIP Caption:")
    print(out["blip_caption"])

    print("\nCLIP Predictions:")
    for r in out["clip_predictions"]:
        print(f"- {r['prompt']}  ({r['score']:.4f})")
