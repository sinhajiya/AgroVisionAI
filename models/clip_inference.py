'''
Load CLIP
Load all prompts from labels_to_prompts.py
Encode all prompts to text_embeddings
Load image
Encode image to image_embedding
Compute cosine similarity → scores
Identify top-1 or top-k classes
'''

import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from models.labels_to_prompts import load_prompts
import os
import torch.nn.functional as F


def load_clip(device="cpu"):
    """
    Load CLIP model + processor on CPU.
    """
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor


def predict_with_clip(image_path, data_dir, top_k=5):
    """
    Given an image path and dataset directory, predict top-k classes using CLIP.
    """
    device = "cpu"
# generate prompts
    prompts = load_prompts(data_dir)
# Load CLIP model and processor
    model, processor = load_clip(device)
    # Load image
    image = Image.open(image_path).convert("RGB")

    # Prepare model inputs
    inputs = processor(
        text=prompts,
        images=image,
        return_tensors="pt",
        padding=True
    ).to(device)

    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract embeddings
    image_embeds = outputs.image_embeds          # shape (1, 512)
    text_embeds = outputs.text_embeds            # shape (num_prompts, 512)

    image_embeds = F.normalize(image_embeds, p=2, dim=-1)
    text_embeds = F.normalize(text_embeds, p=2, dim=-1)
    similarity = image_embeds @ text_embeds.T
    similarity = similarity.squeeze(0)  

    top_k_scores, top_k_indices = similarity.topk(top_k)

    results = []
    for score, idx in zip(top_k_scores, top_k_indices):
        results.append({
            "prompt": prompts[idx],
            "score": float(score.item())
        })

    return results


if __name__ == "__main__":

    data_dir = "./dataset/archive(1)/plantvillage dataset/color"
    test_image = "/home/iiserb/Documents/agro/dataset/plantvillage/PlantVillage/Pepper__bell___Bacterial_spot/0a4c007d-41ab-4659-99cb-8a4ae4d07a55___NREC_B.Spot 1954.JPG"

    if not os.path.exists(test_image):
        raise FileNotFoundError(f"Test image not found: {test_image}")

    preds = predict_with_clip(test_image, data_dir, top_k=5)

    print("\nTop-5 Predictions:\n")
    for p in preds:
        print(f"{p['prompt']}   (score: {p['score']:.4f})")
