from typing import Dict, List

from PIL import Image
from typing import Dict, Any
import torch

# from .models.clip_inference import predict_with_clip
# from .models.blip_caption import caption_image


def build_llm_prompt(
    blip_caption: str,
    clip_predictions: List[Dict],
    all_labels: List[str],
    context_docs: str = "",
    k: int = 5
) -> str:
   

    clip_lines = []
    for pred in clip_predictions[:k]:
        clip_lines.append(f"- {pred['prompt']}  (score: {pred['score']:.4f})")
    clip_block = "\n".join(clip_lines)

    # Format dataset label space
    label_block = "\n".join([f"- {lbl}" for lbl in all_labels])

    # Final structured prompt
    prompt = f"""
You are an agricultural plant disease diagnosis expert.

Your task is to identify the correct crop and disease from the dataset label space,
using the following vision evidence:

BLIP visual caption (general image description):
"{blip_caption}"

Top-{k} CLIP similarity predictions (not always accurate, but useful as rough signals):
{clip_block}

Dataset label space (these are the ONLY valid output classes):
{label_block}

If reference knowledge is provided, use it to match symptoms accurately.
RAG expert knowledge (optional):
{context_docs if context_docs else "No additional expert documents retrieved."}

Your output MUST be a pure JSON object with the following keys:
- "predicted_label": one label from the dataset label space
- "confidence": float between 0 and 1, based on your certainty
- "reasoning": 2–4 sentences explaining how BLIP+CLIP+knowledge led to the label
- "treatment": brief actionable steps IF the leaf is diseased, otherwise "None"

Rules:
1. Prefer BLIP for visual symptoms.
2. Use CLIP only as a weak prior (healthy vs diseased tendency).
3. Use the label space to determine the specific crop/disease.
4. If the leaf appears healthy, choose the appropriate *healthy* class.
5. Respond in valid JSON, no extra text.

Now provide the JSON output:
"""
    return prompt


if __name__ == "__main__":
    # Test example
    blip_cap = "a close up of a leaf with a brown substance on it"
    clip_preds = [
        {"prompt": "a photo of a potato leaf with early blight", "score": 0.38},
        {"prompt": "a photo of a grape leaf with black rot", "score": 0.37},
    ]
    labels = ["Pepper__bell___healthy", "Pepper__bell___Bacterial_spot"]

    prompt = build_llm_prompt(blip_cap, clip_preds, labels)
    print(prompt)
