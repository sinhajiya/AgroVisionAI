# agent/llm_agent.py

import os
import json
from typing import Dict, Any, List

from models.vision_features import extract_visual_features
from agent.llm_prompt_builder import build_llm_prompt
from models.labels_to_prompts import load_prompts   # you already had label utilities
# from agro.models.vision_features import extract_visual_features
# from agro.agent.llm_prompt_builder import build_llm_prompt
# from agro.data.labels import load_dataset_labels


def call_llm(prompt: str) -> str:
    """
    Local CPU inference using Phi-3-mini (instruction-tuned).
    Much more reliable for JSON outputs than flan-t5-small.
    """

    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    model_name = "microsoft/Phi-3-mini-4k-instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,   # CPU-safe
        device_map="cpu"
    )

    # This is important: stupid small models need a system instruction wrapper
    messages = [
        {"role": "system", "content": "You are a precise JSON-generating assistant."},
        {"role": "user", "content": prompt}
    ]

    # Convert to chat template (Phi-3 follows ChatML)
    enc = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(enc, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=700,
            temperature=0.0,       # deterministic, essential for JSON
            do_sample=False
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Strip off the prompt (ChatML models echo)
    if "assistant" in decoded:
        decoded = decoded.split("assistant")[-1].strip()

    return decoded


def run_diagnosis(
    image_path: str,
    data_dir: str,
    rag_context: str = ""
) -> Dict[str, Any]:
    """
    Main pipeline:
    1. Extract visual features (BLIP + CLIP)
    2. Build prompt
    3. Call LLM
    4. Return dict
    """

    # Step 1: Vision features
    features = extract_visual_features(image_path, data_dir)
    blip_caption = features["blip_caption"]
    clip_preds = features["clip_predictions"]

    # Step 2: Load label space from dataset folders
    all_labels = load_prompts(data_dir)

    # Step 3: Build LLM prompt
    prompt = build_llm_prompt(
        blip_caption=blip_caption,
        clip_predictions=clip_preds,
        all_labels=all_labels,
        context_docs=rag_context,
        k=5
    )

    # Step 4: LLM reasoning
    llm_raw = call_llm(prompt)

    # Try to parse JSON
    try:
        parsed = json.loads(llm_raw)
    except json.JSONDecodeError:
        parsed = {"error": "LLM did not return valid JSON", "raw_output": llm_raw}

    return {
        "blip_caption": blip_caption,
        "clip_predictions": clip_preds,
        "llm_output": parsed
    }


if __name__ == "__main__":
    # Quick command line test

    test_image = "/home/iiserb/Documents/agro/dataset/PlantVillage/Pepper__bell___healthy/0ade14b6-8937-43ea-93eb-98343af6bae7___JR_HL 8026.JPG"
    data_dir = "./dataset/archive(1)/plantvillage dataset/color"

    result = run_diagnosis(test_image, data_dir)

    print("\n===== BLIP Caption =====")
    print(result["blip_caption"])

    print("\n===== CLIP Predictions =====")
    for r in result["clip_predictions"]:
        print(f"- {r['prompt']} ({r['score']:.4f})")

    print("\n===== LLM Diagnosis =====")
    print(json.dumps(result["llm_output"], indent=2))
