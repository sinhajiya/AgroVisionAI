from typing import Optional
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

# Use BLIP large for better detail
BLIP_MODEL = "Salesforce/blip-image-captioning-large"
_model = None
_processor = None

def _load_model(device: Optional[str] = None):
    global _model, _processor
    if _model is not None and _processor is not None:
        return _model, _processor

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    _processor = BlipProcessor.from_pretrained(BLIP_MODEL)
    _model = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL).to(device)
    _model.eval()
    return _model, _processor

def caption_image(
    image: Image.Image,
    device: Optional[str] = None,
    max_length: int = 100,
    num_beams: int = 5,
    do_sample: bool = False,
    temperature: float = 1.0,
) -> str:

    model, processor = _load_model(device)

    inputs = processor(
        images=image,
        # text="Describe the visual appearance of this leaf in objective detail.",
        return_tensors="pt"
    ).to(model.device)


    gen_kwargs = {
        "max_length": max_length,
        "num_beams": num_beams,
        "do_sample": do_sample,
    }

    if do_sample:
        gen_kwargs["temperature"] = temperature

    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)

    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption


if __name__ == "__main__":

    import sys
    from pathlib import Path
    
    image_path = Path('/home/iiserb/Documents/agro/dataset/PlantVillage/Pepper__bell___healthy/0ade14b6-8937-43ea-93eb-98343af6bae7___JR_HL 8026.JPG')
    device = 'cpu'

    if not image_path.exists():
        print(f"Image not found: {image_path}", file=sys.stderr)
        sys.exit(1)

    img = Image.open(image_path).convert("RGB")
    cap = caption_image(img, device=device)
    print("BLIP Caption:", cap)
