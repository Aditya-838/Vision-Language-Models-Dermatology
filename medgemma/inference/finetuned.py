"""Fine-tuned MedGemma validation inference extracted from the thesis notebook."""

import csv
from research_utils import read_prompt

INFERENCE_PROMPT = read_prompt("fine_tuning_validation.txt")


def run_inference(val_data, model, processor, output_csv, val_transforms, checkpoint_every=100):
    import torch
    from PIL import Image
    import pandas as pd

    model.eval()
    results = []
    for i, item in enumerate(val_data):
        try:
            image = val_transforms(Image.open(item["image_path"]).convert("RGB"))
            messages = [{"role": "user", "content": [{"type": "image", "image": image},
                {"type": "text", "text": INFERENCE_PROMPT}]}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=text, images=image, return_tensors="pt").to(model.device)
            with torch.no_grad():
                output_ids = model.generate(**inputs, max_new_tokens=100, do_sample=False)
            generated = processor.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        except Exception as exc:
            generated = f"ERROR: {exc}"
        results.append({"image_name": item["image_name"], "fitzpatrick": item["fitzpatrick"],
            "gt_disease": item["disease"], "gt_main_class": item["main_class"],
            "gt_sub_class": item["sub_class"], "model_output": generated})
        if (i + 1) % checkpoint_every == 0:
            pd.DataFrame(results).to_csv(output_csv, index=False)
    pd.DataFrame(results).to_csv(output_csv, index=False)
