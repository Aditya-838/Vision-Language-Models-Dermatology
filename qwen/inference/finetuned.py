"""Fine-tuned Qwen validation inference extracted from qwen-finetuned.ipynb."""

import csv
import os
from research_utils import read_prompt
INFERENCE_PROMPT = read_prompt("fine_tuning_validation.txt")


def run_inference(val_data, model, processor, output_csv, max_tokens=80, checkpoint_every=100):
    import torch
    from PIL import Image
    from qwen_vl_utils import process_vision_info
    from tqdm import tqdm
    processed = set()
    if os.path.exists(output_csv):
        with open(output_csv, encoding="utf-8") as handle: processed = {row["image_name"] for row in csv.DictReader(handle)}
    fields = ["image_name", "true_main_class", "true_sub_class", "true_disease", "fitzpatrick", "generated_text"]
    with open(output_csv, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not processed: writer.writeheader()
        model.eval()
        for index, pair in enumerate(tqdm(val_data, desc="Qwen FT Inference")):
            if pair["image_name"] in processed: continue
            try:
                image = Image.open(pair["image_path"]).convert("RGB")
                messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": INFERENCE_PROMPT}]}]
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = processor(text=[text], images=image_inputs, videos=video_inputs, return_tensors="pt", padding=True).to(model.device)
                with torch.inference_mode(): output = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
                generated = processor.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            except Exception as exc: generated = f"ERROR: {exc}"
            writer.writerow({"image_name": pair["image_name"], "true_main_class": pair["main_class"], "true_sub_class": pair["sub_class"], "true_disease": pair["disease"], "fitzpatrick": pair["fitzpatrick"], "generated_text": generated})
            if (index + 1) % checkpoint_every == 0: handle.flush()
