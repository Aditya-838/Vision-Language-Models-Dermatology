"""Qwen zero-shot workflows extracted from qwen-zeroshot.ipynb."""

import csv
import os
from research_utils import read_prompt, remap_kaggle_dataset_path

STRUCTURED_SYSTEM = "You are a helpful dermatology assistant."
STRUCTURED_PROMPT = read_prompt("structured_classification.txt")
FREETEXT_SYSTEM = read_prompt("zero_shot_free_text_system.txt")
FREETEXT_PROMPT = "What do you observe in this skin image?"


def prepare_validation_paths(val_data, image_base):
    for pair in val_data:
        pair["image_path"] = remap_kaggle_dataset_path(pair["image_path"], image_base)
    return val_data


def load_model(model_id, hf_token=None, min_pixels=256 * 28 * 28, max_pixels=1280 * 28 * 28):
    import torch
    from huggingface_hub import login
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
    if hf_token: login(token=hf_token)
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_storage=torch.bfloat16)
    processor = AutoProcessor.from_pretrained(model_id, min_pixels=min_pixels, max_pixels=max_pixels)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, quantization_config=quantization,
        torch_dtype=torch.bfloat16, device_map="auto")
    model.eval(); return model, processor


def _generate(messages, model, processor, max_tokens):
    import torch
    from qwen_vl_utils import process_vision_info
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, return_tensors="pt", padding=True).to(model.device)
    with torch.inference_mode(): output = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
    return processor.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def run_structured_inference(val_data, model, processor, output_csv, max_tokens=100, checkpoint_every=100):
    from PIL import Image
    from tqdm import tqdm
    processed = set()
    if os.path.exists(output_csv):
        with open(output_csv, encoding="utf-8") as handle: processed = {row["image_name"] for row in csv.DictReader(handle)}
    fields = ["image_name", "true_main_class", "true_sub_class", "true_disease", "fitzpatrick", "generated_text"]
    with open(output_csv, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not processed: writer.writeheader()
        for index, pair in enumerate(tqdm(val_data, desc="Qwen ZS-A Structured")):
            if pair["image_name"] in processed: continue
            try:
                image = Image.open(pair["image_path"]).convert("RGB")
                text = _generate([{"role": "system", "content": [{"type": "text", "text": STRUCTURED_SYSTEM}]},
                    {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": STRUCTURED_PROMPT}]}], model, processor, max_tokens)
            except Exception as exc: text = f"ERROR: {exc}"
            writer.writerow({"image_name": pair["image_name"], "true_main_class": pair["main_class"], "true_sub_class": pair["sub_class"], "true_disease": pair["disease"], "fitzpatrick": pair["fitzpatrick"], "generated_text": text})
            if (index + 1) % checkpoint_every == 0: handle.flush()


def run_freetext_inference(val_data, model, processor, output_csv, clean_output, max_tokens=300, checkpoint_every=100):
    """Uses the same free-text cleaning function as the completed MedGemma notebook."""
    from PIL import Image
    from tqdm import tqdm
    processed = set()
    if os.path.exists(output_csv):
        with open(output_csv, encoding="utf-8") as handle: processed = {row["image_name"] for row in csv.DictReader(handle)}
    fields = ["image_name", "true_main_class", "true_sub_class", "true_disease", "fitzpatrick", "descriptors", "body_part", "generated_text", "word_count"]
    with open(output_csv, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not processed: writer.writeheader()
        for index, pair in enumerate(tqdm(val_data, desc="Qwen ZS-B Free-Text")):
            if pair["image_name"] in processed: continue
            try:
                image = Image.open(pair["image_path"]).convert("RGB")
                text = clean_output(_generate([{"role": "system", "content": [{"type": "text", "text": FREETEXT_SYSTEM}]}, {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": FREETEXT_PROMPT}]}], model, processor, max_tokens))
            except Exception as exc: text = f"ERROR: {exc}"
            writer.writerow({"image_name": pair["image_name"], "true_main_class": pair["main_class"], "true_sub_class": pair["sub_class"], "true_disease": pair["disease"], "fitzpatrick": pair["fitzpatrick"], "descriptors": pair["descriptors"], "body_part": pair["body_part"], "generated_text": text, "word_count": len(text.split())})
            if (index + 1) % checkpoint_every == 0: handle.flush()
