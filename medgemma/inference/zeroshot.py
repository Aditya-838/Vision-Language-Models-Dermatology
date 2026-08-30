"""MedGemma zero-shot workflows extracted from medgemma-zeroshot(1).ipynb."""

import csv
import os
import re
from pathlib import Path

from research_utils import read_prompt, remap_kaggle_dataset_path

STRUCTURED_SYSTEM = "You are a helpful dermatology assistant."
STRUCTURED_PROMPT = read_prompt("structured_classification.txt")
FREETEXT_SYSTEM = read_prompt("zero_shot_free_text_system.txt")
FREETEXT_PROMPT = "What do you observe in this skin image?"
NEGATION_WORDS = ["no", "not", "without", "absent", "none", "negative", "free", "neither"]


def prepare_validation_paths(val_data, image_base):
    for pair in val_data:
        pair["image_path"] = remap_kaggle_dataset_path(pair["image_path"], image_base)
    return val_data


def load_model(model_id, hf_token=None):
    import torch
    from huggingface_hub import login
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

    if hf_token:
        login(token=hf_token)
    quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_storage=torch.bfloat16)
    model = AutoModelForImageTextToText.from_pretrained(model_id, quantization_config=quantization_config,
        attn_implementation="flash_attention_2", dtype=torch.bfloat16, device_map="auto")
    processor = AutoProcessor.from_pretrained(model_id)
    model.eval()
    return model, processor


def run_structured_inference(val_data, model, processor, output_csv, max_tokens=100, checkpoint_every=100):
    """The notebook's resumable structured zero-shot loop."""
    import torch
    from PIL import Image
    from tqdm import tqdm

    processed = set()
    if os.path.exists(output_csv):
        with open(output_csv, encoding="utf-8") as handle:
            processed = {row["image_name"] for row in csv.DictReader(handle)}
    with open(output_csv, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_name", "true_main_class", "true_sub_class",
            "true_disease", "fitzpatrick", "generated_text"])
        if not processed:
            writer.writeheader()
        for i, pair in enumerate(tqdm(val_data, desc="Zero-Shot Structured")):
            if pair["image_name"] in processed:
                continue
            try:
                image = Image.open(pair["image_path"]).convert("RGB")
                messages = [{"role": "system", "content": [{"type": "text", "text": STRUCTURED_SYSTEM}]},
                    {"role": "user", "content": [{"type": "image", "image": image},
                    {"type": "text", "text": STRUCTURED_PROMPT}]}]
                inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                    return_dict=True, return_tensors="pt").to(model.device, dtype=torch.bfloat16)
                with torch.inference_mode():
                    output_ids = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
                generated_text = processor.decode(output_ids[0][inputs["input_ids"].shape[-1]:],
                    skip_special_tokens=True).strip()
            except Exception as exc:  # preserved notebook behavior
                generated_text = f"ERROR: {exc}"
            writer.writerow({"image_name": pair["image_name"], "true_main_class": pair["main_class"],
                "true_sub_class": pair["sub_class"], "true_disease": pair["disease"],
                "fitzpatrick": pair["fitzpatrick"], "generated_text": generated_text})
            if (i + 1) % checkpoint_every == 0:
                handle.flush()


def clean_output(text):
    text = re.sub(r"<unused\d+>.*?</unused\d+>", "", text, flags=re.DOTALL)
    text = re.sub(r"<unused\d+>.*?(?=\n\n|\Z)", "", text, flags=re.DOTALL)
    if "thinking process" in text.lower() or "here's a thinking" in text.lower():
        index = text.find("\n\n", 200)
        if index != -1:
            text = text[index:].strip()
    for opening in ["Okay, I can help with that. Here's a description of the visible lesion in the image:",
        "Based on the image, here's a description of the visible lesion:"]:
        text = text.replace(opening, "")
    for pattern in [r"Disclaimer:.*?(?=\n\n|\Z)", r"Important Note:.*?(?=\n\n|\Z)",
        r"Please consult.*?(?=\n\n|\Z)", r"I cannot provide a diagnosis.*?(?=\n\n|\Z)"]:
        text = re.sub(pattern, "", text, flags=re.DOTALL)
    text = re.sub(r"^\*\s+", "", text.replace("**", ""), flags=re.MULTILINE).strip()
    if text and not text.endswith("."):
        text += "."
    return re.sub(r"\n{3,}", "\n\n", text + "\n\nNote: A definitive diagnosis requires a clinical examination by a healthcare professional.").strip()


def run_freetext_inference(val_data, model, processor, output_csv, max_tokens=300, checkpoint_every=100):
    import torch
    from PIL import Image
    from tqdm import tqdm

    processed = set()
    if os.path.exists(output_csv):
        with open(output_csv, encoding="utf-8") as handle:
            processed = {row["image_name"] for row in csv.DictReader(handle)}
    fields = ["image_name", "true_main_class", "true_sub_class", "true_disease", "fitzpatrick",
        "descriptors", "body_part", "generated_text", "word_count"]
    with open(output_csv, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not processed:
            writer.writeheader()
        for i, pair in enumerate(tqdm(val_data, desc="Zero-Shot Free-Text")):
            if pair["image_name"] in processed:
                continue
            try:
                image = Image.open(pair["image_path"]).convert("RGB")
                messages = [{"role": "system", "content": [{"type": "text", "text": FREETEXT_SYSTEM}]},
                    {"role": "user", "content": [{"type": "image", "image": image},
                    {"type": "text", "text": FREETEXT_PROMPT}]}]
                inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                    return_dict=True, return_tensors="pt").to(model.device, dtype=torch.bfloat16)
                with torch.inference_mode():
                    output_ids = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False,
                        eos_token_id=processor.tokenizer.eos_token_id)
                generated_text = clean_output(processor.decode(output_ids[0][inputs["input_ids"].shape[-1]:],
                    skip_special_tokens=True).strip())
            except Exception as exc:
                generated_text = f"ERROR: {exc}"
            writer.writerow({"image_name": pair["image_name"], "true_main_class": pair["main_class"],
                "true_sub_class": pair["sub_class"], "true_disease": pair["disease"],
                "fitzpatrick": pair["fitzpatrick"], "descriptors": pair["descriptors"],
                "body_part": pair["body_part"], "generated_text": generated_text,
                "word_count": len(generated_text.split())})
            if (i + 1) % checkpoint_every == 0:
                handle.flush()
