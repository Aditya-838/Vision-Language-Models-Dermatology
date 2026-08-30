"""Qwen Mode-B report evaluation extracted from qwen-metrics.ipynb."""

import re
import pandas as pd
from research_utils import read_prompt, require_env

SYSTEM_B = read_prompt("clinical_report_system.txt")
PROMPT_B = read_prompt("clinical_report_user.txt")
JUDGE_RUBRIC = read_prompt("claude_judge_rubric.txt")


def select_stratified_validation_sample(val_data):
    frame = pd.DataFrame(val_data)
    return frame.groupby("main_class", group_keys=False).apply(
        lambda group: group.sample(n=min(10, len(group)), random_state=42)).reset_index(drop=True)


def clean_output(text):
    return re.sub(r"\n{3,}", "\n\n", re.sub(r"[^\x00-\x7F]+", "", text)).strip()


def run_mode_b(image, processor, model):
    import torch
    from qwen_vl_utils import process_vision_info
    messages = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_B}]},
        {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": PROMPT_B}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=250, do_sample=False, repetition_penalty=1.5,
            no_repeat_ngram_size=3, eos_token_id=processor.tokenizer.eos_token_id)
    return clean_output(processor.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))


def run_mode_b_batch(sample, processor, model, output_csv=None):
    """Run the notebook's Mode-B loop for its selected validation sample."""
    from PIL import Image
    from tqdm import tqdm

    results, failed = [], []
    for _, row in tqdm(sample.iterrows(), total=len(sample), desc="Mode B Inference"):
        try:
            generated_text = run_mode_b(Image.open(row["image_path"]).convert("RGB"), processor, model)
            results.append({"image_name": row["image_name"], "main_class": row["main_class"],
                "gt_disease": row["disease"], "gt_sub_class": row["sub_class"],
                "gt_descriptors": row["descriptors"], "gt_body_part": row["body_part"],
                "fitzpatrick": row["fitzpatrick"], "generated_text": generated_text})
        except Exception as exc:
            print(f"Failed: {row['image_name']} — {exc}")
            failed.append(row["image_name"])
    results_frame = pd.DataFrame(results)
    if output_csv:
        results_frame.to_csv(output_csv, index=False)
    return results_frame, failed


def build_pseudo_reference(row):
    return (f"This skin condition is {row['gt_disease']}, categorized under {row['main_class']}. "
        f"Visual features include {row['gt_descriptors']}. Location: {row['gt_body_part']}.")


def compute_text_metrics(mode_b_data, output_csv=None):
    """Apply the notebook's ROUGE, METEOR, and BLEU calculations to a CSV or DataFrame."""
    from rouge_score import rouge_scorer as rs
    from nltk.translate.meteor_score import meteor_score
    from nltk.tokenize import word_tokenize
    from sacrebleu.metrics import BLEU
    frame = mode_b_data.copy() if isinstance(mode_b_data, pd.DataFrame) else pd.read_csv(mode_b_data)
    frame["pseudo_reference"] = frame.apply(build_pseudo_reference, axis=1)
    scorer, bleu = rs.RougeScorer(["rouge1", "rougeL"], use_stemmer=True), BLEU(effective_order=True)
    rouge1_scores, rouge_l_scores, meteor_scores, bleu_scores = [], [], [], []
    for _, row in frame.iterrows():
        scores = scorer.score(row.pseudo_reference, str(row.generated_text))
        rouge1_scores.append(scores["rouge1"].fmeasure)
        rouge_l_scores.append(scores["rougeL"].fmeasure)
        meteor_scores.append(meteor_score([word_tokenize(row.pseudo_reference.lower())], word_tokenize(str(row.generated_text).lower())))
        bleu_scores.append(bleu.sentence_score(str(row.generated_text), [row.pseudo_reference]).score / 100)
    frame["rouge1"] = rouge1_scores
    frame["rougeL"] = rouge_l_scores
    frame["meteor"] = meteor_scores
    frame["bleu"] = bleu_scores
    if output_csv: frame.to_csv(output_csv, index=False)
    return frame


def create_anthropic_client():
    import anthropic
    return anthropic.Anthropic(api_key=require_env("ANTHROPIC_API_KEY"))


def encode_image(image_path, max_size_mb=4):
    """Encode images using the exact resizing and JPEG fallback from qwen-metrics.ipynb."""
    import base64
    import io
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    image.thumbnail((1568, 1568), Image.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    data = buffer.getvalue()
    if len(data) > max_size_mb * 1024 * 1024:
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=60)
        data = buffer.getvalue()
    return base64.standard_b64encode(data).decode("utf-8")


def judge_report(client, image_path, gt_disease, gt_descriptors, gt_body_part, generated_text):
    """Use the archived rubric and notebook's Claude Haiku request parameters."""
    import json

    prompt = JUDGE_RUBRIC.format(gt_disease=gt_disease, gt_descriptors=gt_descriptors,
        gt_body_part=gt_body_part, generated_text=generated_text)
    response = client.messages.create(model="claude-haiku-4-5-20251001", max_tokens=500,
        messages=[{"role": "user", "content": [{"type": "image", "source": {"type": "base64",
        "media_type": "image/jpeg", "data": encode_image(image_path)}}, {"type": "text", "text": prompt}]}])
    return json.loads(re.sub(r"```json|```", "", response.content[0].text.strip()).strip())


def run_claude_judge(mode_b_results, sample, client, output_csv=None):
    """Notebook judge loop; preserves failed-image handling without retries."""
    import time
    from tqdm import tqdm

    results, failed = [], []
    for _, row in tqdm(mode_b_results.iterrows(), total=len(mode_b_results), desc="Claude Judge"):
        try:
            image_path = sample[sample["image_name"] == row["image_name"]]["image_path"].values[0]
            scores = judge_report(client, image_path, row["gt_disease"], row["gt_descriptors"],
                row["gt_body_part"], row["generated_text"])
            results.append({"image_name": row["image_name"], "main_class": row["main_class"],
                "gt_disease": row["gt_disease"], "fitzpatrick": row["fitzpatrick"],
                "generated_text": row["generated_text"], "disease_score": scores["disease_score"],
                "visual_score": scores["visual_score"], "coherence_score": scores["coherence_score"],
                "comments": scores.get("comments", "")})
            time.sleep(0.5)
        except Exception as exc:
            print(f"Failed: {row['image_name']} — {exc}")
            failed.append(row["image_name"])
    results_frame = pd.DataFrame(results)
    if output_csv:
        results_frame.to_csv(output_csv, index=False)
    return results_frame, failed
