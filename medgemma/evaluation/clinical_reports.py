"""Mode-B clinical-report evaluation extracted from medgemma-metrics.ipynb."""

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
    messages = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_B}]},
        {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": PROMPT_B}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=text, images=image, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=250, do_sample=False, repetition_penalty=1.5,
            no_repeat_ngram_size=3, eos_token_id=processor.tokenizer.eos_token_id)
    return clean_output(processor.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))


def build_pseudo_reference(row):
    return (f"This skin condition is {row['gt_disease']}, categorized under {row['main_class']}. "
        f"Visual features include {row['gt_descriptors']}. Location: {row['gt_body_part']}.")


def compute_text_metrics(mode_b_csv, output_csv=None):
    from rouge_score import rouge_scorer
    from nltk.translate.meteor_score import meteor_score
    from nltk.tokenize import word_tokenize
    from sacrebleu.metrics import BLEU
    frame = pd.read_csv(mode_b_csv); frame["pseudo_reference"] = frame.apply(build_pseudo_reference, axis=1)
    scorer, bleu = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True), BLEU(effective_order=True)
    scores = [scorer.score(row.pseudo_reference, str(row.generated_text)) for _, row in frame.iterrows()]
    frame["rouge1"] = [score["rouge1"].fmeasure for score in scores]; frame["rougeL"] = [score["rougeL"].fmeasure for score in scores]
    frame["meteor"] = [meteor_score([word_tokenize(row.pseudo_reference.lower())], word_tokenize(str(row.generated_text).lower())) for _, row in frame.iterrows()]
    frame["bleu"] = [bleu.sentence_score(str(row.generated_text), [row.pseudo_reference]).score / 100 for _, row in frame.iterrows()]
    if output_csv: frame.to_csv(output_csv, index=False)
    return frame


def create_anthropic_client():
    import anthropic
    return anthropic.Anthropic(api_key=require_env("ANTHROPIC_API_KEY"))
