"""Qwen classification evaluation; metric definitions are shared with MedGemma."""

from medgemma.evaluation.classification import (
    evaluate_freetext,
    extract_freetext_disease,
    fst_fairness,
    is_negated,
)
from research_utils import build_disease_lookups


def parse_output(text):
    disease = ""
    for line in str(text).split("\n"):
        if line.strip().lower().startswith("disease:"):
            disease = line.strip().split(":", 1)[1].strip()
    return disease


def evaluate_finetuned(inference_csv, metadata_csv, results_csv=None):
    """Preserves Qwen's true_* inference columns and exact-match evaluation."""
    import pandas as pd
    from sklearn.metrics import f1_score, precision_score, recall_score
    frame = pd.read_csv(inference_csv)
    disease_to_main, disease_to_sub = build_disease_lookups(metadata_csv)
    frame["pred_disease"] = frame["generated_text"].apply(parse_output)
    frame["pred_main_class"] = frame["pred_disease"].str.lower().map(disease_to_main).fillna("Unknown")
    frame["pred_sub_class"] = frame["pred_disease"].str.lower().map(disease_to_sub).fillna("Unknown")
    metrics = {}
    for level, truth, prediction in [("main", "true_main_class", "pred_main_class"),
        ("sub", "true_sub_class", "pred_sub_class"), ("disease", "true_disease", "pred_disease")]:
        metrics[level] = {"accuracy": (frame[truth] == frame[prediction]).mean() * 100,
            "f1_weighted": f1_score(frame[truth], frame[prediction], average="weighted", zero_division=0) * 100,
            "precision_weighted": precision_score(frame[truth], frame[prediction], average="weighted", zero_division=0) * 100,
            "recall_weighted": recall_score(frame[truth], frame[prediction], average="weighted", zero_division=0) * 100}
    if results_csv: frame.to_csv(results_csv, index=False)
    return frame, metrics


def final_comparison_inputs():
    """Names of the eight completed-result CSVs read by qwen-metrics.ipynb."""
    return ["medgemma_zeroshot_structured_results.csv", "medgemma_finetuned_results.csv",
        "qwen_zeroshot_structured_results.csv", "qwen_finetuned_results.csv",
        "claude_judge_results.csv", "qwen_claude_judge_results.csv",
        "modeB_full_evaluation.csv", "qwen_modeB_full_evaluation.csv"]
