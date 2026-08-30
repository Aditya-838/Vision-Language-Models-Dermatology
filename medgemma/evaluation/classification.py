"""Classification and free-text evaluation from the MedGemma notebooks."""

import re
import pandas as pd
from research_utils import build_disease_lookups


def parse_output(text):
    disease = ""
    for line in str(text).split("\n"):
        if line.strip().lower().startswith("disease:"):
            disease = line.strip().split(":", 1)[1].strip()
    return disease


def evaluate_finetuned(inference_csv, metadata_csv, results_csv=None):
    from sklearn.metrics import f1_score, precision_score, recall_score
    df = pd.read_csv(inference_csv)
    disease_to_main, disease_to_sub = build_disease_lookups(metadata_csv)
    df["pred_disease"] = df["model_output"].apply(parse_output)
    df["pred_main_class"] = df["pred_disease"].str.lower().map(disease_to_main).fillna("Unknown")
    df["pred_sub_class"] = df["pred_disease"].str.lower().map(disease_to_sub).fillna("Unknown")
    metrics = {}
    for level, truth, prediction in [("main", "gt_main_class", "pred_main_class"),
        ("sub", "gt_sub_class", "pred_sub_class"), ("disease", "gt_disease", "pred_disease")]:
        metrics[level] = {"accuracy": (df[truth] == df[prediction]).mean() * 100,
            "f1_weighted": f1_score(df[truth], df[prediction], average="weighted", zero_division=0) * 100,
            "precision_weighted": precision_score(df[truth], df[prediction], average="weighted", zero_division=0) * 100,
            "recall_weighted": recall_score(df[truth], df[prediction], average="weighted", zero_division=0) * 100}
    if results_csv:
        df.to_csv(results_csv, index=False)
    return df, metrics


def save_confusion_matrices(df, output_dir):
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.metrics import confusion_matrix
    labels = sorted(df["gt_main_class"].unique())
    matrix = confusion_matrix(df["gt_main_class"], df["pred_main_class"], labels=labels)
    for normalized, filename, title in [(False, "confusion_matrix_finetuned.png", "Fine-Tuned MedGemma — Confusion Matrix (Main Class)"),
        (True, "confusion_matrix_finetuned_normalized.png", "Fine-Tuned MedGemma - Normalized Confusion Matrix")]:
        shown = matrix.astype(float) / matrix.sum(axis=1)[:, np.newaxis] if normalized else matrix
        fig, axis = plt.subplots(figsize=(12, 10)); image = axis.imshow(shown, cmap=plt.cm.Blues)
        plt.colorbar(image, ax=axis); axis.set_xticks(range(len(labels))); axis.set_yticks(range(len(labels)))
        axis.set_xticklabels(labels, rotation=45, ha="right"); axis.set_yticklabels(labels)
        for i in range(len(labels)):
            for j in range(len(labels)):
                value = f"{shown[i, j]:.2f}" if normalized else str(shown[i, j])
                axis.text(j, i, value, ha="center", va="center")
        axis.set_xlabel("Predicted"); axis.set_ylabel("Ground Truth"); axis.set_title(title)
        plt.tight_layout(); plt.savefig(f"{output_dir}/{filename}", dpi=150); plt.close(fig)


def fst_fairness(df):
    rows = []
    for fst in ["FST 3", "FST 4", "FST 5", "FST 6"]:
        subset = df[df["fitzpatrick"] == fst]
        correct = (subset["pred_main_class"] == subset["gt_main_class"]).sum()
        rows.append({"FST": fst, "Images": len(subset), "Correct": correct,
            "Accuracy": correct / len(subset) * 100 if len(subset) else 0})
    return pd.DataFrame(rows)


def is_negated(word, text):
    words = text.lower().split()
    if word.lower() not in words:
        return False
    return any(negation in words[max(0, words.index(word.lower()) - 3):words.index(word.lower())]
        for negation in ["no", "not", "without", "absent", "none", "negative", "free", "neither"])


def evaluate_freetext(output_csv, results_csv=None):
    from thefuzz import fuzz
    rows, insufficient = [], 0
    for row in pd.read_csv(output_csv).to_dict("records"):
        text = str(row["generated_text"]).lower()
        if int(row["word_count"]) < 20:
            insufficient += 1; continue
        descriptors = [item.strip() for item in str(row["descriptors"]).split(",") if item.strip()]
        matched = [word in text and not is_negated(word, text) for word in
            [item.lower().split("(")[0].strip() for item in descriptors]]
        body_parts = [item.strip().lower() for item in str(row["body_part"]).split(",") if item.strip()]
        rows.append({"image_name": row["image_name"], "fitzpatrick": row["fitzpatrick"],
            "true_disease": row["true_disease"], "true_main_class": row["true_main_class"],
            "disease_recognized": fuzz.partial_ratio(str(row["true_disease"]).lower(), text) > 80,
            "descriptor_recall": round(sum(matched) / len(descriptors), 3) if descriptors else 0.0,
            "body_part_match": any(part.split("(")[0].strip() in text and not is_negated(part.split("(")[0].strip(), text)
                for part in body_parts)})
    result = pd.DataFrame(rows)
    if results_csv: result.to_csv(results_csv, index=False)
    return result, insufficient


def extract_freetext_disease(text):
    for pattern in [r"Most Likely (?:Skin )?Condition:\s*([^\n.]+)",
        r"most likely (?:skin )?condition is\s+([^\n.]+)", r"most likely diagnosis is\s+([^\n.]+)"]:
        match = re.search(pattern, str(text), re.IGNORECASE)
        if match: return match.group(1).strip().rstrip(".")
    return ""
