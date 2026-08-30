"""Small shared helpers extracted from the completed experiment notebooks."""

from __future__ import annotations

import json
import os
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parent


def read_prompt(name: str) -> str:
    """Load an archival prompt without altering its wording."""
    return (REPOSITORY_ROOT / "prompts" / name).read_text(encoding="utf-8")


def load_json(path: str | Path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def build_disease_lookups(metadata_csv: str | Path):
    """Build the disease-to-main/sub-class mappings used in all notebooks."""
    import pandas as pd

    metadata = pd.read_csv(metadata_csv)
    disease_to_main, disease_to_sub = {}, {}
    for _, row in metadata.iterrows():
        disease = str(row["Disease_label"]).strip().lower()
        disease_to_main[disease] = str(row["Main_class"]).strip()
        disease_to_sub[disease] = str(row["Sub_class"]).strip()
    return disease_to_main, disease_to_sub


def require_env(name: str) -> str:
    """Get a credential from the environment; never embed it in source code."""
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Set the {name} environment variable before running this workflow.")
    return value


def remap_kaggle_dataset_path(image_path: str, image_base: str) -> str:
    """Preserve the Kaggle-to-Modal path conversion from the zero-shot notebooks."""
    prefix = "/kaggle/input/datasets/adityavarmapenmetsa/indianskincon/indianskindata/DATASET"
    return image_path.replace(prefix, image_base)
