"""MedGemma QLoRA training components extracted from medgemma-finetuned.ipynb."""

import os
from research_utils import read_prompt

PROMPT = read_prompt("fine_tuning_validation.txt")


def fix_path(image_path, dataset_0, dataset_1):
    filename = os.path.basename(image_path)
    for base in [dataset_0, dataset_1]:
        candidate = os.path.join(base, filename)
        if os.path.exists(candidate):
            return candidate
    return None


def load_training_model(model_id, rank=16, alpha=32, dropout=0.05):
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_storage=torch.bfloat16)
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(model_id, quantization_config=bnb_config,
        attn_implementation="flash_attention_2", dtype=torch.bfloat16, device_map="auto")
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=dropout,
        target_modules="all-linear", task_type="CAUSAL_LM"))
    return model, processor


def build_datasets(train_data, val_data):
    import torchvision.transforms as transforms
    from PIL import Image
    from torch.utils.data import Dataset

    train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3), transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05),
        transforms.RandomResizedCrop(size=(896, 896), scale=(0.85, 1.0))])
    val_transforms = transforms.Resize((896, 896))
    class DermaDataset(Dataset):
        def __init__(self, data, is_training=True): self.data, self.is_training = data, is_training
        def __len__(self): return len(self.data)
        def __getitem__(self, index):
            item = self.data[index]
            image = Image.open(item["image_path"]).convert("RGB")
            image = train_transforms(image) if self.is_training else val_transforms(image)
            answer = f"Disease: {item['disease']}\nCategory: {item['main_class']}\nSub-class: {item['sub_class']}"
            return {"images": [image], "messages": [{"role": "user", "content": [{"type": "image"},
                {"type": "text", "text": PROMPT}]}, {"role": "assistant", "content": [{"type": "text", "text": answer}]}]}
    return DermaDataset(train_data), DermaDataset(val_data, False), val_transforms


def build_trainer(model, processor, train_dataset, val_dataset, output_dir):
    from trl import SFTConfig, SFTTrainer
    config = SFTConfig(output_dir=output_dir, num_train_epochs=2, per_device_train_batch_size=1,
        gradient_accumulation_steps=4, per_device_eval_batch_size=1, eval_strategy="epoch",
        save_strategy="epoch", save_steps=100, load_best_model_at_end=True, metric_for_best_model="eval_loss",
        learning_rate=2e-4, lr_scheduler_type="cosine", warmup_ratio=0.03, max_grad_norm=0.3, bf16=True,
        logging_steps=10, report_to="wandb", remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True}, max_length=None)
    return SFTTrainer(model=model, args=config, train_dataset=train_dataset, eval_dataset=val_dataset,
        processing_class=processor)
