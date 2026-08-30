"""Gradio demo extracted from medgemma/demo.ipynb. It is not started automatically."""

import re
from research_utils import read_prompt, require_env

PROMPT_A = read_prompt("fine_tuning_validation.txt")
SYSTEM_B = read_prompt("clinical_report_system.txt")
PROMPT_B = read_prompt("clinical_report_user.txt")


def load_model(base_model_id, adapter_path):
    import torch
    from huggingface_hub import login
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
    login(token=require_env("HF_TOKEN"))
    config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_storage=torch.bfloat16)
    processor = AutoProcessor.from_pretrained(base_model_id)
    model = AutoModelForImageTextToText.from_pretrained(base_model_id, quantization_config=config,
        dtype=torch.bfloat16, device_map="auto")
    return PeftModel.from_pretrained(model, adapter_path).eval(), processor


def clean_output(text):
    return re.sub(r"\n{3,}", "\n\n", re.sub(r"[^\x00-\x7F]+", "", text)).strip()


def make_inference(model, processor):
    import torch
    def run_inference(image, mode):
        if image is None: return "Please upload a skin image."
        if mode == "Mode A: Structured Classification":
            max_tokens, messages = 80, [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": PROMPT_A}]}]
        else:
            max_tokens, messages = 400, [{"role": "system", "content": [{"type": "text", "text": SYSTEM_B}]}, {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": PROMPT_B}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=text, images=image, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False, repetition_penalty=1.5,
                no_repeat_ngram_size=3, eos_token_id=processor.tokenizer.eos_token_id)
        return clean_output(processor.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
    return run_inference


def create_demo(model, processor):
    import gradio as gr
    with gr.Blocks(title="DermaCon-IN MedGemma Demo") as app:
        gr.HTML("<h1>DermaCon-IN MedGemma Demo</h1><b>Disclaimer:</b> This tool is for research purposes only. Not for clinical use. A definitive diagnosis requires examination by a qualified healthcare professional.</div>")
        with gr.Row():
            with gr.Column():
                image = gr.Image(type="pil", label="Upload Skin Image")
                mode = gr.Radio(["Mode A: Structured Classification", "Mode B: Clinical Report"], value="Mode A: Structured Classification", label="Select Mode")
                submit = gr.Button("Analyse", variant="primary")
            with gr.Column(): output = gr.Textbox(label="Model Output", lines=15)
        mode.change(fn=lambda: "", inputs=[], outputs=output)
        submit.click(fn=make_inference(model, processor), inputs=[image, mode], outputs=output)
    return app
