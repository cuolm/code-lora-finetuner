import argparse
import gc
import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import torch
from datasets import IterableDataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from .config import Config


logger = logging.getLogger("src.finetune.train")


class FIMDataCollator:
    def __init__(self, tokenizer, label_pad_token_id):
        self.tokenizer = tokenizer
        self.label_pad_token_id = label_pad_token_id 

    def __call__(self, examples: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        # Find max length for this specific batch.
        max_ex_length = 0 
        for ex in examples:
            if len(ex["input_ids"]) > max_ex_length:
                max_ex_length = len(ex["input_ids"])
        
        # Apply padding. 
        padded_examples = [] 
        for ex in examples:
            pad_length = max_ex_length - len(ex["input_ids"])
            padded_ex = {  
                "input_ids": (ex["input_ids"] + [self.tokenizer.pad_token_id] * pad_length),  
                "attention_mask": (ex["attention_mask"] + [0] * pad_length),  
                "labels": (ex["labels"] + [self.label_pad_token_id] * pad_length)  
            }  
            padded_examples.append(padded_ex)

        # Stack separate examples into a single tensors matrix.
        examples_batch = {
            "input_ids": torch.tensor([ex["input_ids"] for ex in padded_examples], dtype=torch.long),
            "attention_mask": torch.tensor([ex["attention_mask"] for ex in padded_examples], dtype=torch.long),
            "labels": torch.tensor([ex["labels"] for ex in padded_examples], dtype=torch.long)
        }

        return examples_batch


def train_and_save_lora_model(
        config: Config,
        lora_model: AutoModelForCausalLM,
        train_dataset: IterableDataset,
        eval_dataset: IterableDataset,
        user_args: argparse.Namespace
) -> List: 
    logger.info(f"Starting training: {config.trainer_max_steps} steps on {config.device}, batch_size={config.trainer_per_device_train_batch_size}") 

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = config.fim_pad_token
    tokenizer.padding_side = "right"

    training_args = TrainingArguments(
        output_dir=config.trainer_output_dir_path,
        per_device_train_batch_size=config.trainer_per_device_train_batch_size,
        per_device_eval_batch_size=config.trainer_per_device_eval_batch_size,
        gradient_accumulation_steps=config.trainer_gradient_accumulation_steps, # simulate a batch size of 4 but only load 1 example at a time in memory.
        learning_rate=config.trainer_learning_rate,
        weight_decay=config.trainer_weight_decay,
        max_grad_norm=config.trainer_max_grad_norm,
        max_steps=config.trainer_max_steps,
        lr_scheduler_type=config.trainer_lr_scheduler_type,
        warmup_steps=config.trainer_warmup_steps,
        log_level=config.trainer_log_level,
        eval_strategy=config.trainer_eval_strategy,
        logging_steps=config.trainer_logging_steps,
        eval_steps=config.trainer_eval_steps,
        logging_strategy=config.trainer_logging_strategy,
        save_strategy=config.trainer_save_strategy,
        save_steps=config.trainer_save_steps,
        bf16=config.trainer_bf16,  # if set to True fp16 needs to be set to false 
        fp16=config.trainer_fp16,  # if set to True bf16 needs to be set to False
        gradient_checkpointing=config.trainer_gradient_checkpointing
    )

    data_collator = FIMDataCollator(
        tokenizer=tokenizer,
        label_pad_token_id=config.collator_label_pad_token_id
    )
    
    trainer = Trainer(
        model=lora_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class = tokenizer,
        data_collator = data_collator
    )

    if user_args.resume == "last":
        trainer.train(resume_from_checkpoint=True)
    elif user_args.resume is not None:
        trainer.train(resume_from_checkpoint=user_args.resume) 
    else:
        trainer.train() # train from scratch

    lora_model.save_pretrained(config.lora_adapter_path) # save lora adapter only
    log_history = trainer.state.log_history

    if config.device == "cuda":
        # Clear 4-bit model from VRAM to make room for the FP16 base model.
        if 'trainer' in locals():
            del trainer  # clear trainer references

        del lora_model  # remove model reference

        gc.collect()  # force garbage collection

        if torch.cuda.is_available():
            torch.cuda.synchronize()  # wait for gpu
            torch.cuda.empty_cache()  # clear gpu cache
        
        # Ensure offload folder for merging exists before loading the model.
        config.trainer_model_merge_offload_folder_path.mkdir(parents=True, exist_ok=True)

        # Load fresh FP16 base model.
        base_model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=config.model_name,
            dtype=torch.float16,
            device_map="auto",
            offload_folder=str(config.trainer_model_merge_offload_folder_path), # Offload model layers to disk during loading to prevent RAM (OOM) crashes. 
            low_cpu_mem_usage=True
        )

        lora_model = PeftModel.from_pretrained(
            base_model, 
            config.lora_adapter_path,
            offload_folder=str(config.trainer_model_merge_offload_folder_path)
        )

    # Merge lora adapter into base model and save it with the tokenizer of the model.
    merged_model= lora_model.merge_and_unload() 
    merged_model = merged_model.to(torch.float16) 
    merged_model.save_pretrained(config.lora_model_path)
    tokenizer.save_pretrained(config.lora_model_path)

    # Clean up offload folder.
    if config.trainer_model_merge_offload_folder_path.exists():
        shutil.rmtree(config.trainer_model_merge_offload_folder_path)

    return log_history 


def save_log(config: Config, log_history: List) -> None:
    history = {
        "train": {"steps": [], "loss": [], "learning_rate": [], "epoch": []},
        "eval": {"steps": [], "loss": [], "epoch": []}
    }

    for entry in log_history:
        # Training logs
        if "loss" in entry:
            history["train"]["loss"].append(entry["loss"])
            history["train"]["steps"].append(entry["step"])
            history["train"]["epoch"].append(entry.get("epoch"))
            history["train"]["learning_rate"].append(entry.get("learning_rate"))
        
        # Evaluation logs
        elif "eval_loss" in entry:
            history["eval"]["loss"].append(entry["eval_loss"])
            history["eval"]["steps"].append(entry["step"])
            history["eval"]["epoch"].append(entry.get("epoch"))

    log_path = Path(config.trainer_output_dir_path) / "training_log.json"
    with log_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


def plot_loss(config: Config) -> None:
    losses_path = Path(config.trainer_output_dir_path) / "training_log.json"
    with losses_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    plt.figure(figsize=(8, 5))
    
    if data["train"]["steps"]:
        plt.plot(data["train"]["steps"], data["train"]["loss"], label="Train Loss")
    
    if data["eval"]["steps"]:
        plt.plot(data["eval"]["steps"], data["eval"]["loss"], label="Eval Loss", marker='o')

    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training and Evaluation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(Path(config.trainer_output_dir_path) / "loss_plot.png")
    plt.show()
