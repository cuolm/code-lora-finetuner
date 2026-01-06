import os
import json
from typing import Tuple, Dict
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset
from torch.utils.data import Dataset
from pathlib import Path
import matplotlib.pyplot as plt


def check_device() -> str:
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    return device
    

def evaluate_model(project_root_path: str, model: AutoModelForCausalLM, test_dataset: Dataset) -> Tuple[float, Dict] :
    """
    Evaluates the model on the test dataset using Masked Cross-Entropy Loss.

    The function computes the evaluation loss only over the target tokens 
    (the middle section of the FIM task, where labels != -100). It then 
    calculates and returns the Focused Perplexity (e^Loss) to measure 
    how well the model predicts the missing code.

    Perplexity represents the model's uncertainty as the effective number of words (or choices) 
    it considers equally likely for every word it predicts.
    A PP of 1.0 is the ideal minimum (meaning the model always predicts correctly). 
    Example: If PP=10, the model is as uncertain as if it chose from 10 words. 
    If PP=50, the model is 5 times more uncertain. Lower PP is always better.
    """
    eval_args = TrainingArguments(
        output_dir=f"{project_root_path}./eval_results", 
        per_device_eval_batch_size=5, 
    )
    
    trainer = Trainer(
        model = model,
        args = eval_args,
        eval_dataset= test_dataset
    )

    results = trainer.evaluate()
    perplexity = np.exp(results["eval_loss"])
    return perplexity, results

def plot_loss(logging_steps: int, project_root_path: str) -> None:
    losses_path = os.path.join(f"{project_root_path}/results", "training_log.json")
    with open(losses_path, "r") as f:
        data = json.load(f)

    train_losses = data["train_losses"]
    eval_losses = data["eval_losses"]
    train_x = [i * logging_steps for i in range(len(train_losses))]
    eval_x = [i * logging_steps for i in range(len(eval_losses))]
    plt.figure(figsize=(8,5))
    plt.plot(train_x, train_losses, label="Train Loss")
    plt.plot(eval_x, eval_losses, label="Eval Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training and Evaluation Loss")
    plt.legend()
    plt.grid(True)
    plt.show()


def empty_device_cache(device: str) -> None:
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()
    # cpu: nothing to do
    

def main():
    project_root_path = Path(__file__).resolve().parent.parent
    device = check_device()
    test_dataset = load_dataset("json", data_files=f"{project_root_path}/datasets/test_dataset.jsonl")["train"]

    #plot_loss(20, project_root_path)

    base_model_name = "Qwen/Qwen2.5-Coder-0.5B"
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name).to(device)
    base_model_perplexity, base_model_results = evaluate_model(project_root_path, base_model, test_dataset)
    print(base_model_perplexity)
    print(f"Base model perplexity: {base_model_perplexity}")
    print(f"Base model results: {base_model_results}")

    del base_model
    empty_device_cache(device)

    lora_model_name = f"{project_root_path}/lora_model"
    lora_model = AutoModelForCausalLM.from_pretrained(lora_model_name).to(device)
    lora_model_perplexity, lora_model_results = evaluate_model(project_root_path, lora_model, test_dataset)
    print(f"Lora finetuned model perplexity: {lora_model_perplexity}")
    print(f"Lora finetuned model results: {lora_model_results}")


if __name__ == "__main__":
    main()