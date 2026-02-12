import argparse
import json
import math
import shutil
import sys        
import gc
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Dict

import matplotlib.pyplot as plt
import torch
from datasets import Features, Sequence, Value, IterableDataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel

@dataclass
class Config:
    model_name: str = "Qwen/Qwen2.5-Coder-7B"
    fim_pad_token: str = "<|fim_pad|>"
    lora_r: int = 32 
    lora_alpha: int = 64
    lora_dropout: float = 0.1          
    lora_bias: str = "none"            
    lora_target_modules: List[str] = field(default_factory=lambda:[
        "q_proj", 
        "v_proj", 
        "k_proj", 
        "o_proj",
        "gate_proj",  
        "down_proj",  
        "up_proj"     
    ])
    trainer_num_train_epochs: int = 1 
    trainer_per_device_train_batch_size: int = 2 
    trainer_per_device_eval_batch_size: int = 4 
    trainer_gradient_accumulation_steps: int = 16 # Number of forward/backward passes to accumulate before performing one optimizer step.
    trainer_max_steps: int = field(init=False)
    trainer_learning_rate: float = 2e-5
    trainer_logging_steps: int = 10   # Average training loss over trainer_logging_steps period is calculated and logged.
    trainer_eval_strategy: str = "steps"
    trainer_eval_steps: int = 100 
    trainer_save_steps: int = 100 

    train_dataset_length: int = field(init=False) 
    device: str = field(init=False) 
    shuffle_buffer_size: int = 50000 
    shuffle_seed: int = 0

    project_root_path: Path = field(init=False) 
    train_dataset_path: Path = field(init=False) 
    eval_dataset_path: Path = field(init=False) 
    trainer_output_dir_path: Path = field(init=False) 
    lora_adapter_path: Path = field(init=False)
    lora_model_path: Path = field(init=False)
    model_merge_offload_folder_path: Path = field(init=False)

    def __post_init__(self):
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        self.project_root_path = Path(__file__).resolve().parent.parent 
        self.train_dataset_path = self.project_root_path / "datasets" / "train_dataset.jsonl"
        self.eval_dataset_path = self.project_root_path / "datasets" / "eval_dataset.jsonl"
        self.trainer_output_dir_path = self.project_root_path / "results"
        self.lora_adapter_path = self.project_root_path / "lora_adapter"
        self.lora_model_path = self.project_root_path / "lora_model"
        self.model_merge_offload_folder_path = self.project_root_path / "model_merge_offload_folder"


        # Calculate the length of the training dataset separately.
        # because we load it as a streaming dataset iterator, which can only be iterated once.
        with open(self.train_dataset_path, "r", encoding="utf-8") as f:
            self.train_dataset_length = sum(1 for _ in f)

        # Because we use streaming dataset iterators for efficiency, we cannot use num_train_epochs trainer class parameter directly.
        # Instead, we need to calculate max_steps and pass it to the trainer.
        self.trainer_max_steps = math.ceil(self.train_dataset_length /
                                (self.trainer_per_device_train_batch_size * self.trainer_gradient_accumulation_steps)
                                ) * self.trainer_num_train_epochs

def parse_args(config: Config) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start or resume LoRA model training")
    parser.add_argument("--resume",
                        type=str, 
                        default=None, 
                        metavar="CHECKPOINT",
                        help='Checkpoint name to resume training from, or "last"')
    user_args = parser.parse_args()

    if user_args.resume:
        answer = input(f"You passed --resume='{user_args.resume}'. Do you want to resume training from this checkpoint? (y/N): ").strip().lower()
        if answer not in ["y", "yes"]:
            print("Aborting. To start fresh training, run the script without --resume.")
            sys.exit(0)
    
    else:
        train_fresh = input("Do you really want to start training from scratch? (y/N): ").strip().lower()
        if train_fresh not in ["y", "yes"]:
            print("Aborting fresh training run.")
            sys.exit(0) 
        
        answer = input(f"No --resume argument passed. Do you want to delete the entire '{config.trainer_output_dir_path}' folder and recreate it empty? (y/N): ").strip().lower()
        if answer in ["y", "yes"]:
            if config.trainer_output_dir_path.exists():
                print(f"Deleting {config.trainer_output_dir_path} folder and all its contents...")
                shutil.rmtree(config.trainer_output_dir_path)
            config.trainer_output_dir_path.mkdir(parents=True, exist_ok=True)
            print(f"Recreated empty {config.trainer_output_dir_path} folder.")
        
    return user_args

def load_datasets(config: Config) -> Tuple[IterableDataset, IterableDataset, IterableDataset]:
    # Define the expected schema/features of datasets.
    # Use 'int32' which the datasets library and pytorch map correctly to int tensors.
    dataset_features = Features({
        'input_ids': Sequence(feature=Value(dtype='int32')),
        'attention_mask': Sequence(feature=Value(dtype='int32')),
        'labels': Sequence(feature=Value(dtype='int32')),
    })

    # Enable streaming mode to load the dataset as an iterator.
    # This allows processing data samples on-the-fly without downloading or loading the entire dataset into memory.
    # https://huggingface.co/docs/datasets/stream
    train_dataset = load_dataset("json", data_files=str(config.train_dataset_path), features=dataset_features, streaming=True)["train"]
    train_dataset = train_dataset.shuffle(buffer_size=config.shuffle_buffer_size, seed=config.shuffle_seed) # Take up to suffle_buffer_size examples and randomly shuffle them
    eval_dataset = load_dataset("json", data_files=str(config.eval_dataset_path), features=dataset_features, streaming=True)["train"]
    return train_dataset, eval_dataset 

def load_and_configure_lora_model(config: Config) -> AutoModelForCausalLM:
    lora_config = LoraConfig(
        r=config.lora_r, 
        lora_alpha=config.lora_alpha, 
        lora_dropout=config.lora_dropout, 
        bias=config.lora_bias, 
        target_modules=config.lora_target_modules,
    )

    if config.device == "cuda":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=config.model_name,
            quantization_config=bnb_config,
            device_map="auto" # Let bitsandbytes handle placement
        )
        model = prepare_model_for_kbit_training(model)
    elif config.device == "mps":
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=config.model_name,
            torch_dtype=torch.float16  # Reduces weights from 32-bit to 16-bit float.
        ).to("mps")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=config.model_name,
        ).to("cpu")

    lora_model = get_peft_model(model, lora_config)
    return lora_model

class FIMDataCollator:
    def __init__(self, tokenizer, label_pad_token_id):
        self.tokenizer = tokenizer
        self.label_pad_token_id = label_pad_token_id 

    def __call__(self, examples: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        # Initialize max length for this specific batch
        max_ex_length = 0 
        for ex in examples:
            if len(ex["input_ids"]) > max_ex_length:
                max_ex_length = len(ex["input_ids"])
        
        # Apply padding using the saved tokenizer
        for ex in examples:
            pad_length = max_ex_length - len(ex["input_ids"])
            ex["input_ids"] += [self.tokenizer.pad_token_id] * pad_length
            ex["attention_mask"] += [0] * pad_length
            ex["labels"] += [self.label_pad_token_id] * pad_length

        # Stack separate examples into a single tensors matrix
        examples_batch = {
            "input_ids": torch.tensor([ex["input_ids"] for ex in examples], dtype=torch.long),
            "attention_mask": torch.tensor([ex["attention_mask"] for ex in examples], dtype=torch.long),
            "labels": torch.tensor([ex["labels"] for ex in examples], dtype=torch.long)
        }

        return examples_batch

def train_and_save_lora_model(
        config: Config,
        lora_model: AutoModelForCausalLM,
        train_dataset: IterableDataset,
        eval_dataset: IterableDataset,
        user_args: argparse.Namespace
) -> List:  
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = config.fim_pad_token
    tokenizer.padding_side = "right"

    training_args = TrainingArguments(
        output_dir=config.trainer_output_dir_path,
        per_device_train_batch_size=config.trainer_per_device_train_batch_size,
        per_device_eval_batch_size=config.trainer_per_device_eval_batch_size,
        gradient_accumulation_steps=config.trainer_gradient_accumulation_steps, # Simulate a batch size of 4 but only load 1 example at a time in memory.
        logging_steps=config.trainer_logging_steps,
        eval_strategy=config.trainer_eval_strategy,
        eval_steps=config.trainer_eval_steps,
        save_steps=config.trainer_save_steps,
        max_steps=config.trainer_max_steps,
        learning_rate=config.trainer_learning_rate,
        fp16=True,  # Enables 16bit precision for the training loop
        gradient_checkpointing=True
    )

    data_collator = FIMDataCollator(
        tokenizer=tokenizer,
        label_pad_token_id=-100
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
        trainer.train(resume_from_checkpoint=user_args.resume)  # specific checkpoint provided 
    else:
        trainer.train() # train from scratch

    lora_model.save_pretrained(config.lora_adapter_path) # Save lora adapter only
    log_history = trainer.state.log_history

    if config.device == "cuda":
        # Clear 4-bit model from VRAM to make room for the FP16 base model
        if 'trainer' in locals():
            del trainer  # Clear trainer references

        del lora_model  # Remove model reference

        gc.collect()  # Force garbage collection

        if torch.cuda.is_available():
            torch.cuda.synchronize()  # Wait for gpu
            torch.cuda.empty_cache()  # Clear gpu cache
        
        # Ensure offload folder for merging exists before loading the model
        config.model_merge_offload_folder_path.mkdir(parents=True, exist_ok=True)

        # Load fresh FP16 base model
        base_model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=config.model_name,
            dtype=torch.float16,
            device_map="auto",
            offload_folder=str(config.model_merge_offload_folder_path), # Offload model layers to disk during loading to prevent RAM (OOM) crashes. 
            low_cpu_mem_usage=True
        )

        lora_model = PeftModel.from_pretrained(
            base_model, 
            config.lora_adapter_path,
            offload_folder=str(config.model_merge_offload_folder_path)
        )

    # Merge lora adapter into base model and save it with the tokenizer of the model
    merged_model= lora_model.merge_and_unload() 
    merged_model = merged_model.to(torch.float16) 
    merged_model.save_pretrained(config.lora_model_path)
    tokenizer.save_pretrained(config.lora_model_path)

    # Clean up offload folder
    if config.model_merge_offload_folder_path.exists():
        shutil.rmtree(config.model_merge_offload_folder_path)

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

def main():
    config = Config()
    user_args = parse_args(config)
    train_dataset, eval_dataset= load_datasets(config)

    lora_model = load_and_configure_lora_model(config)
    lora_model.print_trainable_parameters()
    log_history = train_and_save_lora_model(config, lora_model, train_dataset, eval_dataset, user_args)
    save_log(config, log_history)

    plot_loss(config)

if __name__ == "__main__":
    main()
