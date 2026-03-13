import math
import torch
from dataclasses import dataclass, field
from typing import List
from pathlib import Path


@dataclass
class Config:
    model_name: str = "Qwen/Qwen2.5-Coder-1.5B"
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
    trainer_per_device_eval_batch_size: int = 2 
    trainer_gradient_accumulation_steps: int = 32 # Number of forward/backward passes to accumulate before performing one optimizer step.
    trainer_max_steps: int = field(init=False)
    trainer_learning_rate: float = 2e-5
    trainer_weight_decay: float = 0.1
    trainer_max_grad_norm: int = 1
    trainer_lr_scheduler_type: str = "cosine"
    trainer_warmup_steps: int = 50
    trainer_log_level: str = "info"
    trainer_logging_steps: int = 10   # Average training loss over trainer_logging_steps period is calculated and logged.
    trainer_eval_strategy: str = "steps"
    trainer_eval_steps: int = 100 
    trainer_save_steps: int = 100 
    trainer_logging_strategy: str = "steps"
    trainer_save_strategy: str = "steps"
    trainer_bf16: bool = True   # If set to True fp16 needs to be set to false 
    trainer_fp16: bool = False  # If set to True bf16 needs to be set to False
    trainer_gradient_checkpointing: bool = True

    collator_label_pad_token_id: int = -100
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
            self.trainer_gradient_checkpointing = False
            self.device = "mps"
        else:
            self.device = "cpu"

        self.project_root_path = Path(__file__).resolve().parent.parent.parent
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