import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import torch


logger = logging.getLogger(__name__)


@dataclass
class Config:
    # --- Model Settings ---
    model_name: str = "Qwen/Qwen2.5-Coder-1.5B"
    model_attn_implementation: str = "sdpa"  # sdpa = built-in PyTorch implementation of scaled dot product attention, imporves performance and memory efficiency
    model_dtype: torch.dtype = field(init=False)
    fim_pad_token: str = "<|fim_pad|>"

    # --- LoRA Settings ---
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.1
    lora_bias: str = "none"
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "v_proj", "k_proj", "o_proj",
        "gate_proj", "down_proj", "up_proj"
    ])

    # --- Trainer Hyperparameters ---
    trainer_num_train_epochs: int = 1
    trainer_per_device_train_batch_size: int = 2
    trainer_per_device_eval_batch_size: int = 2
    trainer_gradient_accumulation_steps: int = 32  # simulates a larger effective batch size, number of forward/backward passes to accumulate before performing one optimizer step
    trainer_learning_rate: float = 2e-5
    trainer_weight_decay: float = 0.1
    trainer_max_grad_norm: float = 1.0
    trainer_lr_scheduler_type: str = "cosine"
    trainer_warmup_steps: int = 50
    trainer_gradient_checkpointing: bool = True  # saves memory by storing only key "checkpoint" activations and re-calculating intermediate ones during the backward pass
    trainer_max_steps: int = field(init=False)
    collator_label_pad_token_id: int = -100

    # --- Logging and Evaluation Strategy ---
    trainer_log_level: str = "info"
    trainer_logging_steps: int = 10  # average training loss over trainer_logging_steps period is calculated and logged
    trainer_eval_strategy: str = "steps"
    trainer_eval_steps: int = 100
    trainer_save_strategy: str = "steps"
    trainer_save_steps: int = 100
    trainer_logging_strategy: str = "steps"

    # --- Dataset ---
    dataset_shuffle_buffer_size: int = 50000
    dataset_shuffle_seed: int = 0
    dataset_train_dataset_length: int = field(init=False)

    # --- Hardware Configuration --- 
    device: str = field(init=False)
    
    # --- Path Management ---
    project_root_path: Path = field(init=False)
    train_dataset_path: Path = field(init=False)
    eval_dataset_path: Path = field(init=False)
    finetune_outputs_dir_path: Path = field(init=False)
    trainer_checkpoints_dir_path: Path = field(init=False)
    trainer_log_path: Path = field(init=False)
    trainer_plot_path: Path = field(init=False)
    lora_adapter_path: Path = field(init=False)
    lora_model_path: Path = field(init=False)
    trainer_model_merge_offload_folder_path: Path = field(init=False)

    def __post_init__(self):
        self._setup_device_and_precision()
        self._setup_paths()
        self._ensure_output_paths_exist()
        self.dataset_train_dataset_length = self._get_dataset_length(self.train_dataset_path)
        self.trainer_max_steps = self._calculate_max_steps()
        
    def _setup_device_and_precision(self):
        if torch.cuda.is_available():
            self.device = "cuda"
            self.model_dtype = torch.bfloat16
        elif torch.backends.mps.is_available():
            self.device = "mps"
            self.model_dtype = torch.float16
        else:
            self.device = "cpu"
            self.model_dtype = torch.float32

    def _setup_paths(self):
        self.project_root_path = Path(__file__).resolve().parents[2]
        self.train_dataset_path = self.project_root_path / "outputs" / "preprocess" / "results" / "datasets" / "train_dataset.jsonl"
        self.eval_dataset_path = self.project_root_path / "outputs" / "preprocess" / "results" / "datasets" / "eval_dataset.jsonl"
        self.finetune_outputs_dir_path = self.project_root_path / "outputs" / "finetune"
        self.trainer_checkpoints_dir_path = self.finetune_outputs_dir_path / "checkpoints"
        self.trainer_model_merge_offload_folder_path = self.finetune_outputs_dir_path / "trainer_model_merge_offload_folder"
        self.trainer_log_path = self.finetune_outputs_dir_path / "results" / "trainer_log.json" 
        self.trainer_plot_path = self.finetune_outputs_dir_path / "results" / "trainer_loss_plot.png"
        self.lora_adapter_path = self.finetune_outputs_dir_path / "results" / "lora_adapter"
        self.lora_model_path = self.finetune_outputs_dir_path / "results" / "lora_model"
    
    def _ensure_output_paths_exist(self) -> None:
        paths = [
            self.finetune_outputs_dir_path,
            self.trainer_checkpoints_dir_path,
            self.trainer_model_merge_offload_folder_path,
            self.trainer_log_path, 
            self.trainer_plot_path,
            self.lora_adapter_path,
            self.lora_model_path
        ]
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)

    def _get_dataset_length(self, path: Path) -> int:
        """
        Calculates line count for streaming dataset progress estimation.
        We need to do this because we load it as a streaming dataset iterator, which can only be iterated once.
        """
        if not path.exists():
            err_msg = f"Training dataset not found at expected path: {path}. Ensure the dataset file exists before initializing the Config."
            logger.error(err_msg)
            raise FileNotFoundError(err_msg)
        with open(path, "r", encoding="utf-8") as file:
            count = 0
            for _ in file:
                count += 1
            return count
        
    def _calculate_max_steps(self) -> int:
        """
        Calculates total steps based on effective batch size and dataset length.
        Because we use streaming dataset iterators for efficiency, we cannot use num_train_epochs trainer class parameter directly.
        Instead, we need to calculate max_steps and pass it to the trainer.
        """
        if self.dataset_train_dataset_length == 0:
            return 0
            
        effective_batch_size = (self.trainer_per_device_train_batch_size * self.trainer_gradient_accumulation_steps)
        steps_per_epoch = math.ceil(self.dataset_train_dataset_length / effective_batch_size)
        max_steps = steps_per_epoch * self.trainer_num_train_epochs
        return max_steps
       