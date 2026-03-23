import logging
import math
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import List, Any

import torch
from omegaconf import OmegaConf, MISSING


logger = logging.getLogger(__name__)


@dataclass
class Config:
# --- Mandatory Parameters ---
    model_name: str = MISSING
    fim_pad_token: str = MISSING
    label_pad_token_id: int = MISSING

    # --- Model Settings ---
    model_attn_implementation: str = "sdpa"  # sdpa = built-in PyTorch implementation of scaled dot product attention, imporves performance and memory efficiency
    model_dtype: Any = field(init=False)    # changed to Any to bypass OmegaConf validation

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
    trainer_resume_from_checkpoint: str | None = "last"
    trainer_clear_checkpoint_dir: bool = False
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

    # --- Logging and Evaluation Strategy ---
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

    @classmethod
    def load_from_yaml(cls, yaml_path: Path) -> "Config":
        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")
        
        logger.info(f"Loading configuration from {yaml_path}") 
        config_dict = OmegaConf.structured(cls)
        try:
            yaml_file_node = OmegaConf.load(yaml_path)
        except Exception as e:
            raise ValueError(f"Failed to load YAML config {yaml_path}") from e

        yaml_file_dict = OmegaConf.to_container(yaml_file_node, resolve=True)
        yaml_finetune_dict = yaml_file_dict.get("finetune", {})

        yaml_finetune_valid_dict = {}
        # Filter YAML fields to include only those defined in the Config dataclass.
        # This prevents OmegaConf from raising an AttributeError when encountering 
        # global YAML anchors or keys not present in the current Config dataclass. 
        for field in fields(cls):
            if field.name in yaml_finetune_dict:
                yaml_finetune_valid_dict[field.name] = yaml_finetune_dict[field.name]
        logger.debug(f"Filtered YAML configuration: {yaml_finetune_valid_dict}")

        merged_config_dict = OmegaConf.merge(config_dict, yaml_finetune_valid_dict)
        return OmegaConf.to_object(merged_config_dict)

    def __post_init__(self) -> None:
        self._setup_device_and_precision()
        self._setup_paths()
        self._ensure_output_paths_exist()
        self.dataset_train_dataset_length = self._get_dataset_length(self.train_dataset_path)
        self.trainer_max_steps = self._calculate_max_steps()
        
    def _setup_device_and_precision(self) -> None:
        if torch.cuda.is_available():
            self.device = "cuda"
            self.model_dtype = torch.bfloat16
        elif torch.backends.mps.is_available():
            self.device = "mps"
            self.model_dtype = torch.float16
        else:
            self.device = "cpu"
            self.model_dtype = torch.float32
        logger.info(f"Execution environment: device={self.device}, dtype={self.model_dtype}")
    
    def _setup_paths(self) -> None:
        self.project_root_path = Path(__file__).resolve().parents[3]
        self.train_dataset_path = self.project_root_path / "outputs" / "preprocess" / "results" / "datasets" / "train_dataset.jsonl"
        self.eval_dataset_path = self.project_root_path / "outputs" / "preprocess" / "results" / "datasets" / "eval_dataset.jsonl"
        self.finetune_outputs_dir_path = self.project_root_path / "outputs" / "finetune"
        self.trainer_checkpoints_dir_path = self.finetune_outputs_dir_path / "checkpoints"
        self.trainer_model_merge_offload_folder_path = self.finetune_outputs_dir_path / "trainer_model_merge_offload_folder"
        self.trainer_log_path = self.finetune_outputs_dir_path / "results" / "trainer_log.json" 
        self.trainer_plot_path = self.finetune_outputs_dir_path / "results" / "trainer_loss_plot.png"
        self.lora_adapter_path = self.finetune_outputs_dir_path / "results" / "lora_adapter"
        self.lora_model_path = self.finetune_outputs_dir_path / "results" / "lora_model"
        logger.debug(f"Resolved project root: {self.project_root_path}")
    
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
            if not path.parent.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                logger.debug(f"Created parent directory: {path.parent}")
            else:
                logger.debug(f"Parent directory already exists: {path.parent}") 

    def _get_dataset_length(self, path: Path) -> int:
        """
        Calculates line count for streaming dataset progress estimation.
        We need to do this because we load it as a streaming dataset iterator, which can only be iterated once.
        """
        if not path.exists():
            raise FileNotFoundError( f"Training dataset not found at expected path: {path}. Ensure the dataset file exists before initializing the Config.")
        count = 0
        with open(path, "r", encoding="utf-8") as file:
            for _ in file:
                count += 1
        logger.debug(f"Dataset length for '{path.name}': {count} lines")
        return count
        
    def _calculate_max_steps(self) -> int:
        """
        Calculates total steps based on effective batch size and dataset length.
        Because we use streaming dataset iterators for efficiency, we cannot use num_train_epochs trainer class parameter directly.
        Instead, we need to calculate max_steps and pass it to the trainer.
        """
        if self.dataset_train_dataset_length == 0:
            logger.warning("Dataset length is 0; max_steps will be 0.")
            return 0
            
        effective_batch_size = (self.trainer_per_device_train_batch_size * self.trainer_gradient_accumulation_steps)
        if effective_batch_size == 0:
            raise ValueError("Effective batch size (batch_size * grad_accum) cannot be zero.")
        steps_per_epoch = math.ceil(self.dataset_train_dataset_length / effective_batch_size)
        max_steps = steps_per_epoch * self.trainer_num_train_epochs
        logger.debug(f"Calculated training schedule: {max_steps} total steps ({steps_per_epoch} steps/epoch for {self.trainer_num_train_epochs} epochs)")
        return max_steps
       