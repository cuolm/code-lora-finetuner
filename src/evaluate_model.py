import os
import json
import gc
import logging.config
import argparse
from dataclasses import dataclass, field 
from typing import Tuple, Dict
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint
from datasets import Features, Sequence, Value, IterableDataset, load_dataset
from torch.utils.data import Dataset
from pathlib import Path
import matplotlib.pyplot as plt

from finetune_model import FIMDataCollator
from peft import PeftModel


@dataclass
class Config:
    model_name: str = "Qwen/Qwen2.5-Coder-7B"
    fim_pad_token: str = "<|fim_pad|>"
    collator_label_pad_token_id: int = -100

    test_dataset_path: Path = field(init=False)
    project_root_path: Path = field(init=False)
    lora_adapter_path: Path = field(init=False)
    lora_model_path: Path = field(init=False)
    trainer_output_dir_path: Path = field(init=False) 
    test_output_dir_path: Path = field(init=False)

    trainer_per_device_eval_batch_size: int = 4
    device: str = field(init=False)

    def __post_init__(self):
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        self.project_root_path = Path(__file__).resolve().parent.parent 
        self.test_dataset_path = self.project_root_path / "datasets" / "test_dataset.jsonl"
        self.lora_adapter_path = self.project_root_path / "lora_adapter"
        self.lora_model_path = self.project_root_path / "lora_model"
        self.trainer_output_dir_path = self.project_root_path / "results"
        self.test_output_dir_path = self.project_root_path / "test_results"


logger = logging.getLogger(__name__)

def _setup_logger(log_level: str) -> None:
    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            }
        },
        "handlers": {
            "stderr_handler": {
                "level": log_level,
                "class": "logging.StreamHandler",
                "formatter": "standard",
            }
        },
        "root": {
            "handlers": ["stderr_handler"],
            "level": log_level,
            "propagate": True
        }
    }
    logging.config.dictConfig(config)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Comparison checkpoint")
    parser.add_argument("--checkpoint",
                        type=str, 
                        required=True,
                        metavar="NAME_OR_LAST",
                        help='Checkpoint name, or "last"')
    user_args = parser.parse_args()
    return user_args


def _evaluate_model(config: Config, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, test_dataset: Dataset) -> Tuple[float, Dict] :
    """
    Evaluates the model on the test dataset calculating the loss and perplexity.

    Perplexity represents the model's uncertainty as the effective number of words (or choices) 
    it considers equally likely for every word it predicts.
    A Perplexity of 1.0 is the ideal minimum (meaning the model always predicts correctly). 
    Example: If Perplexity=10, the model is as uncertain as if it chose from 10 words. 
    If Perplexity=50, the model is 5 times more uncertain. Lower Perplexity is always better.
    """
    eval_args = TrainingArguments(
        per_device_eval_batch_size=config.trainer_per_device_eval_batch_size,
        fp16=True,  # enables 16bit precision
    )

    data_collator = FIMDataCollator(
        tokenizer=tokenizer,
        label_pad_token_id=config.collator_label_pad_token_id
    )
    
    trainer = Trainer(
        model=model,
        args=eval_args,
        eval_dataset=test_dataset,
        processing_class = tokenizer,
        data_collator = data_collator
    )

    results = trainer.evaluate()
    perplexity = np.exp(results["eval_loss"])

    if 'trainer' in locals():
        del trainer  # clear trainer references

    return perplexity, results


def _clear_hardware_cache(config: Config) -> None:
    gc.collect()
    if config.device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    elif config.device == "mps":
        torch.mps.empty_cache()


def main():
    config = Config()
    _setup_logger("INFO")
    user_args = _parse_args()

    dataset_features = Features({
        'input_ids': Sequence(feature=Value(dtype='int32')),
        'attention_mask': Sequence(feature=Value(dtype='int32')),
        'labels': Sequence(feature=Value(dtype='int32')),
    })

    test_dataset = load_dataset("json", data_files=str(config.test_dataset_path), features=dataset_features, streaming=True)["train"]

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = config.fim_pad_token
    tokenizer.padding_side = "right"

    base_model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=config.model_name,
        dtype=torch.float16,
        device_map="auto",
    )
    base_model_perplexity, base_model_results = _evaluate_model(config, base_model, tokenizer, test_dataset)
    logger.info(f"Base model perplexity: {base_model_perplexity}")
    logger.info(f"Base model results: {base_model_results}")

    del base_model
    _clear_hardware_cache(config)

    if user_args.checkpoint == "last":
        checkpoint_path = get_last_checkpoint(str(config.trainer_output_dir_path))
    elif user_args.checkpoint:
        checkpoint_path = str(config.trainer_output_dir_path / user_args.checkpoint)
    else:
        err_msg = "No checkpoint specified. Use --checkpoint <name> or 'last'"
        logger.error(err_msg)
        raise ValueError(err_msg)

    base_model_for_lora = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=config.model_name,
        dtype=torch.float16,
        device_map="auto"
    )

    lora_model = PeftModel.from_pretrained(
        model = base_model_for_lora, 
        model_id = checkpoint_path 
    )
    lora_model_perplexity, lora_model_results = _evaluate_model(config, lora_model, tokenizer, test_dataset)
    logger.info(f"Lora finetuned model perplexity: {lora_model_perplexity}")
    logger.info(f"Lora finetuned model results: {lora_model_results}")

    del lora_model
    _clear_hardware_cache(config)


if __name__ == "__main__":
    main()