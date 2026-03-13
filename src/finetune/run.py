import argparse
import logging.config
import shutil
import sys
from typing import Tuple

from datasets import Features, IterableDataset, Sequence, Value, load_dataset

from .config import Config
from .model import load_and_configure_lora_model
from .train import plot_loss, save_log, train_and_save_lora_model


logger = logging.getLogger("src.finetune.run")


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


def _parse_args(config: Config) -> argparse.Namespace:
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
            logger.info("Aborting. To start fresh training, run the script without --resume.")
            sys.exit(0)
    
    else:
        train_fresh = input("Do you really want to start training from scratch? (y/N): ").strip().lower()
        if train_fresh not in ["y", "yes"]:
            logger.info("Aborting fresh training run.")
            sys.exit(0) 
        
        answer = input(f"No --resume argument passed. Do you want to delete the entire '{config.trainer_output_dir_path}' folder and recreate it empty? (y/N): ").strip().lower()
        if answer in ["y", "yes"]:
            if config.trainer_output_dir_path.exists():
                logger.info(f"Deleting {config.trainer_output_dir_path} folder and all its contents...")
                shutil.rmtree(config.trainer_output_dir_path)
            config.trainer_output_dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Recreated empty {config.trainer_output_dir_path} folder.")
        
    return user_args


def load_datasets(config: Config) -> Tuple[IterableDataset, IterableDataset]:
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
    train_dataset = train_dataset.shuffle(buffer_size=config.dataset_shuffle_buffer_size, seed=config.dataset_shuffle_seed)  # take up to suffle_buffer_size examples and randomly shuffle them
    eval_dataset = load_dataset("json", data_files=str(config.eval_dataset_path), features=dataset_features, streaming=True)["train"]
    return train_dataset, eval_dataset 


def main() -> None:
    config = Config()
    _setup_logger("INFO")
    user_args = _parse_args(config)
    train_dataset, eval_dataset= load_datasets(config)
    logger.info(f"Dataset: {config.dataset_train_dataset_length} train examples, max_steps={config.trainer_max_steps}")

    lora_model = load_and_configure_lora_model(config)
    lora_model.print_trainable_parameters()

    log_history = train_and_save_lora_model(config, lora_model, train_dataset, eval_dataset, user_args)
    save_log(config, log_history)

    plot_loss(config)


if __name__ == "__main__":
    main()
