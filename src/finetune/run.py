import argparse
import logging.config
import shutil
import sys
from typing import Tuple

from datasets import Features, IterableDataset, Sequence, Value, load_dataset
from transformers import AutoTokenizer

from .config import Config
from .model import load_and_configure_lora_model
from .train import plot_loss, save_log, train_lora_model, merge_lora_and_save


logger = logging.getLogger("src.finetune.run")


def _setup_logger(log_level: str) -> None:
    logger_config = {
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
        "loggers": {
            "src.finetune": {  
                "handlers": ["stderr_handler"],
                "level": log_level,
                "propagate": False,
            }
        }
    }
    logging.config.dictConfig(logger_config)


def _parse_args(config: Config) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start or resume LoRA model training")
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help='Resume from "last" or path'
    )
    parser.add_argument(
        "--delete-all-checkpoints",
        action="store_true",
        help="Delete existing output dir"
    )
    args = parser.parse_args()

    output_dir = config.trainer_output_dir_path
    if args.resume and not args.delete_all_checkpoints:
        logger.info(f"Resuming from: {args.resume}")
    elif not args.resume and args.delete_all_checkpoints:
        if output_dir.exists():
            logger.warning(f"Deleting: {output_dir}")
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Cleared: {output_dir}")
    elif args.resume and args.delete_all_checkpoints:
        logger.error("Configuration conflict: Cannot resume from a checkpoint while '--delete-all-checkpoints' is set.")
        sys.exit(1)
    else:
        logger.info(f"Starting fresh training. Existing checkpoints in {output_dir} are preserved.")

    return args


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
    user_args = _parse_args(config)
    _setup_logger(user_args.log_level)
    train_dataset, eval_dataset= load_datasets(config)
    logger.info(f"Dataset: {config.dataset_train_dataset_length} train examples, max_steps={config.trainer_max_steps}")

    lora_model = load_and_configure_lora_model(config)
    lora_model.print_trainable_parameters()

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = config.fim_pad_token
    tokenizer.padding_side = "right"

    log_history = train_lora_model(config, lora_model, tokenizer, train_dataset, eval_dataset, user_args)
    merge_lora_and_save(config, tokenizer)
    # log_history = train_and_save_lora_model(config, lora_model, train_dataset, eval_dataset, user_args)
    save_log(config, log_history)

    plot_loss(config)


if __name__ == "__main__":
    main()
