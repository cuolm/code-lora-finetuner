import argparse
import logging.config
from pathlib import Path

from transformers import AutoTokenizer
import tree_sitter as ts

from .config import Config
from .extractor import get_code_blocks_from_auto_split, get_code_blocks_from_manual_split
from .processor import create_fim_examples, estimate_bytes_per_token_ratio, tokenize_and_save_fim_examples


logger = logging.getLogger(__name__)


def _setup_logger(log_level: str) -> None:
    """
    Configure the root logger ("") to capture logs from the entry point (__main__),
    all internal sub-modules, and third-party libraries via a single handler.
    """
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
            "": {  # "" corresponds to root logger 
                "handlers": ["stderr_handler"],
                "level": log_level,
                "propagate": False,
            }
        }
    }
    logging.config.dictConfig(logger_config)


def _normalize_extension(ext: str) -> str:
    ext = ext.strip().lower()
    return ext if ext.startswith(".") else f".{ext}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess code dataset for FIM fine-tuning.")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="YAML config file path with 'preprocess:' section. All Config.MISSING fields must be provided."
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level.",
    )
    return parser.parse_args()

def _clear_existing_datasets(config: Config) -> None:
    for file in [config.train_dataset_path, config.eval_dataset_path, config.test_dataset_path]:
        if file.exists():
            file.unlink()
            logger.info(f"Deleted old dataset: {file}")


def _validate_and_configure_tokenizer(config: Config, tokenizer: AutoTokenizer):
    required_tokens = [
        config.fim_prefix_token,
        config.fim_middle_token,
        config.fim_suffix_token,
        config.fim_pad_token,
        config.eos_token
    ]

    for token in required_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        
        if token_id == tokenizer.unk_token_id or token_id is None:
            logger.error(f"Token '{token}' defined in Config is missing from Tokenizer vocabulary.")
            raise ValueError(f"Tokenizer mismatch: {token} not found.")

    tokenizer.pad_token = config.fim_pad_token
    tokenizer.eos_token = config.eos_token

    logger.info("Tokenizer validated and configured successfully.")


def run(config: Config) -> None:
    _clear_existing_datasets(config)

    if config.split_mode == "auto":
        logger.info("Using auto-generated dataset split.")
        train_code_blocks_iter, eval_code_blocks_iter, test_code_blocks_iter = get_code_blocks_from_auto_split(config) 
    else: # config.split_mode == "manual":
        logger.info("Using manual dataset split from directories.")
        train_code_blocks_iter, eval_code_blocks_iter, test_code_blocks_iter = get_code_blocks_from_manual_split(config) 

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    _validate_and_configure_tokenizer(config, tokenizer)

    bytes_per_token_ratio = estimate_bytes_per_token_ratio(config, tokenizer, number_of_code_blocks=20000)
    logger.info(f"Estimated bytes_per_token_ratio: {bytes_per_token_ratio}")

    train_fim_examples_iter= create_fim_examples(config, train_code_blocks_iter, bytes_per_token_ratio)
    eval_fim_examples_iter = create_fim_examples(config, eval_code_blocks_iter, bytes_per_token_ratio)
    test_fim_examples_iter = create_fim_examples(config, test_code_blocks_iter, bytes_per_token_ratio)
    
    tokenize_and_save_fim_examples(config, config.train_dataset_path, train_fim_examples_iter, tokenizer)
    tokenize_and_save_fim_examples(config, config.eval_dataset_path, eval_fim_examples_iter, tokenizer)
    tokenize_and_save_fim_examples(config, config.test_dataset_path, test_fim_examples_iter, tokenizer)

    logger.info("Saved train, eval, test datasets to disk")  


def main() -> None:
    user_args = _parse_args()
    preprocess_config = Config.load_from_yaml(user_args.config)
    _setup_logger(user_args.log_level)
    run(config=preprocess_config)


if __name__ == "__main__":
    main()
    