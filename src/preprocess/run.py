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
        "--force-delete-datasets",
        action="store_true",
        help="Delete existing datasets without confirmation.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level.",
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        type=_normalize_extension,
        default=[".c", ".h"],
        help="List of file extensions to include (e.g. .c .h .cpp .py)"
    )
    parser.add_argument(
        "--source-files-language",
        type=str,
        default="c", 
        help="The source code language to process (e.g., c, python, java). Used for Tree-sitter parsing."
    )
    parser.add_argument(
        "--split-mode",
        type=str,
        choices=["auto", "manual"],
        default="auto",  
        help="Dataset splitting mode. Choose 'auto' for automatic ratio-based split or 'manual' for pre-split directories (train/eval/test) in raw_data_path (the directories have to be present if using 'manual')"
    )
    parser.add_argument(
        "--raw-data-path",
        type=Path,
        default=None,
        help="Optional path to the root directory containing the raw source code files. Overrides the default path './data'."
    )
    parser.add_argument(
        "--tree-sitter-parser-path",
        type=Path,
        default=None,
        help="Optional path to a custom compiled Tree-sitter shared library file (.so, .dylib, .dll). If not set, the parser is loaded from the standard language pack.")
    return parser.parse_args()


def _clear_existing_datasets(config: Config) -> None:
    for file in [config.train_path, config.eval_path, config.test_path]:
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


def run(config: Config, split_mode: str) -> None:
    _clear_existing_datasets(config)

    if split_mode == "auto":
        logger.info("Using auto-generated dataset split.")
        train_code_blocks_iter, eval_code_blocks_iter, test_code_blocks_iter = get_code_blocks_from_auto_split(config) 
    else: # user_args.split_mode == "manual":
        logger.info("Using manual dataset split from directories.")
        train_code_blocks_iter, eval_code_blocks_iter, test_code_blocks_iter = get_code_blocks_from_manual_split(config) 

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    _validate_and_configure_tokenizer(config, tokenizer)

    bytes_per_token_ratio = estimate_bytes_per_token_ratio(config, tokenizer, number_of_code_blocks=20000)
    logger.info(f"Estimated bytes_per_token_ratio: {bytes_per_token_ratio}")

    train_fim_examples_iter= create_fim_examples(config, train_code_blocks_iter, bytes_per_token_ratio)
    eval_fim_examples_iter = create_fim_examples(config, eval_code_blocks_iter, bytes_per_token_ratio)
    test_fim_examples_iter = create_fim_examples(config, test_code_blocks_iter, bytes_per_token_ratio)
    
    tokenize_and_save_fim_examples(config, config.train_path, train_fim_examples_iter, tokenizer)
    tokenize_and_save_fim_examples(config, config.eval_path, eval_fim_examples_iter, tokenizer)
    tokenize_and_save_fim_examples(config, config.test_path, test_fim_examples_iter, tokenizer)

    logger.info("Saved train, eval, test datasets to disk")  


def main() -> None:
    user_args = _parse_args()
    config = Config(
        split_mode=user_args.split_mode,
        source_files_language=user_args.source_files_language,
        extensions=user_args.extensions,
        raw_data_path=user_args.raw_data_path,
        tree_sitter_parser_path=user_args.tree_sitter_parser_path
    )
    _setup_logger(user_args.log_level)
    run(config=config, split_mode=user_args.split_mode)


if __name__ == "__main__":
    main()
    