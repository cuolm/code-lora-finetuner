import argparse
import logging.config
from pathlib import Path

from transformers import AutoTokenizer

from .config import Config
from .extractor import get_custom_tree_sitter_parser, get_tree_sitter_language_pack_parser, get_code_blocks_from_auto_split, get_code_blocks_from_manual_split
from .processor import (create_fim_examples, estimate_bytes_per_token_ratio,
                        tokenize_and_save_fim_examples)


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


def _normalize_extension(ext: str) -> str:
    ext = ext.strip().lower()
    return ext if ext.startswith(".") else f".{ext}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess code dataset for FIM fine-tuning.")
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
        help="Optional path to a custom compiled Tree-sitter shared library file (.so, .dylib, .dll). If not set, the parser is loaded from the standard language pack."
    )

    return parser.parse_args()


def _clear_existing_datasets(config: Config) -> None:
    dataset_files = [config.train_path, config.eval_path, config.test_path]
    
    found_existing = False
    for file in dataset_files:
        if file.exists():
            logger.info(f"Found existing dataset: {file}")
            found_existing = True

    if not found_existing:
        return

    confirm = input("Found existing files. Delete and start fresh? (y/n): ").lower()

    if confirm == 'y':
        for f in dataset_files:
            if f.exists():
                f.unlink()
        logger.info("Deleted existing datasets.")
    else:
        logger.info("Aborted by user. Existing datasets preserved.")
        exit()


def main() -> None:
    _setup_logger("INFO")
    user_args = _parse_args()
    config = Config(
        source_files_language=user_args.source_files_language,
        extensions=user_args.extensions,
        split_mode=user_args.split_mode,
        raw_data_path=user_args.raw_data_path,
        tree_sitter_parser_path = user_args.tree_sitter_parser_path

    )

    if user_args.tree_sitter_parser_path:
        config.tree_sitter_parser = get_custom_tree_sitter_parser(config.tree_sitter_parser_path, config.source_files_language)
    else:
        config.tree_sitter_parser = get_tree_sitter_language_pack_parser(config.source_files_language)
    
    if config.tree_sitter_parser is None:
        err_msg = "Tree-sitter parser not initialized" 
        logger.error(err_msg)
        raise RuntimeError(err_msg)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = config.fim_pad_token  # use FIM pad token for padding instead of default pad token

    _clear_existing_datasets(config)

    if user_args.split_mode == "auto":
        logger.info("Using auto-generated dataset split.")
        train_code_blocks_iter, eval_code_blocks_iter, test_code_blocks_iter = get_code_blocks_from_auto_split(config) 
    elif user_args.split_mode == "manual":
        logger.info("Using manual dataset split from directories.")
        train_code_blocks_iter, eval_code_blocks_iter, test_code_blocks_iter = get_code_blocks_from_manual_split(config) 
    else:
        err_msg = f"Unknown split mode: {user_args.split_mode}"
        logger.error(err_msg)
        raise ValueError(err_msg) 

    bytes_per_token_ratio = estimate_bytes_per_token_ratio(config, tokenizer, number_of_code_blocks=20000)
    logger.info(f"Estimated bytes_per_token_ratio: {bytes_per_token_ratio}")

    train_fim_examples_iter= create_fim_examples(config, train_code_blocks_iter, bytes_per_token_ratio)
    eval_fim_examples_iter = create_fim_examples(config, eval_code_blocks_iter, bytes_per_token_ratio)
    test_fim_examples_iter = create_fim_examples(config, test_code_blocks_iter, bytes_per_token_ratio)
    
    tokenize_and_save_fim_examples(config, config.train_path, train_fim_examples_iter, tokenizer)
    tokenize_and_save_fim_examples(config, config.eval_path, eval_fim_examples_iter, tokenizer)
    tokenize_and_save_fim_examples(config, config.test_path, test_fim_examples_iter, tokenizer)

    logger.info("Saved train, eval, test datasets to disk")  

   
if __name__ == "__main__":
    main()