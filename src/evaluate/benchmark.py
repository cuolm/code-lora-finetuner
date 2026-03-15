import argparse
import json
import logging
import re

from datasets import Features, Sequence, Value, load_dataset
from transformers import AutoTokenizer

from .config import Config


logger = logging.getLogger("src.evaluate.benchmark")


def ensure_benchmark_dataset(config: Config, user_args: argparse.Namespace) -> None:
    if user_args.overwrite_dataset or not config.benchmark_dataset_path.exists():
        dataset_len = _create_benchmark_dataset(config)
        logger.info(f"Created new benchmark dataset '{config.benchmark_dataset_path}' with '{dataset_len}' examples")
    else:
        logger.info(f"Proceeding with existing file '{config.benchmark_dataset_path}'...")


def _extract_fim_parts(config: Config, decoded_text: str) -> dict:
    """Extracts prefix, suffix, and middle (reference) using FIM tags from config."""
    # Escape tokens for regex in case they contain special characters
    p = re.escape(config.fim_prefix_token)
    s = re.escape(config.fim_suffix_token)
    m = re.escape(config.fim_middle_token)

    prefix_pattern = f"{p}(.*?){s}"
    suffix_pattern = f"{s}(.*?){m}"
    middle_pattern = f"{m}(.*)"

    prefix = re.search(prefix_pattern, decoded_text, re.DOTALL)
    suffix = re.search(suffix_pattern, decoded_text, re.DOTALL)
    middle = re.search(middle_pattern, decoded_text, re.DOTALL)

    return {
        "prefix": prefix.group(1) if prefix else "",
        "suffix": suffix.group(1) if suffix else "",
        "reference_middle": middle.group(1).replace("<|endoftext|>", "").strip() if middle else ""
    }


def _create_benchmark_dataset(config: Config) -> int:
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    dataset_features = Features({
        "input_ids": Sequence(Value("int64")),
        "attention_mask": Sequence(Value("int64")),
        "labels": Sequence(Value("int64")),
    })

    dataset = load_dataset(
        "json", 
        data_files=str(config.test_dataset_path), 
        features=dataset_features, 
        streaming=True
    )["train"]
    
    shuffled_dataset = dataset.shuffle(
        buffer_size=config.benchmark_shuffle_buffer_size,
        seed=config.benchmark_shuffle_seed
    )

    benchmark_examples = []
    added_examples_count = 0
    for data_example in shuffled_dataset:
        if added_examples_count >= config.benchmark_sample_size:
            break
        
        # decode including special FIM tokens for regex extraction
        decoded_text = tokenizer.decode(data_example["input_ids"], skip_special_tokens=False)
        fim_parts = _extract_fim_parts(config, decoded_text)
        if (len(fim_parts["reference_middle"]) < config.benchmark_min_fim_middle_chars or
            len(fim_parts["prefix"].strip()) == 0):
            continue
        benchmark_examples.append(fim_parts)
        added_examples_count += 1

    with open(config.benchmark_dataset_path, 'w', encoding='utf-8') as f:
        for example in benchmark_examples:
            if example["prefix"] or example["reference_middle"]:
                f.write(json.dumps(example) + '\n')
    
    return len(benchmark_examples)
