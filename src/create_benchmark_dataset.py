import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from datasets import load_dataset, Features, Value, Sequence
from transformers import AutoTokenizer


@dataclass
class Config:
    input_dataset_path: Path
    benchmark_dataset_path: Path 
    sample_size: int
    shuffle_buffer_size: int = 10000000
    shuffle_seed: int = 42
    model_name: str = "Qwen/Qwen2.5-Coder-7B"
    fim_prefix_token: str = "<|fim_prefix|>"
    fim_suffix_token: str = "<|fim_suffix|>"
    fim_middle_token: str = "<|fim_middle|>"
    project_root_path: Path = field(init=False)

    def __post_init__(self):
        self.project_root_path = Path(__file__).resolve().parent.parent


def _extract_fim_parts(decoded_text: str, config: Config) -> dict:
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


def create_benchmark_dataset(input_dataset_path: Path, benchmark_dataset_path: Path, sample_size: int, min_fim_middle_chars: int) -> int:
    config = Config(input_dataset_path=input_dataset_path, benchmark_dataset_path=benchmark_dataset_path, sample_size=sample_size)
    benchmark_dataset_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    dataset_features = Features({
        "input_ids": Sequence(Value("int64")),
        "attention_mask": Sequence(Value("int64")),
        "labels": Sequence(Value("int64")),
    })

    dataset = load_dataset(
        "json", 
        data_files=str(config.input_dataset_path), 
        features=dataset_features, 
        streaming=True
    )["train"]
    
    shuffled_dataset = dataset.shuffle(
        buffer_size=config.shuffle_buffer_size, 
        seed=config.shuffle_seed
    )

    benchmark_examples = []
    added_examples_count = 0
    for data_example in shuffled_dataset:
        if added_examples_count >= config.sample_size:
            break
        
        # decode including special FIM tokens for regex extraction
        decoded_text = tokenizer.decode(data_example["input_ids"], skip_special_tokens=False)
        fim_parts = _extract_fim_parts(decoded_text, config)
        if (len(fim_parts["reference_middle"]) < min_fim_middle_chars or
            len(fim_parts["prefix"].strip()) == 0):
            continue
        benchmark_examples.append(fim_parts)
        added_examples_count += 1

    with open(config.benchmark_dataset_path, 'w', encoding='utf-8') as f:
        for example in benchmark_examples:
            if example["prefix"] or example["reference_middle"]:
                f.write(json.dumps(example) + '\n')
    
    return len(benchmark_examples)
