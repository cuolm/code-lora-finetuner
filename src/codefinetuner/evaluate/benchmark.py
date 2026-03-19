import json
import logging

from datasets import Features, Sequence, Value, load_dataset
from transformers import AutoTokenizer

from .config import Config


logger = logging.getLogger(__name__)


def _extract_fim_parts(config: Config, tokenizer: AutoTokenizer, example_token_ids: list[int]) -> dict:
    fim_prefix_token_id = tokenizer.convert_tokens_to_ids(config.fim_prefix_token)
    fim_suffix_token_id = tokenizer.convert_tokens_to_ids(config.fim_suffix_token)
    fim_middle_token_id = tokenizer.convert_tokens_to_ids(config.fim_middle_token)
    eos_token_id = tokenizer.convert_tokens_to_ids(config.eos_token)

    try:
        fim_prefix_token_position = example_token_ids.index(fim_prefix_token_id)
        fim_suffix_token_position = example_token_ids.index(fim_suffix_token_id)
        fim_middle_token_position = example_token_ids.index(fim_middle_token_id)
        eos_token_position = example_token_ids.index(eos_token_id)
    except ValueError:
        raise ValueError("Missing FIM or EOS tokens in the sequence.")

    if (fim_prefix_token_position >= fim_suffix_token_position  
        or fim_suffix_token_position >= fim_middle_token_position
        or fim_middle_token_position >= eos_token_position):
        raise ValueError("FIM tokens are in the wrong order.")
        
    prefix_token_ids = example_token_ids[fim_prefix_token_position+1 : fim_suffix_token_position]
    suffix_token_ids = example_token_ids[fim_suffix_token_position+1 : fim_middle_token_position]
    middle_token_ids = example_token_ids[fim_middle_token_position+1 : eos_token_position]

    prefix = tokenizer.decode(prefix_token_ids, skip_special_tokens=True)
    suffix = tokenizer.decode(suffix_token_ids, skip_special_tokens=True)
    middle = tokenizer.decode(middle_token_ids, skip_special_tokens=True)

    return {
        "example_token_ids": example_token_ids,
        "prefix_token_ids": prefix_token_ids,
        "suffix_token_ids": suffix_token_ids,
        "middle_token_ids": middle_token_ids,
        "prefix": prefix,
        "suffix": suffix,
        "middle": middle
    }


def create_benchmark_dataset(config: Config) -> int:
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    dataset_features = Features({
        "input_ids": Sequence(Value("int64")),
        "attention_mask": Sequence(Value("int64")),
        "labels": Sequence(Value("int64")),
    })

    test_dataset = load_dataset(
        "json", 
        data_files=str(config.test_dataset_path), 
        features=dataset_features, 
        streaming=True
    )["train"]
    
    shuffled_dataset = test_dataset.shuffle(
        buffer_size=config.benchmark_shuffle_buffer_size,
        seed=config.benchmark_shuffle_seed
    )

    benchmark_examples = []
    added_examples_count = 0
    for idx, example in enumerate(shuffled_dataset):
        if added_examples_count >= config.benchmark_sample_size:
            break
        
        try:
            fim_parts = _extract_fim_parts(config, tokenizer, example["input_ids"])
        except ValueError:
            logger.warning(f"Could not extract fim parts from example {idx}, skipping") 
            continue

        if (len(fim_parts["middle_token_ids"]) < config.benchmark_min_fim_middle_tokens   
            or len(fim_parts["prefix_token_ids"]) == 0):  # skip if no prefix
            continue

        benchmark_examples.append(fim_parts)
        added_examples_count += 1

    with open(config.benchmark_dataset_path, 'w', encoding='utf-8') as f:
        for example in benchmark_examples:
            f.write(json.dumps(example) + '\n')
    
    return len(benchmark_examples)
