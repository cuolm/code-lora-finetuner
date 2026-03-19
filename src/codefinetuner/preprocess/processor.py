import json
import logging
from pathlib import Path
from typing import Iterator, List, Mapping, Tuple

from transformers import AutoTokenizer
import tree_sitter as ts

from .config import Config
from .extractor import auto_create_split_paths, get_code_blocks_from_paths


logger = logging.getLogger(__name__)


def _extract_subblock_ranges(config: Config, node: ts.Node, base_offset: int) -> list[Tuple[int, int]]:
    """
    Recursively depth-first search (DFS) the Abstract syntax tree (AST) and collect all subblock indices.
    Returns subblock ranges relative to the containing code block.
    E.g., a subblock range of (2, 30) means the subblock starts at byte position 2
    and ends at byte position 30, counting from the beginning of the specific code block
    (not the file).
    """
    subblock_ranges = []

    if node.type in config.tree_sitter_subblock_types:
        relative_start_byte = node.start_byte - base_offset
        relative_end_byte = node.end_byte - base_offset
        subblock_ranges.append((relative_start_byte, relative_end_byte))
    for child in node.children:
        child_subblock_ranges = _extract_subblock_ranges(config, child, base_offset) 
        subblock_ranges.extend(child_subblock_ranges)
        
    return subblock_ranges


def _filter_subblocks(subblock_ranges: list[Tuple[int, int]], max_bytes_per_subblock: int) -> list[Tuple[int, int]]:
    """
     Discard subblocks that have a larger end index than max_bytes_per_subblock
    """
    subblock_ranges = sorted(subblock_ranges, key=lambda x: x[1]) # sort ranges by end index
    i = 0
    while i < len(subblock_ranges) and subblock_ranges[i][1] <= max_bytes_per_subblock:
        i += 1
    return subblock_ranges[:i]


def estimate_bytes_per_token_ratio(config: Config, tokenizer: AutoTokenizer, number_of_code_blocks: int) -> float:
    """
    Estimate bytes per token ratio from first `number_of_code_blocks` code blocks in training split.
    Always uses auto-split (ignores config.split_mode). 
    Source code is almost entirely ASCII characters (1 char = 1 byte) 
    except e.g. specific string literals (printf("π ≈ 3.14159\n");). 
    ASCII is a subset of UTF-8, so len(bytes) ≈ character count.
    """
    train_file_paths, _, _ = auto_create_split_paths(config)  # no matter the split mode always use the auto split
    total_bytes = 0
    total_tokens = 0
    i = 0
    block_iter = get_code_blocks_from_paths(config, train_file_paths)
    for block in block_iter:
        total_bytes += len(block[0])  # bytes from code block
        tokenized_block = tokenizer(block[0].decode('utf-8')).tokens()
        total_tokens += len(tokenized_block)
        i += 1
        if i >= number_of_code_blocks:
            break
    
    if total_tokens <= 0 or total_bytes <= 0:
        raise ValueError("Failed to estimate bytes per token ratio")
    bytes_per_token_ratio = total_bytes / total_tokens 
    return bytes_per_token_ratio 


def _generate_fim_examples_from_code_block(config: Config, code_utf8: bytes, subblock_ranges: list[Tuple[int, int]], bytes_per_token_ratio: float) -> list[bytes]:
    fim_prefix_token_utf8 = config.fim_prefix_token.encode('utf8')
    fim_middle_token_utf8 = config.fim_middle_token.encode('utf8')
    fim_suffix_token_utf8 = config.fim_suffix_token.encode('utf8')
    eos_token_utf8 = config.eos_token.encode('utf8')
    
    num_of_subblocks = len(subblock_ranges)
    num_of_fim_examples = max(1, int(num_of_subblocks * config.fim_examples_per_subblock_ratio))  
    unique_random_subblock_indices = config.rng.choice(len(subblock_ranges), size=num_of_fim_examples, replace=False)

    fim_examples = [] 

    for idx in unique_random_subblock_indices:
        middle_start_byte = subblock_ranges[idx][0]
        middle_end_byte = subblock_ranges[idx][1]
        middle_bytes_length = middle_end_byte - middle_start_byte
        middle_tokens_length = middle_bytes_length / bytes_per_token_ratio
        # Allow only examples within a certain range
        if (middle_tokens_length < config.min_middle_tokens_length) or (middle_tokens_length > config.max_middle_tokens_length):
            continue

        prefix = code_utf8[:middle_start_byte]
        middle = code_utf8[middle_start_byte:middle_end_byte]
        suffix = code_utf8[middle_end_byte:]

        fim_example = (
            fim_prefix_token_utf8 + prefix +
            fim_suffix_token_utf8 + suffix +
            fim_middle_token_utf8 + middle +
            eos_token_utf8
        )

        fim_examples.append(fim_example)

    return fim_examples


def create_fim_examples(config: Config, code_blocks_iter: Iterator[Tuple[bytes, ts.Node]], bytes_per_token_ratio: float) -> Iterator[bytes]:
    """
    Generator function that takes a generator iterator of code blocks as input, 
    generates FIM examples from each code block and returns a generator iterator 
    over all created FIM examples. 
    """
    for code_block_utf8, node in code_blocks_iter:
        base_offset = node.start_byte
        subblock_ranges = _extract_subblock_ranges(config, node, base_offset)
        
        if not subblock_ranges:
            continue 

        max_bytes_per_subblock = int(config.max_token_sequence_length * bytes_per_token_ratio)

        subblock_ranges = _filter_subblocks(subblock_ranges, max_bytes_per_subblock) 
        
        code_block_utf8 = code_block_utf8[:max_bytes_per_subblock]  # Trunctate code block code if it is larger than bytes_per_code_block, we only consider subblocks inside this range.

        fim_examples = _generate_fim_examples_from_code_block(config, code_block_utf8, subblock_ranges, bytes_per_token_ratio)
        for fim_example in fim_examples:
            yield fim_example


def _find_first_token_idx(sequence: List[int], token_id: int) -> int:
    for idx, token in enumerate(sequence):
        if token == token_id:
            return idx
    return -1


def _mask_labels(config: Config, input_ids: List[List[int]], tokenizer: AutoTokenizer) -> List[List[int]]:
    fim_middle_token_id = tokenizer.convert_tokens_to_ids(config.fim_middle_token)
    fim_pad_token_id = tokenizer.convert_tokens_to_ids(config.fim_pad_token)

    labels = []

    for idx, sequence in enumerate(input_ids):
        sequence_labels = [-100] * len(sequence)  # initialize labels with -100 (pytorchignore index)

        middle_token_idx = _find_first_token_idx(sequence, fim_middle_token_id)
        if middle_token_idx == -1:
            continue  # middle token not found, skip to next sequence

        middle_start_idx = middle_token_idx + 1

        # copy tokens after the middle token to labels
        for j in range(middle_start_idx, len(sequence)):
            sequence_labels[j] = sequence[j]
        labels.append(sequence_labels)

    return labels


def _save_tokenized_batch_as_jsonl(file_path: Path, batch: Mapping[str, List[List[int]]]) -> None:
    with open(file_path, 'a', encoding='utf-8') as f:
        batch_size = len(batch['input_ids'])
        for i in range(batch_size):
            input_ids = batch['input_ids'][i]
            attention_mask = batch['attention_mask'][i]
            labels = batch['labels'][i]
            
            example = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels
            }
            f.write(json.dumps(example, ensure_ascii=False) + '\n')  # ensre utf8 encoding


def tokenize_and_save_fim_examples(config: Config, file_path: Path, fim_examples_iter: Iterator[bytes], tokenizer: AutoTokenizer) -> None: 
    examples_counter = 0
    batch = []
    for fim_example in fim_examples_iter:
        batch.append(fim_example.decode('utf-8'))
        examples_counter += 1
        if (len(batch) == config.tokenizer_batch_size):
            tokenized_batch = tokenizer(
                batch,
                padding=False,
                return_tensors=None,
                return_attention_mask=True
            )
            # tokenized_batch["labels"] = _mask_labels(config, tokenized_batch["input_ids"], tokenizer)
            tokenized_batch["labels"] = tokenized_batch["input_ids"]
            _save_tokenized_batch_as_jsonl(file_path, tokenized_batch)
            batch = []

    # last batch is smaller than config.tokenizer_batch_size, the "rest"
    if batch:
        tokenized_batch = tokenizer(
            batch,
            padding=False,
            return_tensors=None,
            return_attention_mask=True
        )
        # tokenized_batch["labels"] = _mask_labels(config, tokenized_batch["input_ids"], tokenizer)
        tokenized_batch["labels"] = tokenized_batch["input_ids"]
        _save_tokenized_batch_as_jsonl(file_path, tokenized_batch)

    logger.info(f"Processed and saved {examples_counter} FIM examples to {file_path}")