import json
import argparse
import ctypes

from pathlib import Path
from typing import Iterator, Tuple, Mapping, List
from dataclasses import dataclass, field

from transformers import AutoTokenizer
import numpy as np
import tree_sitter as ts 
from tree_sitter_language_pack import get_parser

CodeBlocks = list[Tuple[bytes, ts.Node]]

@dataclass
class Config:
    model_name: str = "Qwen/Qwen2.5-Coder-7B"
    fim_prefix_token: str = "<|fim_prefix|>"
    fim_middle_token: str = "<|fim_middle|>"
    fim_suffix_token: str = "<|fim_suffix|>"
    fim_pad_token: str = "<|fim_pad|>"
    byte_per_token_ratio: int = 3  # Assuming a byte per token ratio of 3
    bytes_per_code_block: int = 500 * byte_per_token_ratio   
    tokenizer_batch_size: int = 32 
    fim_examples_per_subblock_ratio: float = 1.0 
    train_ratio: float = 0.8
    eval_ratio: float = 0.1
    test_ratio: float = 0.1
    rng_seed: int = 0 
    rng: np.random.Generator = field(init=False)

    source_files_language: str = "c"
    extensions: list[str] = field(default_factory=lambda: [".c", ".h"])
    raw_data_path: Path | None = None
    split_mode: str = "auto"
    tree_sitter_parser_path: Path | None = None

    project_root_path: Path = field(init=False) 
    train_path: Path = field(init=False)
    eval_path: Path = field(init=False)
    test_path: Path = field(init=False)
    tree_sitter_parser: ts.Parser = field(init=False)
    block_types: set[str] = field(init=False)
    subblock_types: set[str] = field(init=False)

    def __post_init__(self):
        total_ratio = self.train_ratio + self.eval_ratio + self.test_ratio
        if not np.isclose(total_ratio, 1.0):
            raise ValueError(f"train+eval+test ratios must sum to 1.0, got {total_ratio}")

        self.project_root_path = Path(__file__).resolve().parent.parent 
        if self.raw_data_path is None:
            self.raw_data_path = self.project_root_path / "data" 
        self.train_path = self.project_root_path / "datasets" / "train_dataset.jsonl"
        self.eval_path = self.project_root_path / "datasets" / "eval_dataset.jsonl"
        self.test_path = self.project_root_path / "datasets" / "test_dataset.jsonl"
        
        self.rng = np.random.default_rng(seed=self.rng_seed)

        blocks_path = self.project_root_path / "config" / "language_block_definitions.json"
        with open(blocks_path, "r", encoding="utf-8") as f:
            language_data = json.load(f)

        language_blocks = language_data.get(self.source_files_language)
        if language_blocks is None:
            raise ValueError(f"Source code language '{self.source_files_language}' not found in {blocks_path}")

        self.block_types = set(language_blocks["block_types"])
        self.subblock_types = set(language_blocks["subblock_types"])

        if self.tree_sitter_parser_path:
            self.tree_sitter_parser = _get_custom_tree_sitter_parser(self.tree_sitter_parser_path, self.source_files_language)
        else:
            self.tree_sitter_parser = get_parser(self.source_files_language)

def _normalize_extension(ext: str) -> str:
    ext = ext.strip().lower()
    return ext if ext.startswith(".") else f".{ext}"

def parse_args() -> argparse.Namespace:
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
    for f in dataset_files:
        if f.exists():
            found_existing = True

    if not found_existing:
        return

    confirm = input("Found existing files. Delete and start fresh? (y/n): ").lower()

    if confirm == 'y':
        for f in dataset_files:
            if f.exists():
                f.unlink()
        print("Old data deleted.")
    else:
        exit()

def _get_custom_tree_sitter_parser(tree_sitter_lib_path: Path, source_files_language: str) -> ts.Parser:
        lib = ctypes.CDLL(str(tree_sitter_lib_path)) # Load C Dynamic Link Library, makes all the public C functions inside the .dylib file available to be called from the Python script.
        entry_point_func_name = f"tree_sitter_{source_files_language}"
        lang_func = getattr(lib, entry_point_func_name) # Retrieves the entry point function of the loaded lib
        lang_func.restype = ctypes.c_void_p # Tells ctypes that this function returns a C-style pointer (void *)
        grammar_rules = ts.Language(lang_func()) # Call lang_func() function and wrap the returned raw C pointer into a tree-sitter Language object
        return ts.Parser(grammar_rules) 
        
def _is_utf8_file(filepath: Path) -> bool:
    try:
        with open(filepath, 'rb') as f:
            f.read().decode("utf8")
        return True
    except UnicodeDecodeError:
        return False

def auto_create_split_paths(config: Config) -> Tuple[list[Path], list[Path], list[Path]]:
    all_file_paths = []
    for filepath in config.raw_data_path.rglob("*"):
        if not (
            filepath.is_file() 
            and any(filepath.suffix.lower() == ext for ext in config.extensions)
            and _is_utf8_file(filepath)
            ):
            continue
        all_file_paths.append(filepath)

    config.rng.shuffle(all_file_paths)

    num_files = len(all_file_paths)
    train_end = int(num_files * config.train_ratio)
    eval_end = train_end + int(num_files * config.eval_ratio)

    train_file_paths = all_file_paths[:train_end]
    eval_file_paths = all_file_paths[train_end:eval_end]
    test_file_paths = all_file_paths[eval_end:]

    print(f"Split {num_files} files into {len(train_file_paths)} train, {len(eval_file_paths)} eval, and {len(test_file_paths)} test files.")

    return train_file_paths, eval_file_paths, test_file_paths

def _extract_code_blocks(config: Config, node: ts.Node, source_code_utf8: bytes) -> CodeBlocks:
    code_blocks = []

    if node.type in config.block_types:
        code_utf8 = source_code_utf8[node.start_byte:node.end_byte]
        code_block = (code_utf8, node)
        code_blocks.append(code_block)

    for child_node in node.children:
        child_code_blocks = _extract_code_blocks(config, child_node, source_code_utf8)
        code_blocks.extend(child_code_blocks)

    return code_blocks

def _get_code_blocks_from_paths(config: Config, file_paths: list[Path]) -> Iterator[Tuple[bytes, ts.Node]]:
    """
    Generator function yielding code blocks from files one-by-one.

    How generators work:
    1. Call function: get iterator object (doesn't run code yet)
    2. next(iterator): processes 1 file, yields first block, then pauses  
    3. next(iterator): resumes, yields next block, then pauses
    4. Repeat until end of code blocks, then the generator is exhausted. Generators can only be used once.
    """
    for path in file_paths:
        if not path.is_file():   
            continue
        try:
            source_code_unicode = path.read_text(encoding='utf8')
        except UnicodeDecodeError:
            print(f"Skipping file '{path}': Not a valid UTF-8 file.")
            continue

        source_code_utf8 = source_code_unicode.encode('utf8')
        try:
            tree = config.tree_sitter_parser.parse(source_code_utf8)
        except Exception as exc:
            print(f"Skipping file '{path}': failed to parse with tree-sitter: {exc}")
            continue
        root_node = tree.root_node

        code_blocks = _extract_code_blocks(config, root_node, source_code_utf8)
        config.rng.shuffle(code_blocks)
        for block in code_blocks:
            yield block  # Yields one block at the time 

def get_code_blocks_from_auto_split(config: Config) -> Tuple[Iterator[Tuple[bytes, ts.Node]], Iterator[Tuple[bytes, ts.Node]], Iterator[Tuple[bytes, ts.Node]]]:
    train_file_paths, eval_file_paths, test_file_paths = auto_create_split_paths(config)
    train_code_blocks_iter = _get_code_blocks_from_paths(config, train_file_paths)
    eval_code_blocks_iter = _get_code_blocks_from_paths(config, eval_file_paths)  
    test_code_blocks_iter =  _get_code_blocks_from_paths(config, test_file_paths)
    return train_code_blocks_iter, eval_code_blocks_iter, test_code_blocks_iter 

def _check_required_directories(root_path: Path, required_dirs: list[str]):
    missing_paths = []
    for dir_name in required_dirs:
        dir_path = root_path / dir_name
        
        if not dir_path.is_dir():
            missing_paths.append(str(dir_path))
    
    if missing_paths:
        raise FileNotFoundError(
            f"Required directories under {root_path}. "
            f"Missing directories: {', '.join(missing_paths)}"
        )
    
def _get_filtered_paths(config: Config, directory: Path) -> list[Path]:
    filtered_paths = []
    
    for entry in directory.rglob("*"):
        filename = entry.name.lower()
        if any(filename.endswith(ext) for ext in config.extensions):
            filtered_paths.append(entry)

    return filtered_paths

def get_code_blocks_from_manual_split(config: Config) -> Tuple[Iterator[Tuple[bytes, ts.Node]], Iterator[Tuple[bytes, ts.Node]], Iterator[Tuple[bytes, ts.Node]]]:
    _check_required_directories(config.raw_data_path, ["train", "eval", "test"])
    train_file_paths = _get_filtered_paths(config, config.raw_data_path / "train")
    eval_file_paths  = _get_filtered_paths(config, config.raw_data_path / "eval")
    test_file_paths  = _get_filtered_paths(config, config.raw_data_path / "test")

    train_code_blocks_iter = _get_code_blocks_from_paths(config, train_file_paths)
    eval_code_blocks_iter = _get_code_blocks_from_paths(config, eval_file_paths) 
    test_code_blocks_iter = _get_code_blocks_from_paths(config, test_file_paths) 
    return train_code_blocks_iter, eval_code_blocks_iter, test_code_blocks_iter

def _extract_subblock_ranges(config: Config, node: ts.Node, base_offset: int) -> list[Tuple[int, int]]:
    """
    Recursively depth-first search (DFS) the Abstract syntax tree (AST) and collect all subblock indices.
    Returns subblock ranges relative to the containing code block.
    E.g., a subblock range of (2, 30) means the subblock starts at byte position 2
    and ends at byte position 30, counting from the beginning of the specific code block
    (not the file).
    """
    subblock_ranges = []

    if node.type in config.subblock_types:
        relative_start_byte = node.start_byte - base_offset
        relative_end_byte = node.end_byte - base_offset
        subblock_ranges.append((relative_start_byte, relative_end_byte))
    for child in node.children:
        child_subblock_ranges = _extract_subblock_ranges(config, child, base_offset) 
        subblock_ranges.extend(child_subblock_ranges)
        
    return subblock_ranges

def _filter_subblocks(subblock_ranges: list[Tuple[int, int]], max_bytes: int) -> list[Tuple[int, int]]:
    """
     Discard subblocks that have a larger end index than max_bytes
    """
    subblock_ranges = sorted(subblock_ranges, key=lambda x: x[1]) # Sort ranges by end index
    i = 0
    while i < len(subblock_ranges) and subblock_ranges[i][1] <= max_bytes:
        i += 1
    return subblock_ranges[:i]

def _generate_fim_examples_from_code_block(config: Config, code_utf8: bytes, subblock_ranges: list[Tuple[int, int]]) -> list[bytes]:
    fim_prefix_token_utf8 = config.fim_prefix_token.encode('utf8')
    fim_middle_token_utf8 = config.fim_middle_token.encode('utf8')
    fim_suffix_token_utf8 = config.fim_suffix_token.encode('utf8')

    num_of_subblocks = len(subblock_ranges)
    num_of_fim_examples = 0
    if (config.fim_examples_per_subblock_ratio >= 1):
        num_of_fim_examples = num_of_subblocks
    else:
        num_of_fim_examples = max(1, int(num_of_subblocks * config.fim_examples_per_subblock_ratio)) # Make sure to always generate at least one fim example
    unique_random_indices = config.rng.choice(len(subblock_ranges), size=num_of_fim_examples, replace=False)

    fim_examples = [] 

    for idx in unique_random_indices:
        middle_start = subblock_ranges[idx][0]
        middle_end = subblock_ranges[idx][1]

        prefix = code_utf8[:middle_start]
        middle = code_utf8[middle_start:middle_end]
        suffix = code_utf8[middle_end:]

        fim_example = (
            fim_prefix_token_utf8 + prefix +
            fim_suffix_token_utf8 + suffix +
            fim_middle_token_utf8 + middle
        )

        fim_examples.append(fim_example)

    return fim_examples

def create_fim_examples(config: Config, code_blocks_iter: Iterator[Tuple[bytes, ts.Node]]) -> Iterator[bytes]:
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

        subblock_ranges = _filter_subblocks(subblock_ranges, config.bytes_per_code_block ) 
        
        code_block_utf8 = code_block_utf8[:config.bytes_per_code_block]  # Trunctate code block code if it is larger than bytes_per_code_block, we only consider subblocks inside this range

        fim_examples = _generate_fim_examples_from_code_block(config, code_block_utf8, subblock_ranges)
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
        sequence_labels = [-100] * len(sequence)  # Initialize labels with -100 (pytorchignore index)

        middle_token_idx = _find_first_token_idx(sequence, fim_middle_token_id)
        if middle_token_idx == -1:
            continue # Middle token not found, skip to next sequence

        middle_start_idx = middle_token_idx + 1

        # Copy tokens after the middle token to labels
        for j in range(middle_start_idx, len(sequence)):
            sequence_labels[j] = sequence[j]
        labels.append(sequence_labels)

    return labels

def _save_tokenized_batch_as_jsonl(file_path: Path, batch: Mapping[str, List[List[int]]]):
    file_path.parent.mkdir(parents=True, exist_ok=True)     # Ensure file parent directories exist

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
            f.write(json.dumps(example, ensure_ascii=False) + '\n')  # Ensre utf8 encoding

def tokenize_and_save_fim_examples(config: Config, file_path: Path, fim_examples_iter: Iterator[bytes], tokenizer: AutoTokenizer): 
    batch = []

    for fim_example in fim_examples_iter:
        batch.append(fim_example.decode('utf-8'))
        if (len(batch) == config.tokenizer_batch_size):
            tokenized_batch = tokenizer(
                batch,
                padding=False,
                return_tensors=None,
                return_attention_mask=True
            )
            tokenized_batch["labels"] = _mask_labels(config, tokenized_batch["input_ids"], tokenizer)
            _save_tokenized_batch_as_jsonl(file_path, tokenized_batch)
            batch = []
    # last batch only
    if batch:
        tokenized_batch = tokenizer(
            batch,
            padding=False,
            return_tensors=None,
            return_attention_mask=True
        )
        tokenized_batch["labels"] = _mask_labels(config, tokenized_batch["input_ids"], tokenizer)
        _save_tokenized_batch_as_jsonl(file_path, tokenized_batch)

def main():
    user_args = parse_args()
    config = Config(
        source_files_language=user_args.source_files_language,
        extensions=user_args.extensions,
        split_mode=user_args.split_mode,
        raw_data_path=user_args.raw_data_path,
        tree_sitter_parser_path = user_args.tree_sitter_parser_path

    )
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = config.fim_pad_token  # Use FIM pad token for padding instead of default pad token

    _clear_existing_datasets(config)

    if user_args.split_mode == "auto":
        print("Using auto-generated dataset split.")
        train_code_blocks_iter, eval_code_blocks_iter, test_code_blocks_iter = get_code_blocks_from_auto_split(config) 
    elif user_args.split_mode == "manual":
        print("Using manual dataset split from directories.")
        train_code_blocks_iter, eval_code_blocks_iter, test_code_blocks_iter = get_code_blocks_from_manual_split(config) 
    else:
        raise ValueError(f"Unknown split mode: {user_args.split_mode}") 

    train_fim_examples_iter= create_fim_examples(config, train_code_blocks_iter)
    eval_fim_examples_iter = create_fim_examples(config, eval_code_blocks_iter)
    test_fim_examples_iter = create_fim_examples(config, test_code_blocks_iter)
    
    tokenize_and_save_fim_examples(config, config.train_path, train_fim_examples_iter, tokenizer)
    tokenize_and_save_fim_examples(config, config.eval_path, eval_fim_examples_iter, tokenizer)
    tokenize_and_save_fim_examples(config, config.test_path, test_fim_examples_iter, tokenizer)

    print("Saved train, eval, test datasets to disk")  
   

if __name__ == "__main__":
    main()
