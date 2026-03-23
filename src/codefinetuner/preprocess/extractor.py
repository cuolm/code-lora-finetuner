import ctypes
import logging
from pathlib import Path
from typing import Iterator, Tuple

import tree_sitter as ts
from tree_sitter_language_pack import get_parser

from .config import Config


logger = logging.getLogger(__name__)


def get_custom_tree_sitter_parser(tree_sitter_lib_path: Path, data_language: str) -> ts.Parser:
    if not tree_sitter_lib_path.exists():
        raise FileNotFoundError(f"Library not found: {tree_sitter_lib_path}")
    try:
        lib = ctypes.CDLL(str(tree_sitter_lib_path)) # Load C Dynamic Link Library, makes all the public C functions inside the .dylib file available to be called from the Python script.
        entry_point_func_name = f"tree_sitter_{data_language}"
        lang_func = getattr(lib, entry_point_func_name) # Retrieves the entry point function of the loaded lib
        lang_func.restype = ctypes.c_void_p # Tells ctypes that this function returns a C-style pointer (void *)
        grammar_rules = ts.Language(lang_func()) # Call lang_func() function and wrap the returned raw C pointer into a tree-sitter Language object
        return ts.Parser(grammar_rules) 
    except (AttributeError, OSError) as e:
        raise RuntimeError(f"Failed to load custom tree sitter language parser: {e}") from e


def get_tree_sitter_language_pack_parser(data_language: str) -> ts.Parser:
    return get_parser(data_language)


def auto_create_split_paths(config: Config) -> Tuple[list[Path], list[Path], list[Path]]:
    all_file_paths = []
    for filepath in config.raw_data_path.rglob("*"):
        if not filepath.is_file():
            continue
        if filepath.suffix.lower() not in config.data_extensions:
            continue
        all_file_paths.append(filepath)

    config.rng.shuffle(all_file_paths)

    num_files = len(all_file_paths)
    if num_files == 0:
        logger.warning(f"No source files found under {config.raw_data_path} with extensions {config.data_extensions}")

    train_end = int(num_files * config.train_ratio)
    eval_end = train_end + int(num_files * config.eval_ratio)

    train_file_paths = all_file_paths[:train_end]
    eval_file_paths = all_file_paths[train_end:eval_end]
    test_file_paths = all_file_paths[eval_end:]

    logger.info(f"Split {num_files} files into {len(train_file_paths)} train, {len(eval_file_paths)} eval, and {len(test_file_paths)} test files.")

    return train_file_paths, eval_file_paths, test_file_paths


def _extract_code_blocks_rec(config: Config, node: ts.Node, source_code_utf8: bytes, max_depth: int) -> list[Tuple[bytes, ts.Node]]:
    """
    Recursively extract code blocks (e.g. functions) that are used for FIM example generation later 
    """
    if max_depth <= 0:
        return []
    
    code_blocks = []
    if node.type in config.tree_sitter_block_types:
        code_utf8 = source_code_utf8[node.start_byte:node.end_byte]
        code_blocks.append((code_utf8, node))
    
    for child in node.children:
        child_blocks = _extract_code_blocks_rec(config, child, source_code_utf8, max_depth-1)
        code_blocks.extend(child_blocks)  
    return code_blocks


def get_code_blocks_from_paths(config: Config, file_paths: list[Path]) -> Iterator[Tuple[bytes, ts.Node]]:
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
            logger.warning(f"Skipping file '{path}': Not a valid UTF-8 file.")
            continue

        source_code_utf8 = source_code_unicode.encode('utf8')
        try:
            tree = config.tree_sitter_parser.parse(source_code_utf8)
        except Exception as exc:
            logger.warning(f"Skipping file '{path}': failed to parse with tree-sitter: {exc}")
            continue
        root_node = tree.root_node

        code_blocks = _extract_code_blocks_rec(config, root_node, source_code_utf8, max_depth=config.max_code_blocks_ast_depth)
        config.rng.shuffle(code_blocks)  # shuffle code blocks extracted from a single file
        for block in code_blocks:
            yield block  # yields one block at the time 


def get_code_blocks_from_auto_split(config: Config) -> Tuple[Iterator[Tuple[bytes, ts.Node]], Iterator[Tuple[bytes, ts.Node]], Iterator[Tuple[bytes, ts.Node]]]:
    """
    Auto-split source code files from /data into train/eval/test paths, then extract top-level 
    code blocks such as functions from each split.
    """
    train_file_paths, eval_file_paths, test_file_paths = auto_create_split_paths(config)
    train_code_blocks_iter = get_code_blocks_from_paths(config, train_file_paths)
    eval_code_blocks_iter = get_code_blocks_from_paths(config, eval_file_paths)  
    test_code_blocks_iter =  get_code_blocks_from_paths(config, test_file_paths)
    return train_code_blocks_iter, eval_code_blocks_iter, test_code_blocks_iter 


def _check_required_directories(root_path: Path, required_dirs: list[str]) -> None:
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
        if not entry.is_file():
            continue
        if entry.suffix.lower() not in config.data_extensions:
            continue
        filtered_paths.append(entry)

    return filtered_paths


def get_code_blocks_from_manual_split(config: Config) -> Tuple[Iterator[Tuple[bytes, ts.Node]], Iterator[Tuple[bytes, ts.Node]], Iterator[Tuple[bytes, ts.Node]]]:
    """
    Extract top-level code blocks such as functions from manually split train/eval/test files.
    """
    _check_required_directories(config.raw_data_path, ["train", "eval", "test"])
    train_file_paths = _get_filtered_paths(config, config.raw_data_path / "train")
    eval_file_paths  = _get_filtered_paths(config, config.raw_data_path / "eval")
    test_file_paths  = _get_filtered_paths(config, config.raw_data_path / "test")

    train_code_blocks_iter = get_code_blocks_from_paths(config, train_file_paths)
    eval_code_blocks_iter = get_code_blocks_from_paths(config, eval_file_paths) 
    test_code_blocks_iter = get_code_blocks_from_paths(config, test_file_paths) 
    return train_code_blocks_iter, eval_code_blocks_iter, test_code_blocks_iter