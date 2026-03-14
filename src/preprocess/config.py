from dataclasses import dataclass, field
import json
from pathlib import Path

import numpy as np
import tree_sitter as ts


@dataclass
class Config:
    # --- Model ---
    model_name: str = "Qwen/Qwen2.5-Coder-7B"
    fim_prefix_token: str = "<|fim_prefix|>"
    fim_middle_token: str = "<|fim_middle|>"
    fim_suffix_token: str = "<|fim_suffix|>"
    fim_pad_token: str = "<|fim_pad|>"
    eos_token: str = "<|endoftext|>"
    tokenizer_batch_size: int = 32

    # --- Example Creation ---
    source_files_language: str = "c"
    extensions: list[str] = field(default_factory=lambda: [".c", ".h"])
    split_mode: str = "auto"
    train_ratio: float = 0.8
    eval_ratio: float = 0.1
    test_ratio: float = 0.1
    max_token_sequence_length: int = 1024  # used with bytes_per_token_ratio to convert bytes to tokens, final token count is thus not exact
    max_code_blocks_ast_depth: int = 2  # depth 1 is root, 2 includes child nodes (e.g. functions)
    min_middle_tokens_length: int = 20  # used with estimated bytes_per_token_ratio to convert bytes to tokens, final token count is thus not exact 
    max_middle_tokens_length: int = 200  # used with estimated bytes_per_token_ratio to convert bytes to tokens, final token count is thus not exact
    fim_examples_per_subblock_ratio: float = 1.0  # 1.0 = all fim examples of a subblock are extracted, 0.5 = onls 50% of fim examples of a subblock are extracted

    # --- Tree Sitter Parser ---
    tree_sitter_parser: ts.Parser = field(init=False)
    tree_sitter_parser_path: Path | None = None
    tree_sitter_block_types: set[str] = field(init=False)
    tree_sitter_subblock_types: set[str] = field(init=False)

    # --- Paths ---
    project_root_path: Path = field(init=False) 
    raw_data_path: Path = field(init=False) 
    train_path: Path = field(init=False)
    eval_path: Path = field(init=False)
    test_path: Path = field(init=False)

    # --- Randomization ---
    rng_seed: int = 0 
    rng: np.random.Generator = field(init=False)


    def __post_init__(self):
        self._validate_ratio()
        self._setup_paths()
        self._load_language_blocks()
        self._init_tree_sitter_parser()
        self.rng = np.random.default_rng(seed=self.rng_seed)
    
    def _validate_ratio(self):
        total_ratio = self.train_ratio + self.eval_ratio + self.test_ratio
        if not np.isclose(total_ratio, 1.0):
            raise ValueError(f"Train + eval + test ratios must sum to 1.0, got {total_ratio}")

    def _setup_paths(self):
        self.project_root_path = Path(__file__).resolve().parents[2]
        self.raw_data_path = self.project_root_path / "data" 
        self.train_path = self.project_root_path / "datasets" / "train_dataset.jsonl"
        self.eval_path = self.project_root_path / "datasets" / "eval_dataset.jsonl"
        self.test_path = self.project_root_path / "datasets" / "test_dataset.jsonl"

 
    def _load_language_blocks(self) -> None:
        blocks_path = self.project_root_path / "config" / "language_block_definitions.json"
        with open(blocks_path, "r", encoding="utf-8") as f:
            language_data = json.load(f)

        language_blocks = language_data.get(self.source_files_language)
        if language_blocks is None:
            raise ValueError(f"Language '{self.source_files_language}' not found in {blocks_path}")

        tree_sitter_block_types = language_blocks.get("block_types")
        tree_sitter_subblock_types = language_blocks.get("subblock_types")
        if not isinstance(tree_sitter_block_types, list) or not isinstance(tree_sitter_subblock_types, list):
            raise ValueError(f"Invalid block definitions for '{self.source_files_language}' in {blocks_path}")

        self.tree_sitter_block_types = set(tree_sitter_block_types)
        self.tree_sitter_subblock_types = set(tree_sitter_subblock_types)

    def _init_tree_sitter_parser(self) -> None:
        if self.tree_sitter_parser_path:
            from .extractor import get_custom_tree_sitter_parser
            self.tree_sitter_parser = get_custom_tree_sitter_parser(self.tree_sitter_parser_path, self.source_files_language)
        else:
            from .extractor import get_tree_sitter_language_pack_parser
            self.tree_sitter_parser = get_tree_sitter_language_pack_parser(self.source_files_language)
        
        if self.tree_sitter_parser is None:
            raise RuntimeError("Tree-sitter parser not initialized")
