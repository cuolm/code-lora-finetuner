from dataclasses import dataclass, field
import json
from pathlib import Path

import numpy as np
import tree_sitter as ts


@dataclass
class Config:
    model_name: str = "Qwen/Qwen2.5-Coder-7B"
    fim_prefix_token: str = "<|fim_prefix|>"
    fim_middle_token: str = "<|fim_middle|>"
    fim_suffix_token: str = "<|fim_suffix|>"
    fim_pad_token: str = "<|fim_pad|>"
    eos_token: str = "<|endoftext|>"
    max_token_sequence_length: int = 1024  # used with estimated bytes_per_token_ratio to convert bytes to tokens, final token count is thus not exact
    max_code_blocks_ast_depth: int = 2  # depth 1 is root, 2 includes child nodes (e.g. functions)
    min_middle_tokens_length: int = 20  # used with estimated bytes_per_token_ratio to convert bytes to tokens, final token count is thus not exact
    max_middle_tokens_length: int = 200  # used with estimated bytes_per_token_ratio to convert bytes to tokens, final token count is thus not exact
    fim_examples_per_subblock_ratio: float = 1.0  # 1.0 = all fim examples of a subblock are extracted, 0.5 = onls 50% of fim examples of a subblock are extracted 
    train_ratio: float = 0.8
    eval_ratio: float = 0.1
    test_ratio: float = 0.1
    tokenizer_batch_size: int = 32
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

        self.project_root_path = Path(__file__).resolve().parent.parent.parent 
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
