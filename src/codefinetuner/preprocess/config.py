import logging
import math
import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import numpy as np
import tree_sitter as ts
from omegaconf import OmegaConf, DictConfig, MISSING


logger = logging.getLogger(__name__)


@dataclass
class Config:
    # --- Mandatory Parameters ---
    model_name: str = MISSING
    fim_prefix_token: str = MISSING
    fim_middle_token: str = MISSING
    fim_suffix_token: str = MISSING
    fim_pad_token: str = MISSING
    eos_token: str = MISSING
    data_language: str = MISSING
    data_extensions: list[str] = MISSING

    # --- Preprocess Local Parameters ---
    split_mode: str = "auto"
    train_ratio: float = 0.8
    eval_ratio: float = 0.1
    test_ratio: float = 0.1
    max_token_sequence_length: int = 1024  # used with bytes_per_token_ratio to convert bytes to tokens, final token count is thus not exact
    max_code_blocks_ast_depth: int = 2  # depth 1 is root, 2 includes child nodes (e.g. functions)
    min_middle_tokens_length: int = 20  # used with estimated bytes_per_token_ratio to convert bytes to tokens, final token count is thus not exact 
    max_middle_tokens_length: int = 200  # used with estimated bytes_per_token_ratio to convert bytes to tokens, final token count is thus not exact
    fim_examples_per_subblock_ratio: float = 1.0  # 1.0 = all fim examples of a subblock are extracted, 0.5 = onls 50% of fim examples of a subblock are extracted
    tokenizer_batch_size: int = 32
    rng_seed: int = 0 

    # --- Tree Sitter Parser ---
    tree_sitter_parser: Any = field(init=False)  # type hint Any because omegaconf does not recognize ts.Parser as a valid type (yaml file cannot contain an object of ts.Parser)
    tree_sitter_parser_path: Path | None = None
    tree_sitter_block_types: Any = field(init=False)  # type hint Any because omegaconf does not support set type
    tree_sitter_subblock_types: Any = field(init=False)  # type hint Any because omegaconf does not support set type

    # --- Paths ---
    project_root_path: Path = field(init=False) 
    raw_data_path: Path | None = None 
    train_dataset_path: Path = field(init=False)
    eval_dataset_path: Path = field(init=False)
    test_dataset_path: Path = field(init=False)

    # --- Randomization ---
    rng: Any = field(init=False)  # type hint Any because omegaconf does not support np.random.Generator type

    @classmethod
    def load_from_yaml(cls, yaml_path: Path) -> "Config":
        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")
        
        config_dict = OmegaConf.structured(cls)
        try:
            yaml_file_node = OmegaConf.load(yaml_path)
        except Exception:
            err_msg = f"Failed to load YAML config {yaml_path}" 
            logger.exception(err_msg)
            raise ValueError(err_msg)
            
        yaml_file_dict = OmegaConf.to_container(yaml_file_node, resolve=True)
        yaml_preprocess_dict = yaml_file_dict.get("preprocess", {})

        yaml_preprocess_valid_dict = {}
        # Filter YAML fields to include only those defined in the Config dataclass.
        # This prevents OmegaConf from raising an AttributeError when encountering 
        # global YAML anchors or keys not present in the current Config dataclass. 
        for field in fields(cls):
            if field.name in yaml_preprocess_dict:
                yaml_preprocess_valid_dict[field.name] = yaml_preprocess_dict[field.name]

        merged_config_dict = OmegaConf.merge(config_dict, yaml_preprocess_valid_dict)
        
        return OmegaConf.to_object(merged_config_dict)

    def __post_init__(self):
        self._validate_ratio()
        self._setup_paths()
        self._ensure_output_paths_exist()
        self._load_language_blocks()
        self._init_tree_sitter_parser()
        self.rng = np.random.default_rng(seed=self.rng_seed)
    
    def _validate_ratio(self):
        total_ratio = self.train_ratio + self.eval_ratio + self.test_ratio
        if not math.isclose(total_ratio, 1.0, rel_tol=1e-6):
            raise ValueError(f"Train + eval + test ratios must sum to 1.0, got {total_ratio}")

    def _setup_paths(self):
        self.project_root_path = Path(__file__).resolve().parents[3]
        if self.raw_data_path is None:
            self.raw_data_path = self.project_root_path / "data"
        self.preprocess_outputs_dir_path = self.project_root_path / "outputs" / "preprocess"
        self.train_dataset_path = self.preprocess_outputs_dir_path / "results" / "datasets" / "train_dataset.jsonl"
        self.eval_dataset_path = self.preprocess_outputs_dir_path / "results" / "datasets" / "eval_dataset.jsonl"
        self.test_dataset_path = self.preprocess_outputs_dir_path / "results" / "datasets" / "test_dataset.jsonl"

    def _ensure_output_paths_exist(self):
        paths = [
            self.preprocess_outputs_dir_path,
            self.train_dataset_path,
            self.eval_dataset_path,
            self.test_dataset_path
        ]

        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
 
    def _load_language_blocks(self) -> None:
        blocks_path = self.project_root_path / "config" / "language_block_definitions.json"
        with open(blocks_path, "r", encoding="utf-8") as f:
            language_data = json.load(f)

        language_blocks = language_data.get(self.data_language)
        if language_blocks is None:
            raise ValueError(f"Language '{self.data_language}' not found in {blocks_path}")

        tree_sitter_block_types = language_blocks.get("block_types")
        tree_sitter_subblock_types = language_blocks.get("subblock_types")
        if not isinstance(tree_sitter_block_types, list) or not isinstance(tree_sitter_subblock_types, list):
            raise ValueError(f"Invalid block definitions for '{self.data_language}' in {blocks_path}")

        self.tree_sitter_block_types = set(tree_sitter_block_types)
        self.tree_sitter_subblock_types = set(tree_sitter_subblock_types)

    def _init_tree_sitter_parser(self) -> None:
        if self.tree_sitter_parser_path:
            from .extractor import get_custom_tree_sitter_parser
            self.tree_sitter_parser = get_custom_tree_sitter_parser(self.tree_sitter_parser_path, self.data_language)
        else:
            from .extractor import get_tree_sitter_language_pack_parser
            self.tree_sitter_parser = get_tree_sitter_language_pack_parser(self.data_language)
        
        if self.tree_sitter_parser is None:
            raise RuntimeError("Tree-sitter parser not initialized")
