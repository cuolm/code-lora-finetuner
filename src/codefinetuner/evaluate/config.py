import logging
import math
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import nltk
import torch
from omegaconf import OmegaConf, MISSING


logger = logging.getLogger(__name__)


@dataclass
class Config:
    # --- Model (Mandatory Parameters) ---
    model_name: str = MISSING  
    fim_prefix_token: str = MISSING
    fim_suffix_token: str = MISSING 
    fim_middle_token: str = MISSING 
    fim_pad_token: str = MISSING 
    eos_token: str = MISSING 
    label_pad_token_id: int = MISSING

    # --- Benchmark ---
    benchmark_sample_size: int = 4
    benchmark_min_fim_middle_tokens: int = 0 
    benchmark_shuffle_buffer_size: int = 10000000
    benchmark_shuffle_seed: int = 42 

    # --- Generation ---
    generation_max_new_tokens: int = 128 
    generation_do_sample: bool = False  # note: if set to False temperature and top_p have no effect
    generation_temperature: float = 0.7
    generation_top_p: float = 0.95

    # --- Metrics: CodeBLEU ---
    codebleu_language: str = "c"
    codebleu_metric_name: str = "codebleu"
    # CodeBLEU weights (must sum to 1.0)
    # see: https://arxiv.org/pdf/2009.10297  Section 4.4 for parameter suggestions 0.1, 0.1, 0.4, 0.4
    codebleu_ngram_weight: float = 0.25  # token-level overlap (standard BLEU) 
    codebleu_weighted_ngram_weight: float = 0.25  # keyword-level overlap (importance-weighted) 
    codebleu_syntax_ast_weight: float = 0.25  # structural correctness (abstract syntax tree)
    codebleu_dataflow_weight: float = 0.25  # logic consistency (variable dependency graph)   

    # --- Metrics: SentenceBLEU ---
    sentencebleu_metric_name: str = "sentencebleu"
    # sentence-BLEU weights (must sum to 1.0)
    sentencebleu_ngram_weight_1: float = 0.25  # 1-gram 
    sentencebleu_ngram_weight_2: float = 0.25  # 2-gram    
    sentencebleu_ngram_weight_3: float = 0.25  # 3-gram 
    sentencebleu_ngram_weight_4: float = 0.25  # 4-gram

    # --- Metrics: Other ---
    exact_match_metric_name: str = "exact_match"
    line_match_metric_name: str = "line_match"
    line_match_number_of_lines: int = 2  
    perplexity_name: str = "perplexity"

    # --- Execution Controls ---
    trainer_checkpoint: str = "last"  # "last"->use last checkpoint in  trainer_checkpoints_dir_path
    plot_only: bool = False  # "skips" generate and evaluate, analyze only
    benchmark_use_existing_dataset: bool = False

    # --- Hardware Configuration ---
    device: str = field(init=False)
    model_dtype: Any = field(init=False)  # # type hint Any because omegaconf does not support torch.dtype 
    _nltk_initialized: bool = field(init=False, default=False)

    # --- Paths ---
    project_root_path: Path = field(init=False)
    trainer_checkpoints_dir_path: Path = field(init=False)
    test_dataset_path: Path = field(init=False)  
    evaluate_outputs_dir_path: Path = field(init=False)
    benchmark_dataset_path: Path = field(init=False)
    benchmark_evaluation_results_dir: Path = field(init=False)
    benchmark_evaluation_results_path: Path = field(init=False)
    benchmark_analysis_results_path: Path = field(init=False)

    @classmethod
    def load_from_yaml(cls, yaml_path: Path) -> "Config":
        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found at: {yaml_path}")
        
        logger.info(f"Loading configuration from {yaml_path}")
        config_dict = OmegaConf.structured(cls)
        try:
            yaml_file_node = OmegaConf.load(yaml_path)
        except Exception as e:
            raise ValueError(f"Failed to load YAML config {yaml_path}") from e

        yaml_file_dict = OmegaConf.to_container(yaml_file_node, resolve=True)
        yaml_evaluate_dict = yaml_file_dict.get("evaluate", {})

        yaml_evaluate_valid_dict = {}
        # Filter YAML fields to include only those defined in the Config dataclass.
        # This prevents OmegaConf from raising an AttributeError when encountering 
        # global YAML anchors or keys not present in the current Config dataclass. 
        for field in fields(cls):
            if field.name in yaml_evaluate_dict:
                yaml_evaluate_valid_dict[field.name] = yaml_evaluate_dict[field.name]
        logger.debug(f"Filtered YAML configuration: {yaml_evaluate_valid_dict}")

        merged_config_dict = OmegaConf.merge(config_dict, yaml_evaluate_valid_dict)
        return OmegaConf.to_object(merged_config_dict)

    def __post_init__(self) -> None:
        self._setup_device_and_precision()
        self._setup_paths()
        self._ensure_output_paths_exist()
        self._validate_metric_weights()
        logger.debug("Config initialization complete.")

    def _setup_device_and_precision(self) -> None:
        if torch.cuda.is_available():
            self.device = "cuda"
            self.model_dtype = torch.bfloat16
        elif torch.backends.mps.is_available():
            self.device = "mps"
            self.model_dtype = torch.float16
        else:
            self.device = "cpu"
            self.model_dtype = torch.float32
        logger.info(f"Execution environment: device={self.device}, dtype={self.model_dtype}")
        
    def _validate_metric_weights(self) -> None:
        codebleu_total_weight = (self.codebleu_ngram_weight + self.codebleu_weighted_ngram_weight + 
                                 self.codebleu_syntax_ast_weight + self.codebleu_dataflow_weight)
        if not math.isclose(codebleu_total_weight, 1.0, rel_tol=1e-6):
            raise ValueError(f"CodeBLEU weights must sum to 1.0, got {codebleu_total_weight}")
        sentencebleu_total_weight = (self.sentencebleu_ngram_weight_1 + self.sentencebleu_ngram_weight_2 + 
                                       self.sentencebleu_ngram_weight_3 + self.sentencebleu_ngram_weight_4) 
        if not math.isclose(sentencebleu_total_weight, 1.0, rel_tol=1e-6):
            raise ValueError(f"SentenceBLEU weights must sum to 1.0, got {sentencebleu_total_weight}")

    def _setup_paths(self) -> None:
        self.project_root_path =  Path(__file__).resolve().parents[3]
        self.trainer_checkpoints_dir_path = self.project_root_path / "outputs" / "finetune" / "checkpoints"
        self.test_dataset_path = self.project_root_path / "outputs" / "preprocess" / "results" / "datasets" / "test_dataset.jsonl"
        self.evaluate_outputs_dir_path = self.project_root_path / "outputs" / "evaluate"
        self.benchmark_dataset_path = self.evaluate_outputs_dir_path / "datasets" / "benchmark_dataset.jsonl"
        self.benchmark_evaluation_results_dir = self.evaluate_outputs_dir_path / "results"
        self.benchmark_evaluation_results_path = self.benchmark_evaluation_results_dir / "evaluation_results.jsonl"
        self.benchmark_analysis_results_path = self.benchmark_evaluation_results_dir / "analysis_results.json"
        logger.debug(f"Resolved project root: {self.project_root_path}")

    def _ensure_output_paths_exist(self) -> None:
        paths = [
            self.evaluate_outputs_dir_path,
            self.benchmark_dataset_path,
            self.benchmark_evaluation_results_dir,
            self.benchmark_evaluation_results_path,
            self.benchmark_analysis_results_path,
        ]
        for path in paths:
            if not path.parent.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                logger.debug(f"Created parent directory: {path.parent}")
            else:
                logger.debug(f"Parent directory already exists: {path.parent}")

    @property
    def metric_configs(self) -> list[tuple[str, bool]]:
        return [
            (self.sentencebleu_metric_name, True),
            (self.codebleu_metric_name, True),
            (self.exact_match_metric_name, True),
            (self.line_match_metric_name, True),
            (self.perplexity_name, False),
        ]
     
    def ensure_nltk_initialized(self) -> None:
        if self._nltk_initialized:
            return

        logger.info("Initializing NLTK data (punkt, punkt_tab)...")
        try:
            nltk.download('punkt', quiet=True)  
            nltk.download('punkt_tab', quiet=True)
            self._nltk_initialized = True
            logger.info("NLTK data initialized successfully.")
        except Exception:
            raise RuntimeError(f"NLTK initializerion failed." )
