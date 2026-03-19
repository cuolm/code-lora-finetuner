import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

import nltk
import torch


logger = logging.getLogger(__name__)


@dataclass
class Config:
    # --- Model ---
    model_name: str = "Qwen/Qwen2.5-Coder-1.5B"
    model_dtype: torch.dtype = field(init=False)
    fim_prefix_token: str = "<|fim_prefix|>"
    fim_suffix_token: str = "<|fim_suffix|>"
    fim_middle_token: str = "<|fim_middle|>"
    fim_pad_token: str = "<|fim_pad|>"
    eos_token: str = "<|endoftext|>"

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

    # --- Hardware Configuration ---
    device: str = field(init=False)
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

    def __post_init__(self):
        self._setup_device_and_precision()
        self._setup_paths()
        self._ensure_output_paths_exist()
        self._validate_metric_weights()

    def _setup_device_and_precision(self):
        if torch.cuda.is_available():
            self.device = "cuda"
            self.model_dtype = torch.bfloat16
        elif torch.backends.mps.is_available():
            self.device = "mps"
            self.model_dtype = torch.float16
        else:
            self.device = "cpu"
            self.model_dtype = torch.float32
        
    def _validate_metric_weights(self):
        codebleu_total_weight = (self.codebleu_ngram_weight + self.codebleu_weighted_ngram_weight + 
                                 self.codebleu_syntax_ast_weight + self.codebleu_dataflow_weight)
        if not math.isclose(codebleu_total_weight, 1.0, rel_tol=1e-6):
            raise ValueError(f"CodeBLEU weights must sum to 1.0, got {codebleu_total_weight}")
        sentencebleu_total_weight = (self.sentencebleu_ngram_weight_1 + self.sentencebleu_ngram_weight_2 + 
                                       self.sentencebleu_ngram_weight_3 + self.sentencebleu_ngram_weight_4) 
        if not math.isclose(sentencebleu_total_weight, 1.0, rel_tol=1e-6):
            raise ValueError(f"SentenceBLEU weights must sum to 1.0, got {sentencebleu_total_weight}")

    def _setup_paths(self):
        self.project_root_path = Path(__file__).resolve().parent.parent.parent
        self.trainer_checkpoints_dir_path = self.project_root_path / "outputs" / "finetune" / "checkpoints"
        self.test_dataset_path = self.project_root_path / "outputs" / "preprocess" / "results" / "datasets" / "test_dataset.jsonl"
        self.evaluate_outputs_dir_path = self.project_root_path / "outputs" / "evaluate"
        self.benchmark_dataset_path = self.evaluate_outputs_dir_path / "datasets" / "benchmark_dataset.jsonl"
        self.benchmark_evaluation_results_dir = self.evaluate_outputs_dir_path / "results"
        self.benchmark_evaluation_results_path = self.benchmark_evaluation_results_dir / "evaluation_results.jsonl"
        self.benchmark_analysis_results_path = self.benchmark_evaluation_results_dir / "analysis_results.json"

    def _ensure_output_paths_exist(self) -> None:
        paths = [
            self.evaluate_outputs_dir_path,
            self.benchmark_dataset_path,
            self.benchmark_evaluation_results_dir,
            self.benchmark_evaluation_results_path,
            self.benchmark_analysis_results_path,
        ]
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)

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
        if not self._nltk_initialized:
            logger.info("Initializing NLTK data (punkt, punkt_tab)...")
            nltk.download('punkt', quiet=True)  
            nltk.download('punkt_tab', quiet=True)
            self._nltk_initialized = True
            logger.info("NLTK data initialized successfully.")
