import argparse
import torch
import gc
import logging.config
import json
import re
import math
from dataclasses import dataclass, field
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.trainer_utils import get_last_checkpoint
from peft import PeftModel
from metrics.codebleu_adapter import codebleu_score
import matplotlib.pyplot as plt
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
from nltk.tokenize import word_tokenize
from create_benchmark_dataset import create_benchmark_dataset


@dataclass
class Config:
    model_name: str = "Qwen/Qwen2.5-Coder-7B"
    fim_prefix_token: str = "<|fim_prefix|>"
    fim_suffix_token: str = "<|fim_suffix|>"
    fim_middle_token: str = "<|fim_middle|>"
    fim_pad_token: str = "<|fim_pad|>"
    input_sample_size: int = 2 
    min_fim_middle_chars: int = 0  

    gen_max_new_tokens: int = 128 
    gen_do_sample: bool = False  # note: if set to False temperature and top_p have no effect
    gen_temperature: float = 0.7
    gen_top_p: float = 0.95

    project_root_path: Path = field(init=False)
    trainer_output_dir_path: Path = field(init=False)
    input_dataset_path: Path = field(init=False)  # tokenized dataset under /datasets e.g. test_dataset.jsonl
    benchmark_dataset_path: Path = field(init=False)
    base_results_tmp_path: Path = field(init=False)
    evaluation_results_path: Path = field(init=False)
    plot_file: Path = field(init=False)
    device: str = field(init=False)

    # CodeBLEU weights (must sum to 1.0)
    # see: https://arxiv.org/pdf/2009.10297  Section 4.4 for parameter suggestions 0.1, 0.1, 0.4, 0.4
    codebleu_language: str = "c"
    codebleu_score_name: str = "codebleu"
    codebleu_ngram_weight: float = 0.1         # token-level overlap (standard BLEU)
    codebleu_weighted_ngram_weight: float = 0.1 # keyword-level overlap (importance-weighted)
    codebleu_syntax_ast_weight: float = 0.4     # structural correctness (Abstract Syntax Tree)
    codebleu_dataflow_weight: float = 0.4       # logic consistency (Variable dependency graph)
    codebleu_plot_file: Path = field(init=False)

    # sentence-BLEU weights (must sum to 1.0)
    sentencebleu_score_name: str = "sentencebleu"
    sentencebleu_ngram_weight_1: float = 0.25  # 1-gram
    sentencebleu_ngram_weight_2: float = 0.25  # 2-gram  
    sentencebleu_ngram_weight_3: float = 0.25  # 3-gram
    sentencebleu_ngram_weight_4: float = 0.25  # 4-gram
    sentencebleu_plot_file: Path = field(init=False)

    exact_match_score_name: str = "exact_match"
    exact_match_plot_file: Path = field(init=False)

    line_match_score_name: str = "line_match"
    line_match_number_of_lines: int = 2  # 5 is standard production value (Sourcegraph, Cursor IDE)
    line_match_plot_file: Path = field(init=False)

    perplexity_name: str = "perplexity"
    perplexity_plot_file: Path = field(init=False)

    def __post_init__(self):
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        # download punkt and punkt_tab automatically, used for sentencebleu
        nltk.download('punkt', quiet=True)  
        nltk.download('punkt_tab', quiet=True)

        cb_total = (self.codebleu_ngram_weight + self.codebleu_weighted_ngram_weight +
            self.codebleu_syntax_ast_weight + self.codebleu_dataflow_weight)
        if abs(cb_total - 1.0) > 1e-6:
            raise ValueError(f"CodeBLEU weights must sum to 1.0, got {cb_total}")

        self.project_root_path = Path(__file__).resolve().parent.parent
        self.trainer_output_dir_path = self.project_root_path / "results"
        self.input_dataset_path = self.project_root_path / "datasets" / "test_dataset.jsonl"
        self.benchmark_dataset_path = self.project_root_path / "benchmarks" / "pebble_test_examples.jsonl"
        self.base_results_tmp_path = self.project_root_path / "benchmarks" / "results" / "base_results_tmp.jsonl"
        self.evaluation_results_path = self.project_root_path / "benchmarks" / "results" / "evaluation_results.jsonl"
        self.codebleu_plot_file = self.project_root_path / "benchmarks" / "results" / "codebleu_plot.png"
        self.sentencebleu_plot_file = self.project_root_path / "benchmarks" / "results" / "sentencebleu_plot.png"
        self.exact_match_plot_file = self.project_root_path / "benchmarks" / "results" / "exact_match_plot.png"
        self.line_match_plot_file = self.project_root_path / "benchmarks" / "results" / "line_match_plot.png"
        self.perplexity_plot_file = self.project_root_path / "benchmarks" / "results" / "perplexity_plot.png"


logger = logging.getLogger(__name__)

def _setup_logger(log_level: str) -> None:
    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            }
        },
        "handlers": {
            "stderr_handler": {
                "level": log_level,
                "class": "logging.StreamHandler",
                "formatter": "standard",
            }
        },
        "root": {
            "handlers": ["stderr_handler"],
            "level": log_level,
            "propagate": True
        }
    }
    logging.config.dictConfig(config)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate code comparison")
    parser.add_argument("--checkpoint",
                        type=str,
                        required=True,
                        metavar="NAME_OR_LAST",
                        help='Checkpoint name, or "last"')
    parser.add_argument("--plot_only",
                        action="store_true",
                        default=False,  
                        help="Skip generation; use existing data to update plots")
    return parser.parse_args()


def _ensure_benchmark_dataset(config: Config) -> None:
    should_create = True
    if config.benchmark_dataset_path.exists():
        choice = input(f"Benchmark file '{config.benchmark_dataset_path}' already exists. Overwrite? [y/N]: ").lower()
        if choice != 'y':
            should_create = False
            logger.info(f"Proceeding with existing file '{config.benchmark_dataset_path}'...")

    if should_create:
        dataset_len = create_benchmark_dataset(
            input_dataset_path=config.input_dataset_path, 
            benchmark_dataset_path=config.benchmark_dataset_path, 
            sample_size=config.input_sample_size,
            min_fim_middle_chars=config.min_fim_middle_chars 
        )
        logger.info(f"Created new benchmark dataset '{config.benchmark_dataset_path}' with '{dataset_len}' examples")


def _ensure_directories_exist(config: Config) -> None:
    """Create all required output directories."""
    directories = [
        config.base_results_tmp_path.parent,
        config.evaluation_results_path.parent,
    ]
    for dir_path in directories:
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured directory exists: {dir_path}")


def _generate_code(config: Config, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, prompt: str) -> str:
    model.eval()
    input_tokens_dict = tokenizer(prompt, return_tensors="pt").to(config.device)

    bad_words_ids = [
        tokenizer.encode(config.fim_prefix_token, add_special_tokens=False),
        tokenizer.encode(config.fim_middle_token, add_special_tokens=False),
        tokenizer.encode(config.fim_suffix_token, add_special_tokens=False)
    ]

    with torch.inference_mode(): 
        outputs = model.generate(
                input_ids=input_tokens_dict["input_ids"],
                attention_mask=input_tokens_dict["attention_mask"], 
                max_new_tokens=config.gen_max_new_tokens,
                do_sample=config.gen_do_sample,
                temperature=config.gen_temperature,
                top_p=config.gen_top_p,
                bad_words_ids=bad_words_ids,
                pad_token_id=tokenizer.pad_token_id
            )
    

    # Slice the output: take everything after the input_length. generate functin returns the whole example, not only the generated text
    input_length = input_tokens_dict["input_ids"].shape[1]
    generated_tokens = outputs[0][input_length:]

    generated_code = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return generated_code 


def _clear_hardware_cache(config: Config) -> None:
    gc.collect()
    if config.device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    elif config.device == "mps":
        torch.mps.empty_cache()


def _codebleu_structure_valid(config: Config, reference: str) -> bool:
    """
    Checks if the reference code is structurally complex enough for CodeBLEU.
    Performs a self-match test focusing only on Syntax (AST) and Data-flow. 
    If the reference is too simple (e.g., import blocks, comment) 
    to build these structures, the example is skipped. 
    """
    # suppress logger warnings
    root_logger = logging.getLogger()
    original_level = root_logger.getEffectiveLevel()
    root_logger.setLevel(logging.ERROR)
    
    try:
        test_weights = (0.0, 0.0, 0.5, 0.5) 
        result = codebleu_score([reference], [reference], 
                               lang=config.codebleu_language, 
                               weights=test_weights)
    
        syntax_valid = result.get('syntax_match_score', 0) > 0
        dataflow_valid = result.get('dataflow_match_score', 0) > 0
        return syntax_valid and dataflow_valid
    except Exception:
        return False
    finally:
        # restore logger
        root_logger.setLevel(original_level)


def _get_codebleu(config: Config, reference: str, prediction: str) -> tuple[float, bool]:
    """
    CodeBLEU: Computes weighted combination of four different similarity metrics:
    1. N-gram Match: Standard surface-level text overlap.
    2. Weighted N-gram: Text overlap with priority on keywords (if, else, etc.).
    3. Syntax (AST): Structural similarity between Abstract Syntax Trees.
    4. Data-flow: Logical similarity based on variable dependencies and usage.

    Examples are skipped if structural components (AST/Data-flow) cannot be 
    extracted from the reference. This prevents the final average score from 
    being unfairly lowered by snippets that cannot be properly parsed, 
    ensuring a more accurate representation of model quality.

    Standard weights are: [0.25, 0.25, 0.25, 0.25].
    CodeBLEU = (codebleu_ngram_weight * ngram_score) + 
               (codebleu_weighted_ngram_weight * weighted_ngram_score) + 
               (codebleu_syntax_ast_weight * syntax_ast_score) + 
               (codebleu_dataflow_weight * dataflow_score)
    """
    if not _codebleu_structure_valid(config, reference):
        return (0.0, False)
        
    try:
        codebleu_algorithm_weights = (
            config.codebleu_ngram_weight, 
            config.codebleu_weighted_ngram_weight, 
            config.codebleu_syntax_ast_weight, 
            config.codebleu_dataflow_weight
        )
        
        result = codebleu_score(
            [reference], [prediction], 
            lang=config.codebleu_language, 
            weights=codebleu_algorithm_weights
        )
        
        return (float(result['codebleu']), True)
        
    except Exception as e:
        logger.exception(f"ERROR in CodeBLEU calculation: {e}")
        return (0.0, False)


def _get_sentencebleu(config: Config, reference: str, prediction: str) -> float:
    """
    SentenceBLEU: Measures n-gram overlap between reference and prediction.
    It rewards matching sequences of words (1-4) and uses smoothing 
    (Method1: Adds a tiny epsilon to all n-gram counts) to prevent a total 
    0.0 score when long sequences (e.g. 4-grams) don't match exactly.
    """
    try:
        reference_tokens = word_tokenize(reference)
        prediction_tokens = word_tokenize(prediction)
        
        weights = (
            config.sentencebleu_ngram_weight_1, 
            config.sentencebleu_ngram_weight_2, 
            config.sentencebleu_ngram_weight_3, 
            config.sentencebleu_ngram_weight_4
        )
        
        smoothing = SmoothingFunction().method1
        
        score = sentence_bleu(
            [reference_tokens], 
            prediction_tokens, 
            weights=weights, 
            smoothing_function=smoothing
        )
        
        return float(score)
        
    except Exception as e:
        logger.exception(f"ERROR in SentenceBLEU calculation: {e}")
        return 0.0


def _get_exact_match(config: Config, reference: str, prediction: str) -> float:
    """Exact match is 1.0 if identical, 0.0 otherwise. Collapese all whitespaces."""
    try:
        # re.sub(r'\s+', ' ', text.strip()): Collapses whitespace to compare logic regardless of formatting.
        ref_norm = re.sub(r'\s+', ' ', reference.strip())
        pred_norm = re.sub(r'\s+', ' ', prediction.strip())
        
        if ref_norm == pred_norm:
            return 1.0
        else:
            return 0.0
    except Exception as e:
        logger.exception(f"ERROR in exact match calculation: {e}")
        return 0.0


def _get_line_match(config: Config, reference: str, prediction: str) -> float:
    """Check if the first n lines match, ignoring trailing whitespace."""
    try:
        n = config.line_match_number_of_lines

        # line.rstrip(): Removes trailing whitespace while preserving leading indentation.
        ref_lines_stripped = []
        for line in reference.splitlines()[:n]:
            ref_lines_stripped.append(line.rstrip())

        pred_lines_stripped = []
        for line in prediction.splitlines()[:n]:
            pred_lines_stripped.append(line.rstrip())

        # Ensure both lists have the required number of lines
        if len(pred_lines_stripped) < n or len(ref_lines_stripped) < n:
            return 0.0

        if pred_lines_stripped == ref_lines_stripped:
            return 1.0
        else:
            return 0.0
    except Exception as e:
        logger.exception(f"Error in line match: {e}")
        return 0.0


def _get_fim_perplexity(config: Config, model: AutoModelForCausalLM, tokenizer: AutoTokenizer,
                       prefix: str, suffix: str, reference_middle: str) -> float:
    """
    FIM perplexity: Measures model confidence in the reference middle code.
    (How surprised is the model by the reference middle code).
    Lower perplexity indicates higher confidence. perplexity = exp(loss).
    """
    try:
        fim_prompt = (
            f"{config.fim_prefix_token}{prefix}"
            f"{config.fim_suffix_token}{suffix}"
            f"{config.fim_middle_token}"
        )
        
        prompt_tokenized = tokenizer(fim_prompt, return_tensors="pt")
        middle_tokenized = tokenizer(reference_middle, return_tensors="pt")

        prompt_ids = prompt_tokenized.input_ids.to(config.device)
        middle_ids = middle_tokenized.input_ids.to(config.device)

        input_ids = torch.cat([prompt_ids, middle_ids], dim=1)
        
        labels = input_ids.clone()
        prompt_len = prompt_ids.shape[1]
        labels[:, :prompt_len] = -100

        with torch.inference_mode():
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            
        return math.exp(loss.item())
        
    except Exception as e:
        logger.exception(f"ERROR in perplexity calculation: {e}")
        return float('inf')


def _evaluate_models_to_file(config: Config, user_args: argparse.Namespace, tokenizer: AutoTokenizer) -> None:
    if user_args.checkpoint == "last":
        checkpoint_path = get_last_checkpoint(config.trainer_output_dir_path)
    else:
        checkpoint_path = config.trainer_output_dir_path / user_args.checkpoint

    logger.info(f"--- Loading Base Model and LoRA Adapter: {checkpoint_path} ---")
    
    # load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=config.model_name,
        dtype=torch.float16,
        device_map="auto" if config.device == "cuda" else None,
        low_cpu_mem_usage=True
    )
    if config.device != "cuda":
        base_model.to(config.device)

    # load LoRA adapter
    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    model.eval()

    logger.info("--- Starting Generation and Evaluation ---")
    
    # process each example 
    with config.benchmark_dataset_path.open("r") as benchmark_dataset_file, \
         config.evaluation_results_path.open("w") as evaluation_results_file:
        
        for i, line in enumerate(benchmark_dataset_file):
            example = json.loads(line)
            prompt = (
                f"{config.fim_prefix_token}{example['prefix']}"
                f"{config.fim_suffix_token}{example['suffix']}"
                f"{config.fim_middle_token}"
            )
            reference_middle = example["reference_middle"]

            # generation with LoRA augmented model
            lora_generated_middle = _generate_code(config, model, tokenizer, prompt)
            lora_perplexity = _get_fim_perplexity(config, model, tokenizer, example["prefix"], example["suffix"], reference_middle)

            # generation with base model by disabling lora adapter 
            with model.disable_adapter():
                base_generated_middle = _generate_code(config, model, tokenizer, prompt)
                base_perplexity = _get_fim_perplexity(config, model, tokenizer, example["prefix"], example["suffix"], reference_middle)

            # metrics calculation 
            base_codebleu, codebleu_valid = _get_codebleu(config, reference_middle, base_generated_middle)
            lora_codebleu, _ = _get_codebleu(config, reference_middle, lora_generated_middle)
            
            base_sentencebleu = _get_sentencebleu(config, reference_middle, base_generated_middle)
            lora_sentencebleu = _get_sentencebleu(config, reference_middle, lora_generated_middle)

            base_exact_match = _get_exact_match(config, reference_middle, base_generated_middle)
            lora_exact_match = _get_exact_match(config, reference_middle, lora_generated_middle)

            base_line_match = _get_line_match(config, reference_middle, base_generated_middle)
            lora_line_match = _get_line_match(config, reference_middle, lora_generated_middle)

            result = {
                "example_id": i,
                "reference_middle": reference_middle,
                "base_generated_middle": base_generated_middle,
                "lora_generated_middle": lora_generated_middle,
                "base_codebleu": base_codebleu,
                "lora_codebleu": lora_codebleu,
                "codebleu_valid": codebleu_valid,
                "base_sentencebleu": base_sentencebleu,
                "lora_sentencebleu": lora_sentencebleu,
                "base_exact_match": base_exact_match,
                "lora_exact_match": lora_exact_match,
                "base_line_match": base_line_match,
                "lora_line_match": lora_line_match,
                "base_perplexity": base_perplexity,
                "lora_perplexity": lora_perplexity
            }
            evaluation_results_file.write(json.dumps(result) + "\n")

            if i % 10 == 0:
                _clear_hardware_cache(config)
                logger.info(f"Processed Example {i}")

    del model
    del base_model
    _clear_hardware_cache(config)


def _plot_metric_stats_from_file(config: Config, score_name: str, plot_file: Path, higher_is_better: bool) -> None:
    base_scores = []
    lora_scores = []
    
    with open(config.evaluation_results_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            if (score_name == config.codebleu_score_name) and data["codebleu_valid"] is False:
                continue
            base_scores.append(data[f'base_{score_name}'])
            lora_scores.append(data[f'lora_{score_name}'])
    
    if not base_scores:
        logger.error(f"No {score_name} results found.")
        return
    
    # stats
    n_examples = len(base_scores)
    avg_base = sum(base_scores) / n_examples
    avg_lora = sum(lora_scores) / n_examples
    
    if higher_is_better:
        improvement = avg_lora - avg_base
    else:
        improvement = avg_base - avg_lora  # lower is better

    logger.info(f"\n=== {score_name.upper()} SUMMARY ===")
    logger.info(f"Examples: {n_examples}")
    logger.info(f"Base avg: {avg_base:.3f}")
    logger.info(f"LoRA avg: {avg_lora:.3f}")
    logger.info(f"Improvement (signed): {improvement:+.3f}")
    
    # 3-panel plot
    plt.figure(figsize=(15, 4))
    
    # 1: averages
    plt.subplot(1, 3, 1)
    plt.bar(['Base', 'LoRA'], [avg_base, avg_lora])
    plt.ylabel(f'{score_name.upper()} Score')
    plt.title('Average Scores')
    
    # 2: improvement histogram
    if higher_is_better:
        improvements = [l - b for l, b in zip(lora_scores, base_scores)]
    else:
        improvements = [b - l for l, b in zip(lora_scores, base_scores)]  # Invert for perplexity
    plt.subplot(1, 3, 2)
    plt.hist(improvements, bins=30, color='green', alpha=0.7, edgecolor='black')
    plt.axvline(0, color='red', linestyle='--', linewidth=2)
    plt.xlabel('Improvement (LoRA - Base)')
    plt.ylabel('Examples')
    plt.title('Improvement Distribution')
    
    # 3: score distributions
    plt.subplot(1, 3, 3)
    plt.hist(base_scores, bins=30, alpha=0.7, label='Base', color='blue')
    plt.hist(lora_scores, bins=30, alpha=0.7, label='LoRA', color='orange')
    plt.xlabel(f'{score_name.upper()} Score')
    plt.ylabel('Examples')
    plt.legend()
    plt.title('Score Distributions')
    
    plt.tight_layout()
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.show()
    logger.info(f"Plot saved: {plot_file}")


def main():
    config = Config()
    _setup_logger("INFO")
    user_args = _parse_args()

    _ensure_benchmark_dataset(config)
    
    if user_args.plot_only == False:  
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        tokenizer.pad_token = config.fim_pad_token
        tokenizer.padding_side = "right"

        _ensure_directories_exist(config)
        _evaluate_models_to_file(config, user_args, tokenizer)

    _plot_metric_stats_from_file(config, config.sentencebleu_score_name, config.sentencebleu_plot_file, higher_is_better=True)
    _plot_metric_stats_from_file(config, config.codebleu_score_name, config.codebleu_plot_file, higher_is_better=True)
    _plot_metric_stats_from_file(config, config.exact_match_score_name, config.exact_match_plot_file, higher_is_better=True)
    _plot_metric_stats_from_file(config, config.line_match_score_name, config.line_match_plot_file, higher_is_better=True)
    _plot_metric_stats_from_file(config, config.perplexity_name, config.perplexity_plot_file, higher_is_better=False)


if __name__ == "__main__":
    main()