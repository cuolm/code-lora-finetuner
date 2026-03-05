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
    comparison_results_path: Path = field(init=False)
    plot_file: Path = field(init=False)
    device: str = field(init=False)

    # CodeBLEU weights (must sum to 1.0)
    # see: https://arxiv.org/pdf/2009.10297  Section 4.4 for parameter suggestions 0.1, 0.1, 0.4, 0.4
    cb_language: str = "c"
    cb_score_name: str = "codebleu"
    cb_ngram_weight: float = 0.1         # token-level overlap (standard BLEU)
    cb_weighted_ngram_weight: float = 0.1 # keyword-level overlap (importance-weighted)
    cb_syntax_ast_weight: float = 0.4     # structural correctness (Abstract Syntax Tree)
    cb_dataflow_weight: float = 0.4       # logic consistency (Variable dependency graph)
    cb_plot_file: Path = field(init=False)

    # sentence-BLEU weights (must sum to 1.0)
    sb_score_name: str = "sentencebleu"
    sb_ngram_weight_1: float = 0.25  # 1-gram
    sb_ngram_weight_2: float = 0.25  # 2-gram  
    sb_ngram_weight_3: float = 0.25  # 3-gram
    sb_ngram_weight_4: float = 0.25  # 4-gram
    sb_plot_file: Path = field(init=False)

    em_score_name: str = "em"
    em_plot_file: Path = field(init=False)

    lm_score_name: str = "lm"
    lm_number_of_lines: int = 2  # 5 is standard production value (Sourcegraph, Cursor IDE)
    lm_plot_file: Path = field(init=False)

    perplexity_name: str = "ppl"
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

        cb_total = (self.cb_ngram_weight + self.cb_weighted_ngram_weight +
            self.cb_syntax_ast_weight + self.cb_dataflow_weight)
        if abs(cb_total - 1.0) > 1e-6:
            raise ValueError(f"CodeBLEU weights must sum to 1.0, got {cb_total}")

        self.project_root_path = Path(__file__).resolve().parent.parent
        self.trainer_output_dir_path = self.project_root_path / "results"
        self.input_dataset_path = self.project_root_path / "datasets" / "test_dataset.jsonl"
        self.benchmark_dataset_path = self.project_root_path / "benchmarks" / "pebble_test_examples.jsonl"
        self.base_results_tmp_path = self.project_root_path / "benchmarks" / "results" / "base_results_tmp.jsonl"
        self.comparison_results_path = self.project_root_path / "benchmarks" / "results" / "comparison_results.jsonl"
        self.cb_plot_file = self.project_root_path / "benchmarks" / "results" / "codebleu_plot.png"
        self.sb_plot_file = self.project_root_path / "benchmarks" / "results" / "sentencebleu_plot.png"
        self.em_plot_file = self.project_root_path / "benchmarks" / "results" / "em_plot.png"
        self.lm_plot_file = self.project_root_path / "benchmarks" / "results" / "lm_plot.png"
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
        config.comparison_results_path.parent,
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
                               lang=config.cb_language, 
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
    CodeBLEU = (cb_ngram_weight * ngram_score) + 
               (cb_weighted_ngram_weight * weighted_ngram_score) + 
               (cb_syntax_ast_weight * syntax_ast_score) + 
               (cb_dataflow_weight * dataflow_score)
    """
    if not _codebleu_structure_valid(config, reference):
        return (0.0, False)
        
    try:
        codebleu_algorithm_weights = (
            config.cb_ngram_weight, 
            config.cb_weighted_ngram_weight, 
            config.cb_syntax_ast_weight, 
            config.cb_dataflow_weight
        )
        
        result = codebleu_score(
            [reference], [prediction], 
            lang=config.cb_language, 
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
            config.sb_ngram_weight_1, 
            config.sb_ngram_weight_2, 
            config.sb_ngram_weight_3, 
            config.sb_ngram_weight_4
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
    """EM: 1.0 if identical, 0.0 otherwise. Collapese all whitespaces."""
    try:
        # re.sub(r'\s+', ' ', text.strip()): Collapses whitespace to compare logic regardless of formatting.
        ref_norm = re.sub(r'\s+', ' ', reference.strip())
        pred_norm = re.sub(r'\s+', ' ', prediction.strip())
        
        if ref_norm == pred_norm:
            return 1.0
        else:
            return 0.0
    except Exception as e:
        logger.exception(f"ERROR in EM calculation: {e}")
        return 0.0


def _get_line_match(config: Config, reference: str, prediction: str) -> float:
    """Check if the first n lines match, ignoring trailing whitespace."""
    try:
        n = config.lm_number_of_lines

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
    Lower perplexity indicates higher confidence. PPL = exp(loss).
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
        logger.exception(f"ERROR in PPL calculation: {e}")
        return float('inf')


def _generate_from_base_model_to_file(config: Config, tokenizer: AutoTokenizer) -> None:
    logger.info("--- Loading Base Model ---")

    if config.device == "cuda":
        base_model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=config.model_name,
            dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=config.model_name,
            dtype=torch.float16,
            low_cpu_mem_usage=True
        ).to(config.device)

    logger.info("--- Generating Base Model Responses ---")
    with open(config.benchmark_dataset_path, "r") as benchmark_dataset_file, \
         open(config.base_results_tmp_path, "w") as base_results_tmp_file:
        for i, line in enumerate(benchmark_dataset_file):
            example = json.loads(line)
            prompt = (
                f"{config.fim_prefix_token}{example['prefix']}"
                f"{config.fim_suffix_token}{example['suffix']}"
                f"{config.fim_middle_token}"
            )
            generated_middle = _generate_code(config, base_model, tokenizer, prompt)

            base_ppl = _get_fim_perplexity(config, base_model, tokenizer, example["prefix"], example["suffix"], example["reference_middle"])

            results = {
                "example_id": i,
                "base_generated_middle": generated_middle,
                "reference_middle": example["reference_middle"],
                "base_ppl": base_ppl
            }
            json.dump(results, base_results_tmp_file)
            base_results_tmp_file.write("\n")
            if i % 10 == 0:
                _clear_hardware_cache(config)
                logger.info(f"Processed: {i}")

    del base_model
    _clear_hardware_cache(config)


def _generate_from_lora_model_to_file(config: Config, user_args: argparse.Namespace, tokenizer: AutoTokenizer) -> None:
    if user_args.checkpoint == "last":
        checkpoint_path = get_last_checkpoint(config.trainer_output_dir_path)
    else:
        checkpoint_path = config.trainer_output_dir_path / user_args.checkpoint

    logger.info(f"--- Loading LoRA Checkpoint: {str(checkpoint_path)} ---")
    if config.device == "cuda":
        base_model_for_lora = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=config.model_name,
            dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
    else:
        base_model_for_lora = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=config.model_name,
            dtype=torch.float16,
            low_cpu_mem_usage=True
        ).to(config.device)

    lora_model = PeftModel.from_pretrained(
        model=base_model_for_lora,
        model_id=checkpoint_path,
    )
    logger.info("--- Generating LoRA Model Responses and Comparing ---")
    with open(config.benchmark_dataset_path, "r") as benchmark_dataset_file, \
         open(config.base_results_tmp_path, "r") as base_results_tmp_file, \
         open(config.comparison_results_path, "w") as comparison_results_file:

        for i, (benchmark_example_line, base_results_line) in enumerate(zip(benchmark_dataset_file, base_results_tmp_file)):
            benchmark_example = json.loads(benchmark_example_line)
            base_results = json.loads(base_results_line)

            assert base_results["example_id"] == i  # sanity check

            prompt = (
                f"{config.fim_prefix_token}{benchmark_example['prefix']}"
                f"{config.fim_suffix_token}{benchmark_example['suffix']}"
                f"{config.fim_middle_token}"
            )
            lora_generated_middle = _generate_code(config, lora_model, tokenizer, prompt)

            full_example = (
                f"{config.fim_prefix_token}{benchmark_example['prefix']}"
                f"{config.fim_suffix_token}{benchmark_example['suffix']}"
                f"{config.fim_middle_token}{benchmark_example['reference_middle']}"
            )
            
            base_codebleu, codebleu_valid = _get_codebleu(config, benchmark_example["reference_middle"], base_results["base_generated_middle"])
            lora_codebleu, _ = _get_codebleu(config, benchmark_example["reference_middle"], lora_generated_middle)

            base_em = _get_exact_match(config, benchmark_example["reference_middle"], base_results["base_generated_middle"])
            lora_em = _get_exact_match(config, benchmark_example["reference_middle"], lora_generated_middle)

            base_sentencebleu = _get_sentencebleu(config, benchmark_example["reference_middle"], base_results["base_generated_middle"])
            lora_sentencebleu = _get_sentencebleu(config, benchmark_example["reference_middle"], lora_generated_middle)

            base_lm = _get_line_match(config, benchmark_example["reference_middle"], base_results["base_generated_middle"])
            lora_lm = _get_line_match(config, benchmark_example["reference_middle"], lora_generated_middle)

            lora_ppl = _get_fim_perplexity(config, lora_model, tokenizer, benchmark_example["prefix"], benchmark_example["suffix"], benchmark_example["reference_middle"])

            result = {
                "example_id": i,
                "full_example": full_example,
                "reference_middle": benchmark_example["reference_middle"],
                "base_generated_middle": base_results["base_generated_middle"],
                "lora_generated_middle": lora_generated_middle,
                "base_codebleu": base_codebleu,
                "lora_codebleu": lora_codebleu,
                "codebleu_valid": codebleu_valid,
                "base_sentencebleu": base_sentencebleu,
                "lora_sentencebleu": lora_sentencebleu,
                "base_em": base_em,
                "lora_em": lora_em,
                "base_lm": base_lm,
                "lora_lm": lora_lm,
                "base_ppl": base_results["base_ppl"],
                "lora_ppl": lora_ppl
            }

            # Write each result as one JSON line.
            json.dump(result, comparison_results_file)
            comparison_results_file.write("\n")
            if i % 10 == 0: 
                _clear_hardware_cache(config)
                logger.info(f"Processed: {i}")

    del lora_model
    _clear_hardware_cache(config)


def _plot_metric_stats_from_file(config: Config, score_name: str, plot_file: Path, higher_is_better: bool) -> None:
    base_scores = []
    lora_scores = []
    
    with open(config.comparison_results_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            if (score_name == config.cb_score_name) and data["codebleu_valid"] is False:
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
        improvements = [b - l for l, b in zip(lora_scores, base_scores)]  # Invert for PPL
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
        _generate_from_base_model_to_file(config, tokenizer)
        _generate_from_lora_model_to_file(config, user_args, tokenizer)

    _plot_metric_stats_from_file(config, config.sb_score_name, config.sb_plot_file, higher_is_better=True)
    _plot_metric_stats_from_file(config, config.cb_score_name, config.cb_plot_file, higher_is_better=True)
    _plot_metric_stats_from_file(config, config.em_score_name, config.em_plot_file, higher_is_better=True)
    _plot_metric_stats_from_file(config, config.lm_score_name, config.lm_plot_file, higher_is_better=True)
    _plot_metric_stats_from_file(config, config.perplexity_name, config.perplexity_plot_file, higher_is_better=False)


if __name__ == "__main__":
    main()