import json
import logging

from .config import Config
from .metrics import get_codebleu, get_exact_match, get_line_match, get_sentencebleu


logger = logging.getLogger(__name__)


def evaluate_and_save(config: Config) -> None:
    results_path = config.benchmark_evaluation_results_path 
    temp_path = results_path.with_name(f"{results_path.name}.tmp")

    try:
        with results_path.open("r") as evaluation_results_file, \
             temp_path.open("w") as tmp_file:
            
            line_counter = 0
            for line in evaluation_results_file:
                result = json.loads(line)

                ref = result["reference_middle"] 
                base_gen = result["base_generated_middle"]
                lora_gen = result["lora_generated_middle"]

                base_cb, cb_valid = get_codebleu(config, ref, base_gen)
                lora_cb, _ = get_codebleu(config, ref, lora_gen)
                
                base_sb = get_sentencebleu(config, ref, base_gen)
                lora_sb = get_sentencebleu(config, ref, lora_gen)

                base_em = get_exact_match(ref, base_gen)
                lora_em = get_exact_match(ref, lora_gen)

                base_lm = get_line_match(config, ref, base_gen)
                lora_lm = get_line_match(config, ref, lora_gen)

                result.update({
                    "base_codebleu": base_cb,
                    "lora_codebleu": lora_cb,
                    "codebleu_valid": cb_valid,
                    "base_sentencebleu": base_sb,
                    "lora_sentencebleu": lora_sb,
                    "base_exact_match": base_em,
                    "lora_exact_match": lora_em,
                    "base_line_match": base_lm,
                    "lora_line_match": lora_lm,
                })

                tmp_file.write(json.dumps(result) + "\n")
                line_counter += 1

                if line_counter % 10 == 0:
                    logger.info(f"Processed {line_counter} examples...")

        temp_path.replace(results_path)
        logger.info(f"Successfully evaluated and saved {line_counter} examples to {config.benchmark_evaluation_results_path}.")

    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        logger.exception(f"Evaluation failed.")
        raise