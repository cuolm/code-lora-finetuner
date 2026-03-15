import argparse
import gc
import json
import logging
import math

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.trainer_utils import get_last_checkpoint

from .config import Config


logger = logging.getLogger("stc.evaluate.generate")


def _load_model(config: Config, user_args: argparse.Namespace) ->AutoModelForCausalLM:
    if user_args.checkpoint == "last":
        checkpoint_path = get_last_checkpoint(config.trainer_output_dir_path)
    else:
        checkpoint_path = config.trainer_output_dir_path / user_args.checkpoint
    
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

    logger.info(f"Loaded base model and LoRA adapter: {checkpoint_path}")
    return model


def _load_tokenizer(config: Config) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = config.fim_pad_token
    tokenizer.padding_side = "right"
    return tokenizer


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


def generate_and_save(config: Config, user_args: argparse.Namespace):
    model = _load_model(config, user_args)
    tokenizer = _load_tokenizer(config)
    
    try:
        line_counter = 0
        with config.benchmark_dataset_path.open("r") as benchmark_dataset_file, \
            config.benchmark_evaluation_results_path.open("w") as evaluation_results_file:
            
            for line in benchmark_dataset_file:
                example = json.loads(line)
                prompt = (
                    f"{config.fim_prefix_token}{example['prefix']}"
                    f"{config.fim_suffix_token}{example['suffix']}"
                    f"{config.fim_middle_token}"
                )

                reference_middle = example["reference_middle"]
                lora_generated_middle = _generate_code(config, model, tokenizer, prompt)
                lora_perplexity = _get_fim_perplexity(config, model, tokenizer, example["prefix"], example["suffix"], reference_middle)
                with model.disable_adapter():
                    base_generated_middle = _generate_code(config, model, tokenizer, prompt)
                    base_perplexity = _get_fim_perplexity(config, model, tokenizer, example["prefix"], example["suffix"], reference_middle)
                
                result = {
                    "example_id": line_counter,
                    "reference_middle": reference_middle,
                    "base_generated_middle": base_generated_middle,
                    "lora_generated_middle": lora_generated_middle,
                    "base_perplexity": base_perplexity,
                    "lora_perplexity": lora_perplexity
                }
                evaluation_results_file.write(json.dumps(result) + "\n")

                line_counter += 1
                if line_counter % 10 == 0:
                    _clear_hardware_cache(config)
                    logger.info(f"Processed Example {line_counter}")
        
        logger.info(f"Successfully generated and saved {line_counter} number of examples to {config.benchmark_evaluation_results_path}.")

    except Exception:
        logger.exception("Generation failed.")
        raise


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
                max_new_tokens=config.generation_max_new_tokens,
                do_sample=config.generation_do_sample,
                temperature=config.generation_temperature,
                top_p=config.generation_top_p,
                bad_words_ids=bad_words_ids,
                pad_token_id=tokenizer.pad_token_id
            )
    

    # slice the output, take everything after the input_length, model.generate() functin returns the whole example, not only the generated text
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