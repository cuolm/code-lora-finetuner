
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel


from .config import Config


def load_and_configure_lora_model(config: Config) -> AutoModelForCausalLM:
    if config.device == "cuda":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",  # quantization type fp4 or nf4"
            bnb_4bit_compute_type=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=config.model_name,
            #attn_implementation="sdpa", # built-in PyTorch implementation of scaled dot product attention
            quantization_config=bnb_config,
            device_map="auto" # let bitsandbytes handle placement
        )
        model = prepare_model_for_kbit_training(model)
    elif config.device == "mps":
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=config.model_name,
            torch_dtype=torch.float16  # reduces weights from 32-bit to 16-bit float.
        ).to("mps")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=config.model_name,
        ).to("cpu")

    lora_config = LoraConfig(
        lora_alpha=config.lora_alpha, 
        lora_dropout=config.lora_dropout, 
        r=config.lora_r, 
        bias=config.lora_bias, 
        task_type="CAUSAL_LM",
        target_modules=config.lora_target_modules,
    )
    lora_model = get_peft_model(model, lora_config)
    return lora_model