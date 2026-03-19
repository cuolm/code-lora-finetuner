
import logging

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

from .config import Config


logger = logging.getLogger(__name__)


def load_and_configure_lora_model(config: Config) -> AutoModelForCausalLM:
    if config.device == "cuda":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",  # nf4 (NormalFloat4) = optimal for LLM training, because llm weights follwo normal distribution  
            bnb_4bit_compute_type=config.model_dtype,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=config.model_name,
            attn_implementation=config.model_attn_implementation,  
            quantization_config=bnb_config,
            device_map="auto" # let bitsandbytes handle placement
        )
        model = prepare_model_for_kbit_training(model)
    elif config.device == "mps":
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=config.model_name,
            attn_implementation=config.model_attn_implementation,
            dtype=config.model_dtype 
        ).to("mps")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=config.model_name,
            attn_implementation=config.model_attn_implementation,
            dtype=config.model_dtype
        ).to("cpu")
    
    logger.info(
        f"Model: {config.model_name} | Device: {config.device} | "
        f"Dtype: {config.model_dtype} | Attn: {config.model_attn_implementation}"
    )
    
    # forces the input to require gradients, ensuring the backward pass graph stays connected when using frozen base models with gradient checkpointing
    if config.trainer_gradient_checkpointing:
        model.enable_input_require_grads()

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
