import torch
import textwrap
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
from peft import PeftModel

def check_device() -> str:
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    return device

def generate_from(model_name: str, prompt: str) -> str:
    device = check_device()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16).to(device)

    model.eval()
    model_inputs = tokenizer(prompt, return_tensors="pt")
    model_inputs = {k: v.to(device) for k, v in model_inputs.items()}  # Fixes RuntimeError: Placeholder storage has not been allocated on MPS device!

    model_outputs = model.generate(
        **model_inputs,
        max_new_tokens=200,  # or your preferred max length
    )

    generated_model_text = tokenizer.decode(model_outputs[0], skip_special_tokens=True)
    return generated_model_text

def generate_from_lora_augmented(base_model_path: str, lora_adapter_path: str, prompt: str) -> str:
    device = check_device()
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    base_model = AutoModelForCausalLM.from_pretrained(base_model_path, dtype=torch.float16).to(device)

    base_model_lora_augmented = PeftModel.from_pretrained(base_model, lora_adapter_path).to(device)
    base_model_lora_augmented.eval()

    model_inputs = tokenizer(prompt, return_tensors="pt")
    model_inputs = {k: v.to(device) for k, v in model_inputs.items()}  # Fixes RuntimeError: Placeholder storage has not been allocated on MPS device!
    model_outputs = base_model_lora_augmented.generate(
        **model_inputs,
        max_new_tokens=200,  # or your preferred max length
    )

    generated_model_text = tokenizer.decode(model_outputs[0], skip_special_tokens=True)
    return generated_model_text

def main():
    project_root_path = Path(__file__).resolve().parent.parent

    prefix_c = textwrap.dedent("""\
    static inline int 
    """)
    suffix_c = textwrap.dedent("""\
    """)

    prefix_prompt = prefix_c 

    suffix_promt = suffix_c
    prompt = f"""<|fim_prefix|>{prefix_prompt}<|fim_suffix|>{suffix_promt}<|fim_middle|>"""

    base_model_path = "Qwen/Qwen2.5-Coder-1.5B"
    lora_model_path = f"{project_root_path}/lora_model"
    lora_adapter_path = f"{project_root_path}/lora_adapter"

    base_model_generated = generate_from(base_model_path, prompt)
    lora_model_generated = generate_from(lora_model_path, prompt)

    base_model_lora_augmented = generate_from_lora_augmented(base_model_path, lora_adapter_path, prompt)

    print("==================== BASE MODEL ====================")
    print(base_model_generated)
    print("\n")

    print("=================== BASE MODEL + LoRA ADAPTER AUGMENTED ===================")
    print(base_model_lora_augmented)
    print("\n") 

    print("======================= MERGED LORA MODEL =========================")
    print(lora_model_generated)
    print("\n") 

if __name__ == "__main__":
    main()
