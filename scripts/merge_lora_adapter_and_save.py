import json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

project_root_path = Path(__file__).resolve().parent.parent
print(project_root_path)

lora_adapter_path = project_root_path / "lora_adapter"
with open(lora_adapter_path / "adapter_config.json", "r") as f:
    adapter_config = json.load(f)

base_model_name_or_path = adapter_config["base_model_name_or_path"]

base_model = AutoModelForCausalLM.from_pretrained(base_model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)

peft_model = PeftModel.from_pretrained(base_model, lora_adapter_path) # Load PEFT model (LoRA adapter) on top of base model

merged_model = peft_model.merge_and_unload() # Merge LoRA weights into base model 
merged_model.save_pretrained("./lora_model")
tokenizer.save_pretrained("./lora_model") # Save tokenizer as well

print(f"Merged lora adapter model into base model {base_model_name_or_path} and saved to ./lora_model")
