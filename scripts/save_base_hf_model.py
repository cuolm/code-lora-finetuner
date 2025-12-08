import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

project_root_path = Path(__file__).resolve().parent.parent

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

model_name = "Qwen/Qwen2.5-Coder-1.5B"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=model_name,
    dtype=torch.float16  # Reduces weights from 32-bit float to 16-bit float
    ).to(device)

model.save_pretrained(f"{project_root_path}/base_models/{model_name}")
tokenizer.save_pretrained(f"{project_root_path}/base_models/{model_name}")

