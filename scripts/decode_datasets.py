import json
import pprint
from pathlib import Path
from transformers import AutoTokenizer

pp = pprint.PrettyPrinter(indent=4)

project_root_path = Path(__file__).resolve().parent.parent

file_path = f"{project_root_path}/datasets/train_dataset.jsonl"
data = []

with open(file_path, 'r', encoding='utf-8') as f:
    for line in f:
        json_obj = json.loads(line)
        data.append(json_obj)

tokenizer = AutoTokenizer.from_pretrained(f"{project_root_path}/lora_model")

decoded_data = []
for item in data:
    decoded_input_ids = tokenizer.decode(item["input_ids"], skip_special_tokens=True)
    labels_without_ignore = [token for token in item["labels"] if token != -100] # Filter out any -100 tokens, they are ignored 
    decoded_labels = tokenizer.decode(labels_without_ignore, skip_special_tokens=True)
    decoded_data.append({"decoded_input_ids": decoded_input_ids, "decoded_labels": decoded_labels})

for i in range(0, 2):
    print("==================================================================")
    print(f"--- EXAMPLE {i} SATART ---")
    print("==================================================================")
    pp.pprint(decoded_data[i]["decoded_input_ids"])
    print("\n")
    print("===========================LABELS=================================")
    pp.pprint(decoded_data[i]["decoded_labels"])
    print("\n")
        