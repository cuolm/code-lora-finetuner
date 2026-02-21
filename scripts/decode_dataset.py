import argparse
import json
from pathlib import Path
from transformers import AutoTokenizer

def main():
    parser = argparse.ArgumentParser(description="Decode jsonl dataset lines")
    parser.add_argument("--dataset_path", type=Path, required=True, help="Path to jsonl dataset")
    parser.add_argument("--start_line", type=int, required=True, help="Start line (0-indexed)")
    parser.add_argument("--end_line", type=int, required=True, help="End line (exclusive)")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B", help="Tokenizer model")
    
    args = parser.parse_args()
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    output_path = Path(f"{args.dataset_path.stem}_decoded.jsonl")
    
    with open(args.dataset_path) as f_in, open(output_path, 'w') as f_out:
        for i, line in enumerate(f_in):
            if args.start_line <= i < args.end_line:
                example = json.loads(line)
                input_ids = example['input_ids']
                decoded = tokenizer.decode(input_ids, skip_special_tokens=False)
                example['decoded_text'] = decoded
                f_out.write(json.dumps(decoded, ensure_ascii=False) + '\n')
            if i >= args.end_line:
                break
    
    print(f"Decoded lines {args.start_line}-{args.end_line} → {output_path}")

if __name__ == "__main__":
    main()