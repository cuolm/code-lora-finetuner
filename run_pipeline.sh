#!/bin/bash
set -e

source .venv/bin/activate
echo "Activated virtual environment."

echo "Ensure the base model matches in all 3 scripts."
echo "(preprocess_data.py, finetune_model.py, evaluate_model.py)"
read -p "Confirm with [Enter]..."

echo "Enter your project settings:"
read -p "File extensions (ex: .c .h): " EXTENSIONS
read -p "Source language (ex: c): " LANGUAGE
read -p "Data folder path (ex: ./data): " DATA_PATH

echo "Choose split mode:"
echo "1) Auto split"
echo "2) Manual split (create train/, eval/, test/ folders INSIDE $DATA_PATH first)"
read -p "Enter 1 or 2: " choice

if [ "$choice" = "1" ]; then
  SPLIT_MODE="auto"
else
  SPLIT_MODE="manual"
fi

echo "Preprocessing code files ($SPLIT_MODE)..."
python src/preprocess_data.py \
  --extensions $EXTENSIONS \
  --source-files-language $LANGUAGE \
  --split-mode $SPLIT_MODE \
  --raw-data-path "$DATA_PATH"

echo "Training model..."
python src/finetune_model.py

echo "Evaluating model..."
python src/evaluate_model.py

echo "Converting to GGUF..."
python ../llama.cpp/convert.py ./lora_model --outfile ./lora_model_gguf --outtype f16

echo "Pipeline complete!"
