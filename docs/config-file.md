# Configuration Reference

This document defines all mandatory and optional parameters in the YAML cofig file. The pipeline uses YAML anchors (`&globals`) and aliases (`<<: *globals`) to propagate shared settings across the `preprocess`, `finetune`, and `evaluate` stages. See [`config/codefinetuner_config.yaml`](/config/codefinetuner_config.yaml) for an example file.

## Global Parameters (Mandatory)
*Shared across all stages.* 
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `workspace_path` | str/null | `null` | Workspace of the project, `null` defaults to CWD.|
| `model_name` | str | `"Qwen/Qwen2.5-Coder-7B"` | HuggingFace model repository ID. |
| `fim_prefix_token` | str | `"<\|fim_prefix\|>"` | FIM prefix special token. |
| `fim_middle_token` | str | `"<\|fim_middle\|>"` | FIM middle special token. |
| `fim_suffix_token` | str | `"<\|fim_suffix\|>"` | FIM suffix special token. |
| `fim_pad_token` | str | `"<\|fim_pad\|>"` | FIM padding token. |
| `eos_token` | str | `"<\|endoftext\|>"` | End-of-sequence token. |
| `label_pad_token_id` | int | `-100` | Token ID ignored in loss calculation. |
| `data_language` | str | `"c"` | Tree-sitter language identifier. |
| `data_extensions` | list | `[".c", ".h"]` | File extensions to include in data preprocessing. |

## Preprocess Parameters (Optional)
*Controls how raw code is converted into FIM training examples.*

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `split_mode` | str | `"auto"` | Dataset splitting strategy: `"auto"` or `"manual"`. For manual split create `train`, `eval`, and `test` subfolders inside `raw_data_path`. |
| `train_ratio` | float | `0.8` | Training set ratio (used in `auto` split). |
| `eval_ratio` | float | `0.1` | Validation set ratio (used in `auto` split). |
| `test_ratio` | float | `0.1` | Test set ratio (used in `auto` split). |
| `max_token_sequence_length` | int | `1024` | Maximum tokens per training example. |
| `max_code_blocks_ast_depth` | int | `2` | Tree Sitter AST depth limit for block extraction. Depth 1 is root, 2 includes child nodes (e.g. functions) |
| `min_middle_tokens_length` | int | `20` | Minimum tokens required in the FIM "middle" section of an example. |
| `max_middle_tokens_length` | int | `200` | Maximum tokens allowed in the FIM "middle" section of an example. |
| `fim_examples_per_subblock_ratio` | float | `1.0` | Number of FIM examples generated per subblock.  1.0 = all fim examples of a subblock are extracted, 0.5 = onls 50% of fim examples of a subblock are extracted |
| `tokenizer_batch_size` | int | `32` | Batch size for the tokenizer. The number of examples processed simultaneously by the tokenizer to improve throughput. |
| `raw_data_path` | str/null | `"data"` | Location of source code files used to generate the datasets. `null` defaults to `<workspace>/data`. |
| `tree_sitter_parser_path` | str/null | `null` | Path to custom `.so`/`.dylib` parser file. |
| `tree_sitter_definitions_path` | str/null | `null` | Path to custom language block definitions JSON. |
| `rng_seed` | int | `0` | Random seed for data shuffling and splitting. |

## Finetune Parameters (Optional)
*Configures the LoRA training process and HuggingFace Trainer.*

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_attn_implementation` | str | `"sdpa"` | Attention backend. `flash_attention_2` is fastest (requires Ampere+ GPUs); `sdpa` is the efficient PyTorch default. |
| `lora_r` | int | `32` | Rank of adapter matrices. Controls parameter count and the complexity of learned patterns. |
| `lora_alpha` | int | `64` | Scaling factor for LoRA updates. Usually set to $2 \times$ `lora_r` to maintain numerical stability. |
| `lora_dropout` | float | `0.1` | Dropout probability for LoRA layers to prevent overfitting on specific code snippets. |
| `lora_bias` | str | `"none"` | Specifies if bias parameters are trained (`"none"`, `"all"`, `"lora_only"`). |
| `lora_target_modules` | list | `["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "down_proj", "up_proj"]` | Model layers targeted for adaptation. Increasing the list improves performance but consumes more VRAM. |
| `trainer_resume_from_checkpoint` | str/null| `null` | Path to a specific checkpoint or `"last"` to continue a previous run. |
| `trainer_clear_checkpoint_dir` | bool | `false` | If `true`, deletes the output folder before starting a new training run. |
| `trainer_num_train_epochs` | int | `1` | Total passes through the training dataset. |
| `trainer_per_device_train_batch_size` | int | `2` | Number of examples processed per GPU at once. Limited by physical VRAM. |
| `trainer_per_device_eval_batch_size` | int | `2` | Batch size used during the evaluation phase of training. |
| `trainer_gradient_accumulation_steps` | int | `32` | Steps to sum gradients before a weight update. Simulates larger batch sizes without increasing memory usage. |
| `trainer_learning_rate` | float | `2e-5` | Initial step size for weight updates. High values can cause divergence; low values result in slow learning. |
| `trainer_weight_decay` | float | `0.1` | L2 regularization coefficient that penalizes large weights to force the model to learn general patterns. |
| `trainer_max_grad_norm` | float | `1.0` | Maximum gradient norm for clipping to prevent exploding gradients. |
| `trainer_lr_scheduler_type` | str | `"cosine"` | Strategy for decaying the learning rate. `cosine` is standard for smooth convergence. |
| `trainer_warmup_steps` | int | `50` | Initial steps where the learning rate ramps up from zero to stabilize early training. |
| `trainer_gradient_checkpointing` | bool | `true` | Memory-saving feature that recomputes activations during the backward pass. Essential for large models. |
| `trainer_logging_steps` | int | `10` | Frequency (in steps) for reporting training metrics to the console/logs. |
| `trainer_eval_strategy` | str | `"steps"` | Trigger for evaluation (`"steps"`, `"epoch"`, or `"no"`). |
| `trainer_eval_steps` | int | `100` | Interval of training steps between model evaluations. Only active if `trainer_eval_strategy` is set to `"steps"`|
| `trainer_save_strategy` | str | `"steps"` | Trigger for saving model checkpoints (`"steps"`, `"epoch"`, or `"no"`). |
| `trainer_save_steps` | int | `100` | Interval of training steps between model checkpoint saves. Only active if `trainer_save_strategy` is set to `"steps"` |
| `trainer_logging_strategy` | str | `"steps"` | Trigger for logging metrics (`"steps"`, `"epoch"`, or `"no"`). |
| `dataset_shuffle_buffer_size` | int | `50000` | Number of training examples to load into memory for randomly shuffling datasets since steramable datasets are used. |
| `dataset_shuffle_seed` | int | `0` | Seed for shuffling datasets. |

## Evaluate Parameters (Optional)
*Configures the benchmark generation and metrics calculation.*

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `benchmark_sample_size` | int | `4` | Total number of FIM examples extracted to form the evaluation suite. |
| `benchmark_min_fim_middle_tokens` | int | `0` | Examples with fewer tokens than this in the middle section are ignored for benchmark dataset creation. |
| `benchmark_shuffle_buffer_size` | int | `10000000` | Size of the shuffle buffer used to shuffle benchmark examples. |
| `benchmark_shuffle_seed` | int | `42` | Random seed ensuring that dataset shuffling is deterministic and reproducible. |
| `benchmark_use_existing_dataset` | bool | `false` | If `true`, the pipeline reuses a previously generated benchmark dataset file instead of creating a new one. |
| `generation_max_new_tokens` | int | `128` | Upper limit on the number of tokens the models are permitted to generate for each code completion. |
| `generation_do_sample` | bool | `false` | Enables probabilistic sampling. If `false`, the model uses greedy decoding (picking only the top token) and `generation_temperature` and `generation_top_p parameters` are ignored.|
| `generation_temperature` | float | `0.7` | Probability smoothing factor. Values < 1.0 make the model more confident; > 1.0 make it more random. Only active when `generation_do_sample` is set to `true`.|
| `generation_top_p` | float | `0.95` | Cumulative probability threshold for nucleus sampling. Limits choices to the most likely tokens totaling 95% probability. Only active when `generation_do_sample` is set to `true`.|
| `codebleu_language` | str | `"c"` | Forces the CodeBLEU parser to use a specific language's grammar rules. |
| `codebleu_ngram_weight` | float | `0.25` | Standard token match. Measures how many exact words or symbols match the ground truth, treating every character (like `;` or `sum`) with equal importance. |
| `codebleu_weighted_ngram_weight` | float | `0.25` | Keyword-based scoring. Gives higher points for correctly predicting programming keywords (like `if`, `while`, `return`, `int`) than for standard symbols, ensuring the score reflects the model's grasp of the code's logic. |
| `codebleu_syntax_ast_weight` | float | `0.25` | Structural correctness. Uses an Abstract Syntax Tree (AST) to check if the code "shape" is correct, even if variable names differ. Ensures the code is grammatically valid for the target language. |
| `codebleu_dataflow_weight` | float | `0.25` | Logic consistency. Tracks how variables are defined and used throughout the code. Checks if the model understands the relationship between inputs and outputs, not just the text. |
| `sentencebleu_ngram_weight_1` | float | `0.25` | Individual word accuracy. Weight of single tokens (1-grams) in the final BLEU score calculation. |
| `sentencebleu_ngram_weight_2` | float | `0.25` | Short phrase accuracy. Weight of 2-token pairs (2-grams) in the final BLEU score calculation. |
| `sentencebleu_ngram_weight_3` | float | `0.25` | Medium phrase accuracy. Weight of 3-token chains (3-grams) in the final BLEU score calculation. |
| `sentencebleu_ngram_weight_4` | float | `0.25` | Long phrase accuracy. Weight of 4-token blocks (4-grams) in the final BLEU score calculation. |
| `line_match_number_of_lines` | int | `2` | Number of identical consecutive lines required between prediction and ground truth to score a line match. |
| `trainer_checkpoint` | str | `"last"` | Specifies the checkpoint folder to load. Use `last` for the latest or provide a specific directory name. |
| `plot_only` | bool | `false` | If `true`, skips the heavy inference and scoring steps to only generate charts from existing results. |