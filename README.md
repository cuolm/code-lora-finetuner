# CodeFinetuner
[![Build](https://github.com/cuolm/codefinetuner/actions/workflows/tests.yml/badge.svg)](https://github.com/cuolm/codefinetuner/actions)
[![PyPI](https://img.shields.io/pypi/v/codefinetuner.svg)](https://pypi.org/project/codefinetuner/)
[![License](https://img.shields.io/github/license/cuolm/codefinetuner.svg)](LICENSE.txt)
[![Coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/cuolm/gist_id/raw/coverage.json)](https://github.com/cuolm/codefinetuner/actions)


Create your own local code autocomplete model, fine-tuned on your custom code repository, for use in editors like VS Code or Vim/Neovim.

Fine-tuning is achieved by training a Low-Rank Adapter ([LoRA](https://arxiv.org/abs/2106.09685)) to perform Fill-In-the-Middle ([FIM](https://arxiv.org/abs/2207.14255)) completion. 

## Table of Contents
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [How Training Examples Are Created](#how-training-examples-are-created)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Usage](#usage)
- [Deployment](#deployment)
- [Docker](#docker)
- [Tree-sitter Setup](#tree-sitter-setup) <!-- Link to new docs -->
- [Tests](#tests)
- [Resources](#resources)
- [License](#license)


## Architecture
```text
Raw Code Files
     |
     v
[Preprocess]  -- tree-sitter parsing -> FIM examples -> tokenized JSONL
     |
     v
[Finetune]    -- LoRA adapter training -> merged model
     |
     v
[Evaluate]    -- CodeBLEU, SentenceBLEU, exact match, line match, perplexity
     |
     v
[Export]      -- GGUF conversion -> quantized model for deployment
```

## Project Structure
```text
.
├── src/
│   └── codefinetuner/           # Core packages
│       ├── preprocess/
│       ├── finetune/
│       ├── evaluate/
│       └── export/              # Handles LoRA merging & GGUF conversion
├── config/                      # User configuration
│   └── codefinetuner_config.yaml
├── data/                        # Default data directory (Workspace root)
├── outputs/                     # Pipeline artifacts (Workspace root)
├── scripts/                     # Utility scripts
├── tests/                       # Unit tests 
├── third_party/                 # External submodules (e.g., custom parsers)
└── docs/                        # Documentation and assets
```

## How Training Examples Are Created
To generate high-quality FIM examples, high-level structural code blocks are extracted (e.g., functions, classes). From these blocks, logical sub-blocks (e.g., statements, expressions) are masked to serve as the "middle" section for the model to predict.

Here is an example illustrating how a single FIM example is created:
<table>
  <tr>
    <td align="center" valign="top">
      <strong>Source Code File</strong><br>
      <img src="docs/code_file.png" alt="code_file" width="250">
    </td>
    <td align="center" valign="top">
      <strong>Code Block</strong><br>
      <img src="docs/code_block.png" alt="code_block" width="250">
    </td>
    <td align="center" valign="top">
      <strong>One Subblock</strong><br>
      <img src="docs/code_subblock.png" alt="code_subblock" width="250">
    </td>
  </tr>
</table>

```python
<|fim_prefix|>uint32_t count_bits(uint32_t value){\n  uint32_t count = 0;\n  while(value){\n    
<|fim_suffix|>    }\n    return count;
<|fim_middle|>count = count + (value & 1);\n    value = (value >> 1);
```

Using this technique, rather than randomly splitting code into unrelated text chunks, helps the model learn the logical patterns and structure of your specific codebase.

## Installation

### From PyPI

**Core Pipeline**
Installs preprocessing, fine-tuning, and evaluation modules.
```bash
uv add codefinetuner
# or
pip install codefinetuner
```

**Full Pipeline (Recommended)**
Includes dependencies for GGUF model export and quantization.
```bash
uv add "codefinetuner[export]"
# or
pip install "codefinetuner[export]"
```

### From Source (Development)
```bash
git clone --recurse-submodules https://github.com/cuolm/codefinetuner
cd codefinetuner

# Using uv (Recommended)
uv sync

# Using pip
pip install -r requirements.txt
pip install -e .
```

---

## Quick Start
```python
import codefinetuner

# Run the complete pipeline
codefinetuner.run_pipeline("config.yaml")
```

## Configuration

The pipeline uses a single-source-of-truth YAML configuration file. It utilizes YAML anchors (`&globals`) to share core parameters across all stages (`preprocess`, `finetune`, `evaluate`), ensuring consistency and reducing redundancy.

### Configuration Structure

Create a codefinetuner_config.yaml using the template below. For a full list of all available parameters and their effects, see the [Configuration Reference Guide](/docs/config-file.md).

```yaml
# globals contain all the mandatory parameters.
globals: &globals
  workspace_path: null  # null: defaults to current working directory (CWD)
  model_name: "Qwen/Qwen2.5-Coder-1.5B" 
  fim_prefix_token: "<|fim_prefix|>"
  fim_middle_token: "<|fim_middle|>"
  fim_suffix_token: "<|fim_suffix|>"
  fim_pad_token: "<|fim_pad|>"
  eos_token: "<|endoftext|>"
  label_pad_token_id: -100
  data_language: "c"
  data_extensions: [".c", ".h"]

preprocess:
  <<: *globals                   # Inherits all global parameters
  split_mode: "manual"
  max_token_sequence_length: 1024
  # ... (preprocess specific settings)

finetune:
  <<: *globals
  lora_r: 32
  trainer_num_train_epochs: 1
  # ... (finetune specific settings)

evaluate:
  <<: *globals
  benchmark_sample_size: 4
  # ... (evaluate specific settings)

```
> **Note:** For a complete, production-ready example, see [`config/codefinetuner_config.yaml`](/config/codefinetuner_config.yaml).

### Data Preparation
Place source files in your `raw_data_path` (default: `workspace_path/data`).
* **Auto Split:** Place files directly in the directory.
* **Manual Split:** Create `train`, `eval`, and `test` subfolders inside `raw_data_path` and assign files according to your manual split preferences.

## Usage

### CLI Usage
Run the pipeline using the unified CLI:
```bash
uv run codefinetuner --config="config/codefinetuner_config.yaml"
```

**Pipeline Flags:**
* `--config`: Specify path to a different config file.
* `--skip-preprocess`, `--skip-finetune`, `--skip-evaluate`, `--skip-export`: Skip specific stages.

### Python Module Usage
```python
import codefinetuner

# Full pipeline
codefinetuner.run_pipeline("path/to/codefinetuner_config.yaml")

# Skip stages
codefinetuner.run_pipeline(
    "path/to/codefinetuner_config.yaml",
    skip_preprocess=True,
    skip_export=True
)

# Individual stages
from codefinetuner import preprocess, finetune
preprocess.run("config.yaml")
```

## Deployment: Using the Model
The `export` stage converts the model to GGUF format. The final GGUF file is located under `outputs/export/results/lora_model.gguf`.  
For a detailed guide on how to use the gguf model with the VS Code extension [llama.vscode](https://github.com/ggml-org/llama-vscode), check out the [inference-vscode](/docs/inference-vscode.md) guide.


## Create And Run Docker Image

#### 1. Build the Docker Image
Build the image from the `Dockerfile`, tagging it as codefinetuner-image.
```bash
docker build -t codefinetuner-image .
```
#### 2. Prepare Data and Run the Container
To allow the container to access your data for fine-tuning, use a bind mount to link your host machine's `data` directory to the container.
- On your host machine (where you run Docker), create a folder named `data` if it doesn't already exist.
- Put all files you want to use for fine-tuning inside the `data` directory. For `manual` mode, include `train`, `eval`, and `test` subdirectories containing your manually splitted files.
- Start the container with the bind mount, and open a Bash shell depending on your host machine hardware:

#### NVIDIA GPU (Recommended)
Use this command to enable CUDA support for `torch` and `bitsandbytes`. Requires the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed on the host machine.

```bash
docker run --gpus all -it --rm \
  -v $(pwd)/data:/app/data \
  codefinetuner-image /bin/bash
```
#### CPU Only

Use this command if no compatible GPU is available. Note that fine-tuning will be significantly slower.
```bash
docker run -it --rm \
  -v $(pwd)/data:/app/data \
  codefinetuner-image /bin/bash
```

## Tree-sitter Customization
Tree-sitter parses code into structural blocks for generating FIM training examples. Customize for new languages or build missing parsers.

- [Add Language Definitions](docs/tree-sitter-customization.md#add-new-language-block-definitions) - Define `block_types`/`subblock_types` in JSON.
- [Build Custom Parser](docs/tree-sitter-customization.md#build-custom-parser) - Compile from source (e.g., Mojo).


## Tests
```bash
pytest
```

## Useful Resources
- [Qwen2.5-Coder Technical Report](https://arxiv.org/pdf/2409.12186)
- [Structure-Aware Fill-in-the-Middle Pretraining for Code](https://arxiv.org/pdf/2506.00204)
- [LORA: LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS](https://arxiv.org/pdf/2106.09685)
- [Efficient Training of Language Models to Fill in the Middle](https://arxiv.org/pdf/2207.14255)
- [From Output to Evaluation: Does Raw Instruction-Tuned Code LLMs Output Suffice for Fill-in-the-Middle Code Generation?](https://arxiv.org/pdf/2505.18789)
- [CodeBLEU: a Method for Automatic Evaluation of Code Synthesis](https://arxiv.org/pdf/2009.10297)
- [HF LLM Course](https://huggingface.co/learn/llm-course/chapter1/1)
- [llama.vscode](https://github.com/ggml-org/llama.vscode)

## License
Licensed under the [Apache License 2.0](LICENSE.txt).