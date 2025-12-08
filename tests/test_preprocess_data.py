import sys
import pytest
import numpy as np
import torch
from pathlib import Path

root_path = Path(__file__).parent.parent.absolute()
src_path = str(root_path / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import preprocess_data

@pytest.fixture(scope="module")  # Module scope creates fixture only oncc, shares it for all tests in current file
def config_fixture():
    config_fixture = preprocess_data.Config()
    config_fixture.source_files_language = "c"
    config_fixture.extensions = [".c"]
    config_fixture.project_root_path = Path(__file__).parent.parent
    config_fixture.raw_data_path = Path(__file__).parent / "test_data"
    return config_fixture

@pytest.fixture
def c_file_path_fixture():
    return Path(__file__).parent / "test_data" / "test.c"

@pytest.fixture(scope="module") 
def qwen_tokenizer_fixture():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-0.5B")
    tokenizer.pad_token = "<|fim_pad|>"
    return tokenizer

def test_extract_code_blocks(config_fixture, c_file_path_fixture):
    blocks = list(preprocess_data._get_code_blocks_from_paths(config_fixture, [c_file_path_fixture]))
    
    assert len(blocks) == 2
    assert all(node.type == "function_definition" for _, node in blocks)
    assert b"count_bits" in blocks[0][0]
    assert b"add" in blocks[1][0]

def test_create_fim_examples(config_fixture, c_file_path_fixture):
    blocks = list(preprocess_data._get_code_blocks_from_paths(config_fixture, [c_file_path_fixture]))
    fim_examples = preprocess_data.create_fim_examples(config_fixture, blocks)
    
    assert len(fim_examples) >= 2  # At least one per function
    for example in fim_examples:
        assert b"<|fim_prefix|>" in example
        assert b"<|fim_suffix|>" in example
        assert b"<|fim_middle|>" in example

def test_tokenize_examples(config_fixture, qwen_tokenizer_fixture, c_file_path_fixture):
    blocks = list(preprocess_data._get_code_blocks_from_paths(config_fixture, [c_file_path_fixture]))
    fim_examples = preprocess_data.create_fim_examples(config_fixture, blocks)
    tokenized_fim_examples = preprocess_data.tokenize_examples(config_fixture, fim_examples, qwen_tokenizer_fixture)
    
    assert "labels" in tokenized_fim_examples
    assert tokenized_fim_examples["input_ids"].shape == tokenized_fim_examples["labels"].shape

    print(tokenized_fim_examples["labels"])
    for labels in tokenized_fim_examples["labels"]:  
        assert torch.sum(labels != -100) > 1  # Make sure that there is at least a non -100 token in labels
