import sys
import pytest
import json
import torch
from pathlib import Path
from unittest.mock import patch, MagicMock

root_path = Path(__file__).parent.parent.absolute()
src_path = str(root_path / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import finetune_model

@pytest.fixture(scope="module")
def config_fixture():
    test_data_dir = Path(__file__).parent / "test_data"
    
    config = finetune_model.Config()
    config.project_root_path = test_data_dir.parent
    config.train_dataset_path = test_data_dir / "train_dataset.jsonl"
    config.eval_dataset_path = test_data_dir / "eval_dataset.jsonl"
    config.test_dataset_path = test_data_dir / "test_dataset.jsonl"
    config.trainer_output_dir_path = test_data_dir / "results"
    return config

def test_load_datasets(config_fixture):
    train_dataset, eval_dataset, test_dataset = finetune_model.load_datasets(config_fixture)
    for dataset in [train_dataset, eval_dataset, test_dataset]:
        features = dataset.features
        assert "input_ids" in features
        assert "attention_mask" in features
        assert "labels" in features

def test_load_and_configure_lora_model(config_fixture):
    model = finetune_model.load_and_configure_lora_model(config_fixture)
    peft_config = model.peft_config['default']

    # Test that loaded model was configured correctly by testing some parameters 
    assert peft_config.base_model_name_or_path == config_fixture.model_name
    assert len(peft_config.target_modules) == len(config_fixture.lora_target_modules)
    assert peft_config.r == config_fixture.lora_r  

def test_save_log(config_fixture):
    log_file = config_fixture.trainer_output_dir_path / "training_log.json"
    log_file.unlink(missing_ok=True)  # Delete if file exists in directory
    trainer = MagicMock()
    trainer.state.log_history = [
        {"loss": 1.23, "epoch": 1, "step": 10},
        {"eval_loss": 1.10, "epoch": 1, "step": 20},
    ]

    finetune_model.save_log(config_fixture, trainer)

    assert log_file.exists()
    content = json.loads(log_file.read_text())
    assert "train_losses" in content and len(content["train_losses"]) == 1
    assert "eval_losses" in content and len(content["eval_losses"]) == 1
    log_file.unlink(missing_ok=True)  # Delete log file so folder is clean for next test run
