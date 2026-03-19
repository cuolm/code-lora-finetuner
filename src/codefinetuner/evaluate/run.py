import argparse
import logging
import logging.config

from transformers.trainer_utils import get_last_checkpoint

from .analyze import analyze_metric, save_all_metric_stats, get_plot_path, plot_metric_and_save, plot_all_metric_averages_and_save
from .config import Config
from .evaluate import evaluate_and_save
from .generate import generate_and_save
from .benchmark import create_benchmark_dataset


logger = logging.getLogger(__name__)


def _setup_logger(log_level: str) -> None:
    """
    Configure the root logger ("") to capture logs from the entry point (__main__),
    all internal sub-modules, and third-party libraries via a single handler.
    """
    logger_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            }
        },
        "handlers": {
            "stderr_handler": {
                "level": log_level,
                "class": "logging.StreamHandler",
                "formatter": "standard",
            }
        },
        "loggers": {
            "": {  # "" corresponds to root logger 
                "handlers": ["stderr_handler"],
                "level": log_level,
                "propagate": False,
            }
        }
    }
    logging.config.dictConfig(logger_config)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run evaluation")
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        metavar="NAME_OR_LAST",
        help='Checkpoint name, or "last"'
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        default=False,  
        help="Skip generation, use existing data to update plots"
    )
    parser.add_argument(
        "--overwrite-dataset",
        action="store_true",
        default=False,  
        help="Overwrite benchmark dataset"
    )
    return parser.parse_args()


def _ensure_checkpoints(config: Config) -> None:
    checkpoints_dir = config.trainer_checkpoints_dir_path
    if not checkpoints_dir.exists():
        logger.error(f"Checkpoint directory does not exist: {checkpoints_dir}")
        raise RuntimeError("Checkpoint directory not found.")
    if not list(checkpoints_dir.iterdir()):
        logger.error(f"Checkpoint directory is empty: {checkpoints_dir}")
        raise RuntimeError("No checkpoints in directory.")


def run(config: Config, checkpoint: str, plot_only: bool, overwrite_dataset: bool) -> None:
    if overwrite_dataset or not config.benchmark_dataset_path.exists():
        dataset_len = create_benchmark_dataset(config)
        logger.info(f"Created new benchmark dataset '{config.benchmark_dataset_path}' with '{dataset_len}' examples")
    else:
        logger.info(f"Proceeding with existing file '{config.benchmark_dataset_path}'...")
    
    if not plot_only: 
        _ensure_checkpoints(config)
        if checkpoint == "last":
            checkpoint_path = get_last_checkpoint(config.trainer_checkpoints_dir_path)
        else:
            checkpoint_path = config.trainer_checkpoints_dir_path / checkpoint
        generate_and_save(config, checkpoint_path)
        evaluate_and_save(config)

    all_metric_stats_np = []
    for metric_name, higher_is_better in config.metric_configs:
        metric_stats_np = analyze_metric(config, metric_name, higher_is_better)
        plot_path = get_plot_path(config.benchmark_evaluation_results_dir, metric_name)
        plot_metric_and_save(metric_stats_np, metric_name, plot_path)
        all_metric_stats_np.append(metric_stats_np)
    
    all_metric_averages_plot_path = get_plot_path(config.benchmark_evaluation_results_dir, "all_metric_averages")
    plot_all_metric_averages_and_save(all_metric_stats_np, all_metric_averages_plot_path) 
    save_all_metric_stats(config, checkpoint, all_metric_stats_np)


def main():
    config = Config()
    user_args = _parse_args()
    _setup_logger(user_args.log_level)
    run(
        config=config, 
        checkpoint=user_args.checkpoint,
        plot_only=user_args.plot_only, 
        overwrite_dataset=user_args.overwrite_dataset
    )


if __name__ == "__main__":
    main()
