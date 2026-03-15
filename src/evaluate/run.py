import argparse
import logging
import logging.config

from .analyze import calculate_metric_stats, save_evaluation_report, plot_metric_and_save, plot_metric_averages_and_save
from .config import Config
from .evaluate import evaluate_and_save
from .generate import generate_and_save
from .benchmark import ensure_benchmark_dataset


logger = logging.getLogger(__name__)


def _setup_logger(log_level: str) -> None:
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
            "src.evaluate": {  
                "handlers": ["stderr_handler"],
                "level": log_level,
                "propagate": False,
            }
        }
    }
    logging.config.dictConfig(logger_config)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate code comparison")
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


def main():
    config = Config()
    user_args = _parse_args()
    _setup_logger(user_args.log_level)

    ensure_benchmark_dataset(config, user_args)
    
    if not user_args.plot_only:  
        generate_and_save(config, user_args)
        evaluate_and_save(config)

    metric_configurations = [
        (config.sentencebleu_metric_name, config.sentencebleu_plot_path, True),
        (config.codebleu_metric_name, config.codebleu_plot_path, True),
        (config.exact_match_metric_name, config.exact_match_plot_path, True),
        (config.line_match_metric_name, config.line_match_plot_path, True),
        (config.perplexity_name, config.perplexity_plot_path, False),
    ]

    all_metric_stats_np = []
    for metric_name, plot_path, higher_is_better in metric_configurations:
        metric_stats_np = calculate_metric_stats(config, metric_name, higher_is_better)
        plot_metric_and_save(metric_stats_np, metric_name, plot_path)
        all_metric_stats_np.append(metric_stats_np)
    
    save_evaluation_report(config, user_args.checkpoint, all_metric_stats_np)
    plot_metric_averages_and_save(config.benchmark_evaluation_report_path, config.benchmark_evaluation_averages_path) 


if __name__ == "__main__":
    main()