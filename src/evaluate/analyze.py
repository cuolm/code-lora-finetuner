import json
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from .config import Config


logger = logging.getLogger("src.evaluate.analyze")


def calculate_metric_stats(config, metric_name: str, higher_is_better: bool) -> dict:
    base_scores, lora_scores = [], []
    
    with open(config.benchmark_evaluation_results_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            if metric_name == config.codebleu_metric_name and not data.get("codebleu_valid", True):
                continue
            base_scores.append(data[f'base_{metric_name}'])
            lora_scores.append(data[f'lora_{metric_name}'])
    
    if not base_scores:
        logger.error(f"No {metric_name} results found.")
        return {}
    
    base_array_np = np.array(base_scores)
    lora_array_np = np.array(lora_scores)
    base_average_np = np.mean(base_array_np)
    lora_average_np = np.mean(lora_array_np)
    if higher_is_better:
        improvement_np = lora_average_np - base_average_np
    else:
        improvement_np = base_average_np - lora_average_np
    
    logger.info(f"\n=== {metric_name.upper()} SUMMARY ===")
    logger.info(f"Examples: {len(base_array_np)}")
    logger.info(f"Base average: {base_average_np:.3f} | LoRA avg: {lora_average_np:.3f}")
    logger.info(f"Improvement: {improvement_np:+.3f}")

    is_binary = False
    if (metric_name == config.exact_match_metric_name) or (metric_name == config.line_match_metric_name):
        is_binary = True
    
    metric_stats_np =  {
        "metric": metric_name,
        "base_array_np": base_array_np,
        "lora_array_np": lora_array_np,
        "base_average_np": base_average_np,
        "lora_average_np": lora_average_np,
        "is_binary": is_binary, 
        "higher_is_better": higher_is_better
    }

    return metric_stats_np


def plot_metric_and_save(stats: dict, metric_name: str, plot_path: Path) -> None:
    base_array_np = stats["base_array_np"]
    lora_array_np = stats["lora_array_np"]
    base_average_np = stats["base_average_np"]
    lora_average_np = stats["lora_average_np"]
    
    if stats["is_binary"]:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f'{metric_name.title()} Scores', fontsize=16) 
        
        ax1.bar(['Base', 'LoRA'], [base_average_np, lora_average_np], color=['steelblue', 'darkorange'])
        ax1.set_title(f'{metric_name.title()} Success Rate')
        ax1.set_ylim(0, 1)
        ax1.set_ylabel('Rate')
        
        wins = np.sum((lora_array_np == 1) & (base_array_np == 0))
        losses = np.sum((lora_array_np == 0) & (base_array_np == 1))
        ties = len(base_array_np) - wins - losses
        ax2.bar(['LoRA Wins', 'Ties', 'LoRA Losses'], [wins, ties, losses], color=['green', 'gray', 'red'])
        ax2.set_title('Example-Level Transitions')
        ax2.set_ylabel('Count')
        
    else:
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'{metric_name.title()} Scores', fontsize=16)
        y_axis_limit = 25
        
        # dynamic bounds, prevent outlier examples to screw the plot 
        max_val = max(np.max(base_array_np), np.max(lora_array_np))
        use_limit = max_val > y_axis_limit
        if use_limit:
            axis_upper_bound = y_axis_limit
        else:
            axis_upper_bound = max_val * 1.1

        axs[0, 0].bar(['Base', 'LoRA'], [base_average_np, lora_average_np], color=['steelblue', 'darkorange'])
        axs[0, 0].set_title('Average Scores')
        
        axs[0, 1].boxplot([base_array_np, lora_array_np], labels=['Base', 'LoRA'], patch_artist=True, boxprops=dict(facecolor='lightblue', color='blue'), medianprops=dict(color='red', linewidth=2))
        axs[0, 1].set_title('Score Distribution')
        axs[0, 1].set_ylim(0, axis_upper_bound)
        if use_limit:
            axs[0, 1].text(0.95, 0.95, f'Note: Values > {y_axis_limit} hidden', transform=axs[0, 1].transAxes, ha='right', va='top', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
        
        axs[1, 0].scatter(base_array_np, lora_array_np, alpha=0.5, color='steelblue')
        axs[1, 0].plot([0, axis_upper_bound], [0, axis_upper_bound], 'k--', alpha=0.75, label='y=x (Tie)')
        axs[1, 0].set_title('Base vs LoRA (Per Example)')
        axs[1, 0].set_xlim(0, axis_upper_bound)
        axs[1, 0].set_ylim(0, axis_upper_bound)
        axs[1, 0].legend(loc='upper left')
        if use_limit:
            axs[1, 0].text(0.95, 0.05, f'Note: Values > {y_axis_limit} hidden', transform=axs[1, 0].transAxes, ha='right', va='bottom', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
        
        differences = (lora_array_np - base_array_np) if stats["higher_is_better"] else (base_array_np - lora_array_np)
        axs[1, 1].hist(differences, bins=30, color='purple', alpha=0.7, edgecolor='black')
        axs[1, 1].axvline(0, color='black', linestyle='dashed')
        axs[1, 1].axvline(np.mean(differences), color='red', linestyle='solid')
        axs[1, 1].axvline(np.mean(differences), color='red', linestyle='solid', linewidth=2, label='Mean Diff')
        axs[1, 1].set_title('Improvement Distribution')
        hist_range = (-axis_upper_bound, axis_upper_bound)
        axs[1, 1].set_xlim(hist_range)
        axs[1, 1].legend(loc='upper left')
        if use_limit:
            axs[1, 1].text(0.95, 0.95, f'Note: |Diff| > {y_axis_limit} hidden', transform=axs[1, 1].transAxes, ha='right', va='top', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close(fig)


def save_evaluation_report(config: Config, checkpoint_name: str, all_metric_stats_np: list[dict]) -> None:
    """Writes all processed metrics to a single summary JSON file."""
    all_metric_stats = []
    for stat_np in all_metric_stats_np:
        stat = {
            "metric": stat_np["metric"],
            "base_array": stat_np["base_array_np"].tolist(),
            "lora_array": stat_np["lora_array_np"].tolist(),
            "base_average": float(stat_np["base_average_np"]),
            "lora_average": float(stat_np["lora_average_np"]),
            "is_binary": stat_np["is_binary"], 
            "higher_is_better": stat_np["higher_is_better"]
        }
        all_metric_stats.append(stat)

    report_content = {
        "checkpoint": checkpoint_name,
        "evaluation_date": datetime.now().isoformat(timespec="seconds"),
        "all_metric_stats": all_metric_stats 
    }
    with open(config.benchmark_evaluation_report_path, "w") as report_file:
        json.dump(report_content, report_file, indent=4)
    logger.info(f"Evaluation report saved to: {config.benchmark_evaluation_report_path}")


def plot_metric_averages_and_save(benchmark_evaluation_report_path: Path, plot_path: Path) -> None:
    try:
        with benchmark_evaluation_report_path.open("r") as f:
            data = json.load(f)
    except Exception:
        logger.exception(f"Failed to process evaluation report at: {benchmark_evaluation_report_path}")
        return

    all_metric_stats = data.get("all_metric_stats", [])
    if not all_metric_stats:
        logger.error("No results found in evaluation report.")
        return

    number_of_metrics = len(all_metric_stats)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, metric_stats in enumerate(all_metric_stats):
        metric_name = metric_stats["metric"]
        base_average = metric_stats["base_average"]
        lora_average = metric_stats["lora_average"]
        n_examples = len(metric_stats["base_array"])
        
        ax = axes[idx]
        bars = ax.bar(['Base', 'LoRA'], [base_average, lora_average],color=['steelblue', 'darkorange'], alpha=0.8)
        
        max_axes_val = max(base_average, lora_average) * 1.1
        ax.set_ylim(0, max_axes_val)
        
        ax.set_ylabel('Score')
        ax.set_title(f'{metric_name.title()}\nN={n_examples}')
        
        for bar, val in zip(bars, [base_average, lora_average]):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max_axes_val*0.02, f'{val:.3f}', ha='center', va='bottom', fontsize=11)
    
    for idx in range(number_of_metrics, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Base vs LoRA Average Scores', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"All-metrics average plotted and saved to: {plot_path}")
