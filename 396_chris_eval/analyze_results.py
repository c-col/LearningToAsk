import json
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

def load_results(base_dir: str, model_size: str, dataset: str) -> Dict:
    """Load results for a specific model and dataset combination.
    
    Args:
        base_dir: Base directory containing results
        model_size: Model size (7B, 32B)
        dataset: Dataset name (8, 16, bigbench)
    
    Returns:
        Dictionary containing results data
    """
    if dataset == "8 things (C = 8)":
        dataset_name = "contrast_sets_8"
    elif dataset == "16 things (C = 16)":
        dataset_name = "contrast_sets_16"
    else:
        assert dataset == "bigbench (C = 29)", f"Invalid dataset: {dataset}"
        dataset_name = "contrast_sets_bigbench"
        
    if model_size == "7B":
        model_name = "deepseek-r1-distill-qwen-7b-mka"
    else:
        model_name = "deepseek-r1-distill-qwen-32b-ldr"
    
    results_path = Path(base_dir) / f"results__{dataset_name}__{model_name}/game_results_clean.json"
    if not results_path.exists():
        print(f"Warning: No results found for {model_size} on {dataset} at {results_path}")
        return None
    
    return json.loads(results_path.read_text())

def analyze_games(results: Dict) -> Tuple[float, float, float, float]:
    """Analyze game results for wins, turns per win, and information gains.
    
    Args:
        results: Dictionary containing game results
        
    Returns:
        Tuple of (win_rate, avg_turns_per_win, avg_info_gain, avg_ideal_info_gain)
    """
    if not results:
        return 0.0, 0.0, 0.0, 0.0
        
    games = results["results"]
    total_games = len(games)
    wins = 0
    total_turns_wins = 0
    total_info_gain = 0
    total_ideal_info_gain = 0
    total_turns = 0
    
    for game in games.values():
        # Count wins
        if game["won_on_turn"] is not None:
            wins += 1
            total_turns_wins += game["won_on_turn"]
            
        # Calculate average information gains
        for turn in game["turn_history"]:
            if turn["information_gain"] is not None:
                total_info_gain += turn["information_gain"]
                total_ideal_info_gain += turn["ideal_information_gain"]
                total_turns += 1
    
    win_rate = wins / total_games if total_games > 0 else 0
    avg_turns_per_win = total_turns_wins / wins if wins > 0 else 0
    avg_info_gain = total_info_gain / total_turns if total_turns > 0 else 0
    avg_ideal_info_gain = total_ideal_info_gain / total_turns if total_turns > 0 else 0
    
    return win_rate, avg_turns_per_win, avg_info_gain, avg_ideal_info_gain

def plot_model_comparison(results: Dict[str, Dict[str, Tuple[float, float, float, float]]], plots_dir: Path):
    """Plot comparison of models across datasets.
    
    Args:
        results: Dictionary mapping model sizes to dataset results
        plots_dir: Directory to save plots
    """
    # datasets = ["8 things (C = 8)", "16 things (C = 16)", "bigbench (C = 29)"]
    datasets = ["8 things (C = 8)", "16 things (C = 16)"]
    metrics = ["Win Rate", "Avg Turns per Win", "Avg Info Gain", "Avg Ideal Info Gain"]
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle("Model Performance Comparison")
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        data = {
            "7B": [results["7B"][dataset][i] for dataset in datasets],
            "32B": [results["32B"][dataset][i] for dataset in datasets]
        }
        
        x = np.arange(len(datasets))
        width = 0.35
        
        ax.bar(x - width/2, data["7B"], width, label="7B")
        ax.bar(x + width/2, data["32B"], width, label="32B")
        
        ax.set_title(metric)
        ax.set_xticks(x)
        ax.set_xticklabels(datasets)
        ax.legend()
        
        if metric == "Win Rate":
            ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(plots_dir / "model_comparison.png")
    plt.close()

def plot_info_gain_over_turns(results: Dict, model: str, dataset: str, plots_dir: Path):
    """Plot average information gain and ideal information gain vs turn number.
    
    Args:
        results: Dictionary containing results for a model/dataset combination
        model: Model name (e.g., "7B")
        dataset: Dataset name
        plots_dir: Directory to save plots
    """
    if not results:
        return
        
    games = results["results"]
    
    # Calculate averages per turn
    max_turns = max(len(game["turn_history"]) for game in games.values())
    avg_info_gains = np.zeros(max_turns)
    avg_ideal_gains = np.zeros(max_turns)
    counts = np.zeros(max_turns)
    
    for game in games.values():
        for turn_idx, turn in enumerate(game["turn_history"]):
            if turn["information_gain"] is not None:
                avg_info_gains[turn_idx] += turn["information_gain"]
                avg_ideal_gains[turn_idx] += turn["ideal_information_gain"]
                counts[turn_idx] += 1
    
    # Avoid division by zero
    mask = counts > 0
    avg_info_gains[mask] /= counts[mask]
    avg_ideal_gains[mask] /= counts[mask]
    
    # Create figure and plot averages
    plt.figure(figsize=(10, 6))
    turns = np.arange(1, max_turns + 1)
    
    plt.plot(turns[mask], avg_info_gains[mask], 'b-', linewidth=2, label='Actual')
    plt.plot(turns[mask], avg_ideal_gains[mask], 'r-', linewidth=2, label='Ideal')
    
    plt.title(f'Average Information Gain per Turn ({model} on {dataset})')
    plt.xlabel('Turn Number')
    plt.ylabel('Information Gain')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plt.savefig(plots_dir / f'info_gain_over_turns_{model}_{dataset.replace(" ", "_")}.png')
    plt.close()

def main():
    base_dir = "../data/game_sets/test/outputs"
    models = ["7B", "32B"]
    # datasets = ["8 things (C = 8)", "16 things (C = 16)", "bigbench (C = 29)"]
    datasets = ["8 things (C = 8)", "16 things (C = 16)"]
    
    # Create plots directory
    plots_dir = Path("../data/game_sets/test/outputs/plots")
    plots_dir.mkdir(exist_ok=True)

    # Collect results
    results = {model: {} for model in models}
    for model in models:
        for dataset in datasets:
            data = load_results(base_dir, model, dataset)
            if data:
                results[model][dataset] = analyze_games(data)
                # Plot information gain over turns for this model/dataset
                plot_info_gain_over_turns(data, model, dataset, plots_dir)
            else:
                results[model][dataset] = (0.0, 0.0, 0.0, 0.0)
    
    # Print analysis
    print("\nModel Performance Analysis")
    print("=" * 50)
    
    # How does 7B fare as games get harder
    print("\n7B Performance Across Datasets:")
    print("-" * 30)
    for dataset in datasets:
        win_rate, turns, info_gain, ideal_info_gain = results["7B"][dataset]
        print(f"\nDataset: {dataset}")
        print(f"Win Rate: {win_rate:.2%}")
        print(f"Avg Turns per Win: {turns:.1f}")
        print(f"Avg Info Gain: {info_gain:.3f}")
        print(f"Avg Ideal Info Gain: {ideal_info_gain:.3f}")
        print(f"Info Gain Efficiency: {(info_gain/ideal_info_gain*100):.1f}% of ideal")

    # How does 32B fare as games get harder
    print("\n32B Performance Across Datasets:")
    print("-" * 30)
    for dataset in datasets:
        win_rate, turns, info_gain, ideal_info_gain = results["32B"][dataset]
        print(f"\nDataset: {dataset}")
        print(f"Win Rate: {win_rate:.2%}")
        print(f"Avg Turns per Win: {turns:.1f}")
        print(f"Avg Info Gain: {info_gain:.3f}")
        print(f"Avg Ideal Info Gain: {ideal_info_gain:.3f}")
        print(f"Info Gain Efficiency: {(info_gain/ideal_info_gain*100):.1f}% of ideal")
    
    # Compare 7B vs 32B on each dataset
    print("\n7B vs 32B Comparison:")
    print("-" * 30)
    for dataset in datasets:
        print(f"\nDataset: {dataset}")
        for model in models:
            win_rate, turns, info_gain, ideal_info_gain = results[model][dataset]
            print(f"\n{model}:")
            print(f"Win Rate: {win_rate:.2%}")
            print(f"Avg Turns per Win: {turns:.1f}")
            print(f"Avg Info Gain: {info_gain:.3f}")
            print(f"Avg Ideal Info Gain: {ideal_info_gain:.3f}")
            print(f"Info Gain Efficiency: {(info_gain/ideal_info_gain*100):.1f}% of ideal")
    
    # Generate plots
    plot_model_comparison(results, plots_dir)
    print("\nPlots saved in plots/:")
    print("- model_comparison.png")
    print("- info_gain_over_turns_*.png")

if __name__ == "__main__":
    main() 