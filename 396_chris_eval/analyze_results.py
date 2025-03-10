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

def find_failed_game_with_zero_gain(results: Dict) -> Tuple[str, Dict]:
    """Find a game where the model failed and had consecutive zero information gain questions.
    
    Args:
        results: Dictionary containing game results
        
    Returns:
        Tuple of (game_idx, game_data) or (None, None) if no such game found
    """
    if not results:
        return None, None
        
    games = results["results"]
    
    for game_idx, game in games.items():
        # Check if game was lost
        if game["won_on_turn"] is None:
            # Look for consecutive zero information gain questions
            zero_streak = 0
            for turn in game["turn_history"]:
                if turn["information_gain"] == 0:
                    zero_streak += 1
                    if zero_streak >= 3:  # At least 3 consecutive zero gain questions
                        return game_idx, game
                else:
                    zero_streak = 0
                    
    return None, None

def format_judge_response(response):
    """Format judge response nicely.
    
    Args:
        response: Judge response (string or dictionary)
        
    Returns:
        Formatted string
    """
    if isinstance(response, dict):
        # Format dictionary of entity-answer pairs
        lines = []
        for entity, answer in response.items():
            lines.append(f"  {entity}: {answer}")
        return "\n" + "\n".join(lines)
    else:
        # Simple string response
        return response

def create_game_text_summary(game_idx: str, game: Dict) -> str:
    """Create a human-readable text summary of a game.
    
    Args:
        game_idx: Index of the game
        game: Game data dictionary
        
    Returns:
        Text summary of the game
    """
    lines = []
    
    # Game overview
    lines.append("GAME OVERVIEW")
    lines.append("=" * 50)
    lines.append(f"Game Index: {game_idx}")
    lines.append(f"Target: {game['target']}")
    lines.append(f"Candidate Entities: {', '.join(game['candidate_entities'])}")
    lines.append(f"Won on Turn: {game['won_on_turn'] if game['won_on_turn'] is not None else 'Failed'}")
    lines.append(f"Final Entities: {', '.join(game['final_entities']) if game['final_entities'] else 'None'}")
    lines.append(f"Number of Turns: {game['number_of_turns']}")
    lines.append(f"Game Over: {game['game_over']}")
    
    # Turn history
    lines.append("\nTURN HISTORY")
    lines.append("=" * 50)
    
    for i, turn in enumerate(game['turn_history'], 1):
        lines.append(f"\nTURN {i}")
        lines.append("-" * 50)
        
        if "guesser_output" in turn:
            lines.append(f"Guesser Output: {turn['guesser_output']}\n")

        lines.append("-" * 10)
        
        lines.append(f"Question: {turn['question']}")
        lines.append(f"Answer: {turn['judge_response'][game['target']]}")
        

        if "remaining_entities" in turn:
            entities = turn["remaining_entities"]
            lines.append(f"Remaining Entities ({len(entities)}): {', '.join(entities)}")
        
        lines.append(f"Information Gain: {turn['information_gain']:.3f}")
        lines.append(f"Ideal Information Gain: {turn['ideal_information_gain']:.3f}")
    
    return "\n".join(lines)

def save_failed_game_analysis(game_idx: str, game: Dict, model: str, dataset: str, failed_games_dir: Path) -> str:
    """Save failed game data to JSON and TXT files.
    
    Args:
        game_idx: Index of the game
        game: Game data dictionary
        model: Model name
        dataset: Dataset name
        failed_games_dir: Directory to save analysis files
        
    Returns:
        Path to the saved JSON file
    """
    # Add game index to the game data
    game_data = {
        "game_idx": game_idx,
        **game
    }
            
    # Get actual dataset name
    if dataset == "8 things (C = 8)":
        dataset_name = "contrast_sets_8"
    elif dataset == "16 things (C = 16)":
        dataset_name = "contrast_sets_16"
    else:
        assert dataset == "bigbench (C = 29)", f"Invalid dataset: {dataset}"
        dataset_name = "contrast_sets_bigbench"
        
    # Get actual model name
    if model == "7B":
        model_name = "deepseek-r1-distill-qwen-7b-mka"
    else:
        model_name = "deepseek-r1-distill-qwen-32b-ldr"
    
    # Base filename
    base_filename = f"failed_game__{dataset_name}__{model_name}__game_{game_idx}"
    
    # Save JSON file
    json_filename = f"{base_filename}.json"
    json_filepath = failed_games_dir / json_filename
    
    with open(json_filepath, 'w') as f:
        json.dump(game_data, f, indent=2)
    
    # Save TXT file
    txt_filename = f"{base_filename}.txt"
    txt_filepath = failed_games_dir / txt_filename
    
    with open(txt_filepath, 'w') as f:
        f.write(create_game_text_summary(game_idx, game))
        
    return json_filename

def main():
    base_dir = "../data/game_sets/test/outputs"
    models = ["7B", "32B"]
    datasets = ["8 things (C = 8)", "16 things (C = 16)"]
    
    # Create output directories
    plots_dir = Path("../data/game_sets/test/outputs/plots")
    failed_games_dir = Path("../data/game_sets/test/outputs/failed_games")
    plots_dir.mkdir(exist_ok=True)
    failed_games_dir.mkdir(exist_ok=True)

    # Collect results and find failed games
    results = {model: {} for model in models}
    failed_games_found = False
    
    for model in models:
        for dataset in datasets:
            data = load_results(base_dir, model, dataset)
            if data:
                results[model][dataset] = analyze_games(data)
                # Plot information gain over turns for this model/dataset
                plot_info_gain_over_turns(data, model, dataset, plots_dir)
                
                # Find and analyze failed games
                game_idx, failed_game = find_failed_game_with_zero_gain(data)
                if failed_game:
                    filename = save_failed_game_analysis(game_idx, failed_game, model, dataset, failed_games_dir)
                    failed_games_found = True
                    print(f"Failed game analysis saved to: {filename}")
            else:
                results[model][dataset] = (0.0, 0.0, 0.0, 0.0)
    
    if not failed_games_found:
        print("\nNo failed games with consecutive zero information gain found.")
    
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

    # How does 32B fare as games get harder
    print("\n32B Performance Across Datasets:")
    print("-" * 30)
    for dataset in datasets:
        win_rate, turns, info_gain, ideal_info_gain = results["32B"][dataset]
        print(f"\nDataset: {dataset}")
        print(f"Win Rate: {win_rate:.2%}")
        print(f"Avg Turns per Win: {turns:.1f}")
        print(f"Avg Info Gain: {info_gain:.3f}")
    
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
    
    # Generate plots
    plot_model_comparison(results, plots_dir)
    print("\nPlots saved in plots/:")
    print("- model_comparison.png")
    print("- info_gain_over_turns_*.png")

if __name__ == "__main__":
    main() 