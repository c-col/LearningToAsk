import json
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import re

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

def normalize_question(question: str) -> str:
    """Normalize a question for comparison to detect repetitions.
    
    Args:
        question: The original question string
        
    Returns:
        Normalized question string
    """
    # Convert to lowercase
    normalized = question.lower()
    
    # Remove punctuation and extra whitespace
    normalized = re.sub(r'[^\w\s]', '', normalized)
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    
    return normalized

def analyze_games(results: Dict) -> Tuple[float, float, float, float, float, float, int, Dict]:
    """Analyze game results for wins, turns per win, and information gains.
    
    Args:
        results: Dictionary containing game results
        
    Returns:
        Tuple of (win_rate, avg_turns_per_win, avg_info_gain, avg_ideal_info_gain, 
                 zero_gain_question_rate, repeated_question_rate, total_questions,
                 repeated_questions_details)
    """
    if not results:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, {}
        
    games = results["results"]
    total_games = len(games)
    wins = 0
    total_turns_wins = 0
    total_info_gain = 0
    total_ideal_info_gain = 0
    total_turns = 0
    zero_gain_questions = 0
    repeated_questions = 0
    
    # Track repeated questions across all games
    repeated_questions_details = {}  # Format: {game_idx: {question: count}}
    
    for game_idx, game in games.items():
        # Count wins
        won_on_turn = game["won_on_turn"]
        if won_on_turn is not None:
            wins += 1
            total_turns_wins += won_on_turn
        
        # Track questions to detect repetitions
        asked_questions = set()
        question_counts = {}  # Count occurrences within this game
            
        # Calculate average information gains and count zero gain questions
        for turn_idx, turn in enumerate(game["turn_history"], 1):
            if turn["information_gain"] is not None:
                total_info_gain += turn["information_gain"]
                total_ideal_info_gain += turn["ideal_information_gain"]
                total_turns += 1
                
                # Count zero information gain questions, but exclude winning questions
                if turn["information_gain"] == 0 and not (won_on_turn is not None and turn_idx == won_on_turn):
                    zero_gain_questions += 1
                
                # Check for repeated questions
                normalized_question = normalize_question(turn["question"])
                if normalized_question in asked_questions:
                    repeated_questions += 1
                    # Track the repeated question
                    question_counts[normalized_question] = question_counts.get(normalized_question, 1) + 1
                else:
                    asked_questions.add(normalized_question)
                    question_counts[normalized_question] = 1
        
        # Save questions that were repeated (count > 1)
        game_repeated = {q: count for q, count in question_counts.items() if count > 1}
        if game_repeated:
            repeated_questions_details[game_idx] = game_repeated
    
    win_rate = wins / total_games if total_games > 0 else 0
    avg_turns_per_win = total_turns_wins / wins if wins > 0 else 0
    avg_info_gain = total_info_gain / total_turns if total_turns > 0 else 0
    avg_ideal_info_gain = total_ideal_info_gain / total_turns if total_turns > 0 else 0
    zero_gain_rate = zero_gain_questions / total_turns if total_turns > 0 else 0
    repeated_question_rate = repeated_questions / total_turns if total_turns > 0 else 0
    
    return win_rate, avg_turns_per_win, avg_info_gain, avg_ideal_info_gain, zero_gain_rate, repeated_question_rate, total_turns, repeated_questions_details

def plot_model_comparison(results: Dict[str, Dict[str, Tuple[float, float, float, float, float, float, int]]], plots_dir: Path):
    """Plot comparison of models across datasets with metrics grouped into three plots.
    
    Args:
        results: Dictionary mapping model sizes to dataset results
        plots_dir: Directory to save plots
    """
    datasets = ["8 things (C = 8)", "16 things (C = 16)", "bigbench (C = 29)"]
    
    # Define the three plot groups
    plot_groups = [
        {
            "title": "Win Rate and Turns per Win",
            "metrics": ["Win Rate", "Avg Turns per Win"],
            "indices": [0, 1],
            "y_limits": [(0, 1), None],
            "y_labels": ["Percentage", "Number of Turns"],
            "formatters": [lambda y, _: '{:.0%}'.format(y), None]
        },
        {
            "title": "Information Gain per Turn",
            "metrics": ["Avg Info Gain per Turn", "Avg Ideal Info Gain per Turn"],
            "indices": [2, 3],
            "y_limits": [None, None],
            "y_labels": ["Information Gain", "Information Gain"],
            "formatters": [None, None]
        },
        {
            "title": "Question Quality Metrics",
            "metrics": ["Zero Gain Rate", "Repeated Question Rate"],
            "indices": [4, 5],
            "y_limits": [(0, 1), (0, 1)],
            "y_labels": ["Percentage", "Percentage"],
            "formatters": [lambda y, _: '{:.0%}'.format(y), lambda y, _: '{:.0%}'.format(y)]
        }
    ]
    
    # Create and save each plot group
    for group_idx, group in enumerate(plot_groups):
        # Create figure with 2 subplots side by side
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(group["title"], fontsize=16)
        
        for i, (metric, idx, y_limit, y_label, formatter) in enumerate(zip(
            group["metrics"], group["indices"], group["y_limits"], 
            group["y_labels"], group["formatters"]
        )):
            ax = axes[i]
            data = {
                "7B": [results["7B"][dataset][idx] for dataset in datasets],
                "32B": [results["32B"][dataset][idx] for dataset in datasets]
            }
            
            x = np.arange(len(datasets))
            width = 0.35
            
            # Plot bars
            ax.bar(x - width/2, data["7B"], width, label="7B", color='#1f77b4')
            ax.bar(x + width/2, data["32B"], width, label="32B", color='#ff7f0e')
            
            # Set title and labels
            ax.set_title(metric, fontsize=14)
            ax.set_xlabel("Dataset", fontsize=12)
            ax.set_ylabel(y_label, fontsize=12)
            
            # Set x-ticks
            ax.set_xticks(x)
            ax.set_xticklabels(datasets, rotation=45, ha='right', fontsize=10)
            
            # Set y-limits if specified
            if y_limit:
                ax.set_ylim(y_limit)
            
            # Format y-axis if formatter is provided
            if formatter:
                ax.yaxis.set_major_formatter(plt.FuncFormatter(formatter))
            
            # Add legend
            ax.legend()
            
            # Add grid for readability
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Add value labels on top of bars
            for j, rect in enumerate(ax.patches):
                height = rect.get_height()
                if metric in ["Win Rate", "Zero Gain Rate", "Repeated Question Rate"]:
                    label = f"{height:.0%}"
                elif metric == "Avg Turns per Win":
                    label = f"{height:.1f}"
                else:
                    label = f"{height:.2f}"
                    
                ax.text(rect.get_x() + rect.get_width()/2., height + 0.01,
                        label, ha='center', va='bottom', fontsize=9, rotation=0)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.88)  # Make room for the suptitle
        plt.savefig(plots_dir / f"model_comparison_{group_idx+1}_{group['title'].replace(' ', '_').lower()}.png", dpi=300)
        plt.close()
        
    # Also create a summary plot with just win rate and info gain
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Key Performance Metrics", fontsize=16)
    
    # Plot Win Rate
    ax = axes[0]
    data = {
        "7B": [results["7B"][dataset][0] for dataset in datasets],
        "32B": [results["32B"][dataset][0] for dataset in datasets]
    }
    
    x = np.arange(len(datasets))
    width = 0.35
    
    ax.bar(x - width/2, data["7B"], width, label="7B", color='#1f77b4')
    ax.bar(x + width/2, data["32B"], width, label="32B", color='#ff7f0e')
    
    ax.set_title("Win Rate", fontsize=14)
    ax.set_xlabel("Dataset", fontsize=12)
    ax.set_ylabel("Percentage", fontsize=12)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha='right', fontsize=10)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add value labels
    for j, rect in enumerate(ax.patches):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., height + 0.01,
                f"{height:.0%}", ha='center', va='bottom', fontsize=9, rotation=0)
    
    # Plot Info Gain
    ax = axes[1]
    data = {
        "7B": [results["7B"][dataset][2] for dataset in datasets],
        "32B": [results["32B"][dataset][2] for dataset in datasets]
    }
    
    ax.bar(x - width/2, data["7B"], width, label="7B", color='#1f77b4')
    ax.bar(x + width/2, data["32B"], width, label="32B", color='#ff7f0e')
    
    ax.set_title("Avg Info Gain", fontsize=14)
    ax.set_xlabel("Dataset", fontsize=12)
    ax.set_ylabel("Information Gain", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha='right', fontsize=10)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add value labels
    for j, rect in enumerate(ax.patches):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., height + 0.01,
                f"{height:.2f}", ha='center', va='bottom', fontsize=9, rotation=0)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.savefig(plots_dir / "model_comparison_summary.png", dpi=300)
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

def find_successful_game(results: Dict) -> Tuple[str, Dict]:
    """Find a game where the model successfully won.
    
    Args:
        results: Dictionary containing game results
        
    Returns:
        Tuple of (game_idx, game_data) or (None, None) if no such game found
    """
    if not results:
        return None, None
        
    games = results["results"]
    
    # First try to find a game with high information gain
    best_game_idx = None
    best_game = None
    best_avg_info_gain = -1
    
    for game_idx, game in games.items():
        # Check if game was won
        if game["won_on_turn"] is not None:
            # Calculate average information gain
            total_info_gain = sum(turn["information_gain"] for turn in game["turn_history"] 
                                if turn["information_gain"] is not None)
            avg_info_gain = total_info_gain / len(game["turn_history"])
            
            if avg_info_gain > best_avg_info_gain:
                best_avg_info_gain = avg_info_gain
                best_game_idx = game_idx
                best_game = game
    
    return best_game_idx, best_game

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

def save_game_analysis(game_idx: str, game: Dict, model: str, dataset: str, output_dir: Path, prefix: str = "game") -> str:
    """Save game data to JSON and TXT files.
    
    Args:
        game_idx: Index of the game
        game: Game data dictionary
        model: Model name
        dataset: Dataset name
        output_dir: Directory to save analysis files
        prefix: Prefix for the filename (default: "game")
        
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
    base_filename = f"{prefix}__{dataset_name}__{model_name}__game_{game_idx}"
    
    # Save JSON file
    json_filename = f"{base_filename}.json"
    json_filepath = output_dir / json_filename
    
    with open(json_filepath, 'w') as f:
        json.dump(game_data, f, indent=2)
    
    # Save TXT file
    txt_filename = f"{base_filename}.txt"
    txt_filepath = output_dir / txt_filename
    
    with open(txt_filepath, 'w') as f:
        f.write(create_game_text_summary(game_idx, game))
        
    return json_filename

def save_repeated_questions(model: str, dataset: str, repeated_questions_details: Dict, analysis_dir: Path):
    """Save detailed information about repeated questions.
    
    Args:
        model: Model name
        dataset: Dataset name
        repeated_questions_details: Dictionary of repeated questions by game
        analysis_dir: Directory to save analysis
    """
    if not repeated_questions_details:
        return
    
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
    
    # Create output
    output = []
    output.append(f"Repeated Questions Analysis: {model} on {dataset}")
    output.append("=" * 50)
    
    # Sort games by number of repeated questions
    sorted_games = sorted(
        repeated_questions_details.items(),
        key=lambda x: sum(count-1 for count in x[1].values()),
        reverse=True
    )
    
    for game_idx, questions in sorted_games:
        # Sort questions by count
        sorted_questions = sorted(questions.items(), key=lambda x: x[1], reverse=True)
        
        output.append(f"\nGame {game_idx}:")
        output.append("-" * 30)
        
        for question, count in sorted_questions:
            output.append(f"Asked {count} times: {question}")
    
    # Save to file
    filename = f"repeated_questions__{dataset_name}__{model_name}.txt"
    filepath = analysis_dir / filename
    
    with open(filepath, 'w') as f:
        f.write('\n'.join(output))
    
    return filepath

def save_performance_analysis(results, datasets, models, analysis_dir):
    """Save model performance analysis to a text file.
    
    Args:
        results: Dictionary of results
        datasets: List of dataset names
        models: List of model names
        analysis_dir: Directory to save analysis
    """
    output = []
    
    # Title
    output.append("Model Performance Analysis")
    output.append("=" * 50)
    
    # 7B Performance Across Datasets
    output.append("\n7B Performance Across Datasets:")
    output.append("-" * 30)
    for dataset in datasets:
        win_rate, turns, info_gain, ideal_info_gain, zero_gain_rate, repeated_rate, total_questions = results["7B"][dataset]
        output.append(f"\nDataset: {dataset}")
        output.append(f"Win Rate: {win_rate:.2%}")
        output.append(f"Avg Turns per Win: {turns:.1f}")
        output.append(f"Avg Info Gain: {info_gain:.3f}")
        output.append(f"Avg Ideal Info Gain: {ideal_info_gain:.3f}")
        output.append(f"Zero Gain Questions: {zero_gain_rate:.2%} ({int(zero_gain_rate * total_questions)} of {total_questions})")
        output.append(f"Repeated Questions: {repeated_rate:.2%} ({int(repeated_rate * total_questions)} of {total_questions})")

    # 32B Performance Across Datasets
    output.append("\n32B Performance Across Datasets:")
    output.append("-" * 30)
    for dataset in datasets:
        win_rate, turns, info_gain, ideal_info_gain, zero_gain_rate, repeated_rate, total_questions = results["32B"][dataset]
        output.append(f"\nDataset: {dataset}")
        output.append(f"Win Rate: {win_rate:.2%}")
        output.append(f"Avg Turns per Win: {turns:.1f}")
        output.append(f"Avg Info Gain: {info_gain:.3f}")
        output.append(f"Avg Ideal Info Gain: {ideal_info_gain:.3f}")
        output.append(f"Zero Gain Questions: {zero_gain_rate:.2%} ({int(zero_gain_rate * total_questions)} of {total_questions})")
        output.append(f"Repeated Questions: {repeated_rate:.2%} ({int(repeated_rate * total_questions)} of {total_questions})")
    
    # Compare 7B vs 32B on each dataset
    output.append("\n7B vs 32B Comparison:")
    output.append("-" * 30)
    for dataset in datasets:
        output.append(f"\nDataset: {dataset}")
        for model in models:
            win_rate, turns, info_gain, ideal_info_gain, zero_gain_rate, repeated_rate, total_questions = results[model][dataset]
            output.append(f"\n{model}:")
            output.append(f"Win Rate: {win_rate:.2%}")
            output.append(f"Avg Turns per Win: {turns:.1f}")
            output.append(f"Avg Info Gain: {info_gain:.3f}")
            output.append(f"Avg Ideal Info Gain: {ideal_info_gain:.3f}")
            output.append(f"Zero Gain Questions: {zero_gain_rate:.2%} ({int(zero_gain_rate * total_questions)} of {total_questions})")
            output.append(f"Repeated Questions: {repeated_rate:.2%} ({int(repeated_rate * total_questions)} of {total_questions})")
    
    # Save to file
    filepath = analysis_dir / "model_performance_analysis.txt"
    with open(filepath, 'w') as f:
        f.write('\n'.join(output))
    
    return filepath

def plot_info_gain_comparison(results_7b: Dict, results_32b: Dict, dataset: str, plots_dir: Path):
    """Plot information gain comparison between 7B and 32B models on the same graph.
    
    Args:
        results_7b: Results for 7B model
        results_32b: Results for 32B model
        dataset: Dataset name
        plots_dir: Directory to save plots
    """
    if not results_7b or not results_32b:
        return
        
    games_7b = results_7b["results"]
    games_32b = results_32b["results"]
    
    # Calculate averages per turn for 7B
    max_turns_7b = max(len(game["turn_history"]) for game in games_7b.values())
    avg_info_gains_7b = np.zeros(max_turns_7b)
    avg_ideal_gains_7b = np.zeros(max_turns_7b)
    counts_7b = np.zeros(max_turns_7b)
    
    for game in games_7b.values():
        for turn_idx, turn in enumerate(game["turn_history"]):
            if turn["information_gain"] is not None:
                avg_info_gains_7b[turn_idx] += turn["information_gain"]
                avg_ideal_gains_7b[turn_idx] += turn["ideal_information_gain"]
                counts_7b[turn_idx] += 1
    
    # Calculate averages per turn for 32B
    max_turns_32b = max(len(game["turn_history"]) for game in games_32b.values())
    avg_info_gains_32b = np.zeros(max_turns_32b)
    avg_ideal_gains_32b = np.zeros(max_turns_32b)
    counts_32b = np.zeros(max_turns_32b)
    
    for game in games_32b.values():
        for turn_idx, turn in enumerate(game["turn_history"]):
            if turn["information_gain"] is not None:
                avg_info_gains_32b[turn_idx] += turn["information_gain"]
                avg_ideal_gains_32b[turn_idx] += turn["ideal_information_gain"]
                counts_32b[turn_idx] += 1
    
    # Avoid division by zero
    mask_7b = counts_7b > 0
    avg_info_gains_7b[mask_7b] /= counts_7b[mask_7b]
    avg_ideal_gains_7b[mask_7b] /= counts_7b[mask_7b]
    
    mask_32b = counts_32b > 0
    avg_info_gains_32b[mask_32b] /= counts_32b[mask_32b]
    avg_ideal_gains_32b[mask_32b] /= counts_32b[mask_32b]
    
    # Create figure and plot averages
    plt.figure(figsize=(12, 7))
    turns_7b = np.arange(1, max_turns_7b + 1)
    turns_32b = np.arange(1, max_turns_32b + 1)
    
    # Plot 7B lines (blue)
    plt.plot(turns_7b[mask_7b], avg_info_gains_7b[mask_7b], 'b-', linewidth=2, label='7B Actual')
    plt.plot(turns_7b[mask_7b], avg_ideal_gains_7b[mask_7b], 'b:', linewidth=2, label='7B Ideal')
    
    # Plot 32B lines (orange)
    plt.plot(turns_32b[mask_32b], avg_info_gains_32b[mask_32b], color='#ff7f0e', linestyle='-', linewidth=2, label='32B Actual')
    plt.plot(turns_32b[mask_32b], avg_ideal_gains_32b[mask_32b], color='#ff7f0e', linestyle=':', linewidth=2, label='32B Ideal')
    
    plt.title(f'Information Gain Comparison on {dataset}', fontsize=14)
    plt.xlabel('Turn Number', fontsize=12)
    plt.ylabel('Information Gain', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Save plot
    dataset_name = dataset.replace(" ", "_")
    plt.savefig(plots_dir / f'info_gain_comparison_{dataset_name}.png', dpi=300)
    plt.close()

def main():
    base_dir = "../data/game_sets/test/outputs"
    models = ["7B", "32B"]
    datasets = ["8 things (C = 8)", "16 things (C = 16)", "bigbench (C = 29)"]
    
    # Create output directories
    plots_dir = Path("../data/game_sets/test/outputs/plots")
    failed_games_dir = Path("../data/game_sets/test/outputs/failed_games")
    successful_games_dir = Path("../data/game_sets/test/outputs/successful_games")
    analysis_dir = Path("../data/game_sets/test/outputs/analysis")
    
    plots_dir.mkdir(exist_ok=True)
    failed_games_dir.mkdir(exist_ok=True)
    successful_games_dir.mkdir(exist_ok=True)
    analysis_dir.mkdir(exist_ok=True)

    # Collect results and find games
    results = {model: {} for model in models}
    repeated_questions_data = {model: {} for model in models}
    failed_games_found = False
    successful_games_found = False
    
    # Store raw results for comparison plots
    raw_results = {model: {} for model in models}
    
    for model in models:
        for dataset in datasets:
            data = load_results(base_dir, model, dataset)
            if data:
                # Store raw results for comparison plots
                raw_results[model][dataset] = data
                
                analysis_results = analyze_games(data)
                results[model][dataset] = analysis_results[:-1]  # Exclude repeated_questions_details
                repeated_questions_data[model][dataset] = analysis_results[-1]  # Just the repeated_questions_details
                
                # Save repeated questions analysis
                if repeated_questions_data[model][dataset]:
                    repeated_file = save_repeated_questions(model, dataset, repeated_questions_data[model][dataset], analysis_dir)
                    print(f"Repeated questions analysis saved to: {repeated_file}")
                
                # Plot information gain over turns for this model/dataset
                plot_info_gain_over_turns(data, model, dataset, plots_dir)
                
                # Find and analyze failed games
                game_idx, failed_game = find_failed_game_with_zero_gain(data)
                if failed_game:
                    filename = save_game_analysis(game_idx, failed_game, model, dataset, failed_games_dir, "failed_game")
                    failed_games_found = True
                    print(f"Failed game analysis saved to: {filename}")
                
                # Find and analyze successful games
                game_idx, successful_game = find_successful_game(data)
                if successful_game:
                    filename = save_game_analysis(game_idx, successful_game, model, dataset, successful_games_dir, "successful_game")
                    successful_games_found = True
                    print(f"Successful game analysis saved to: {filename}")
            else:
                results[model][dataset] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)
                repeated_questions_data[model][dataset] = {}
                raw_results[model][dataset] = None
    
    # Create comparison plots for each dataset
    for dataset in datasets:
        if raw_results["7B"][dataset] and raw_results["32B"][dataset]:
            plot_info_gain_comparison(raw_results["7B"][dataset], raw_results["32B"][dataset], dataset, plots_dir)
            print(f"Information gain comparison plot saved for {dataset}")
    
    if not failed_games_found:
        print("\nNo failed games with consecutive zero information gain found.")
        
    if not successful_games_found:
        print("\nNo successful games found.")
    
    # Print and save analysis
    print("\nModel Performance Analysis")
    print("=" * 50)
    
    # How does 7B fare as games get harder
    print("\n7B Performance Across Datasets:")
    print("-" * 30)
    for dataset in datasets:
        win_rate, turns, info_gain, ideal_info_gain, zero_gain_rate, repeated_rate, total_questions = results["7B"][dataset]
        print(f"\nDataset: {dataset}")
        print(f"Win Rate: {win_rate:.2%}")
        print(f"Avg Turns per Win: {turns:.1f}")
        print(f"Avg Info Gain: {info_gain:.3f}")
        print(f"Avg Ideal Info Gain: {ideal_info_gain:.3f}")
        print(f"Zero Gain Questions: {zero_gain_rate:.2%} ({int(zero_gain_rate * total_questions)} of {total_questions})")
        print(f"Repeated Questions: {repeated_rate:.2%} ({int(repeated_rate * total_questions)} of {total_questions})")

    # How does 32B fare as games get harder
    print("\n32B Performance Across Datasets:")
    print("-" * 30)
    for dataset in datasets:
        win_rate, turns, info_gain, ideal_info_gain, zero_gain_rate, repeated_rate, total_questions = results["32B"][dataset]
        print(f"\nDataset: {dataset}")
        print(f"Win Rate: {win_rate:.2%}")
        print(f"Avg Turns per Win: {turns:.1f}")
        print(f"Avg Info Gain: {info_gain:.3f}")
        print(f"Avg Ideal Info Gain: {ideal_info_gain:.3f}")
        print(f"Zero Gain Questions: {zero_gain_rate:.2%} ({int(zero_gain_rate * total_questions)} of {total_questions})")
        print(f"Repeated Questions: {repeated_rate:.2%} ({int(repeated_rate * total_questions)} of {total_questions})")
    
    # Compare 7B vs 32B on each dataset
    print("\n7B vs 32B Comparison:")
    print("-" * 30)
    for dataset in datasets:
        print(f"\nDataset: {dataset}")
        for model in models:
            win_rate, turns, info_gain, ideal_info_gain, zero_gain_rate, repeated_rate, total_questions = results[model][dataset]
            print(f"\n{model}:")
            print(f"Win Rate: {win_rate:.2%}")
            print(f"Avg Turns per Win: {turns:.1f}")
            print(f"Avg Info Gain: {info_gain:.3f}")
            print(f"Avg Ideal Info Gain: {ideal_info_gain:.3f}")
            print(f"Zero Gain Questions: {zero_gain_rate:.2%} ({int(zero_gain_rate * total_questions)} of {total_questions})")
            print(f"Repeated Questions: {repeated_rate:.2%} ({int(repeated_rate * total_questions)} of {total_questions})")
    
    # Save analysis to file
    analysis_file = save_performance_analysis(results, datasets, models, analysis_dir)
    print(f"\nAnalysis saved to: {analysis_file}")
    
    # Generate plots
    plot_model_comparison(results, plots_dir)
    print("\nPlots saved in plots/:")
    print("- model_comparison_1_win_rate_and_turns_per_win.png")
    print("- model_comparison_2_information_gain_metrics.png")
    print("- model_comparison_3_question_quality_metrics.png")
    print("- model_comparison_summary.png")
    print("- info_gain_over_turns_*.png")
    print("- info_gain_comparison_*.png")

if __name__ == "__main__":
    main() 