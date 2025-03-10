import json
from pathlib import Path
import sys

def clean_results(file_path: str) -> None:
    """Clean results JSON file according to specified rules.
    
    Args:
        file_path: Path to results JSON file
    """
    # Load results
    results_path = Path(file_path)
    if not results_path.exists():
        print(f"Error: File {file_path} not found")
        return
        
    data = json.loads(results_path.read_text())
    
    # Clean game results
    game_results = data["results"]
    
    # Rule 1: Check number of games and fix indices
    if len(game_results) == 10:
        # Convert all keys to integers for proper sorting
        game_indices = sorted([int(idx) for idx in game_results.keys()])
        
        # If indices are 1-10, shift to 0-9
        if all(i+1 in game_indices for i in range(10)):
            new_results = {}
            for i in range(10):
                new_results[str(i)] = game_results[str(i+1)]
            game_results = new_results
        # If indices are already 0-9, just ensure they're properly ordered
        elif all(i in game_indices for i in range(10)):
            new_results = {}
            for i in range(10):
                new_results[str(i)] = game_results[str(i)]
            game_results = new_results
        else:
            print(f"Warning: Unexpected game indices: {sorted(game_results.keys())}")
            return
    else:
        print(f"Error: Number of games is not 10. Number of games: {len(game_results)}")
        return
    
    # Process each game
    for game_idx, game in game_results.items():
        # Rule 2: Add number_of_turns if missing
        if "number_of_turns" not in game:
            game["number_of_turns"] = len(game["turn_history"])
            
        # Process turn history
        for turn_idx, turn in enumerate(game["turn_history"]):
            # Rule 3: Fix turn numbers that are off by 1
            if "turn_number" in turn and turn["turn_number"] == turn_idx + 2:
                turn["turn_number"] = turn_idx + 1
            
            # Rule 4: Add missing turn numbers
            if "turn_number" not in turn:
                turn["turn_number"] = turn_idx + 1
                
            # Ensure turn_number is first key by creating new ordered dict
            new_turn = {
                "turn_number": turn["turn_number"]
            }
            # Add remaining keys in original order
            for key in turn:
                if key != "turn_number":
                    new_turn[key] = turn[key]
            game["turn_history"][turn_idx] = new_turn
        
        # Rule 5: Fix won_on_turn if it doesn't match turn history length
        if game.get("won_on_turn") is not None:
            # Always set won_on_turn to match turn history length
            game["won_on_turn"] = len(game["turn_history"])
        else:
            game["won_on_turn"] = None
            
        # Ensure consistent key order for each game
        ordered_game = {
            "target": game["target"],
            "candidate_entities": game["candidate_entities"],
            "turn_history": game["turn_history"],
            "won_on_turn": game["won_on_turn"],
            "game_over": game["game_over"],
            "final_entities": game["final_entities"],
            "number_of_turns": game["number_of_turns"]
        }
        game_results[game_idx] = ordered_game
    
    # Update results
    data["results"] = game_results
    
    # Save cleaned results
    output_path = results_path.parent / f"{results_path.stem}_clean{results_path.suffix}"
    output_path.write_text(json.dumps(data, indent=2))
    print(f"Cleaned results saved to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python clean_results.py <results_json_file>")
        sys.exit(1)
    
    clean_results(sys.argv[1]) 

    # python clean_results.py ../data/game_sets/test/outputs/results__contrast_sets_8__deepseek-r1-distill-qwen-7b-mka/game_results.json
    # python clean_results.py ../data/game_sets/test/outputs/results__contrast_sets_8__deepseek-r1-distill-qwen-14b-znu/game_results.json
    # python clean_results.py ../data/game_sets/test/outputs/results__contrast_sets_8__deepseek-r1-distill-qwen-32b-ldr/game_results.json
    # python clean_results.py ../data/game_sets/test/outputs/results__contrast_sets_16__deepseek-r1-distill-qwen-7b-mka/game_results.json