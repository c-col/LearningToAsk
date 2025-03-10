from typing import List, Dict, Optional
from tqdm import tqdm
import argparse
import json
from pathlib import Path
import time
from datetime import datetime


from utils import check_if_question_names_entities, generate_json_from_question_entities
from model_client import ModelClient
from game_utils import GameState, GameConfig
from guesser_llm import guesser_prompt_fn, cot_guesser_prompt_fn, cot_guesser_initial_prompt_fn, guesser_asks, cot_guesser_asks
from judge_llm import judge_prompt_fn, judge_feedback


def get_results_dir(base_output_dir: Path, dataset_path: Path, model_name: str, debug: bool=False, debug_dataset: bool=False) -> Path:
    """Generate results directory path based on dataset and model names.
    
    Args:
        dataset_path: Path to the dataset file
        model_name: Name of the guesser model
        debug: Whether running in debug mode
        debug_dataset: Whether running in debug dataset mode
        
    Returns:
        Path object for the results directory
    """
    # Extract dataset name from path (remove .json and get final component)
    if debug and not debug_dataset:
        dataset_name = "debug"
    else:
        dataset_name = dataset_path.stem
        if debug_dataset:
            dataset_name += "_debug"
    
    # Clean up model name to be filesystem friendly
    clean_model_name = model_name.lower().split("/")[-1]
    
    # Construct results directory name
    return base_output_dir / f"results__{dataset_name}__{clean_model_name}"


def load_game_dataset(dataset_path: Path) -> Dict[str, Dict[str, List[str]]]:
    """Load game dataset from a JSON file.
    
    Args:
        dataset_path: Path to the JSON dataset file
        
    Returns:
        Dictionary mapping indices to game dictionaries containing 'entities' and 'target'
    """
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
        
    return json.loads(dataset_path.read_text())


def play_game(game_state: GameState, guesser_client: ModelClient, judge_client: ModelClient, config: GameConfig) -> Dict:
    """Play a game of 20 questions.
    
    Args:
        game_state: Initialized game state with target and entities
        guesser_client: ModelClient instance for generating questions
        judge_client: ModelClient instance for judging answers
        config: Game configuration settings
        
    Returns:
        Dictionary containing game history and results
    """
    # Initialize conversation based on guesser type
    if config.guesser_type == "r1":
        guesser_conversation = [
            dict(role='user', content=guesser_prompt_fn(game_state))
        ]
    else:
        guesser_conversation = [
            dict(role='user', content=cot_guesser_initial_prompt_fn(game_state))
        ]

    turn_history = []
    won_on_turn_number = None
    game_over = False
    max_turns = 3 if config.debug else 20
    
    for turn_number in range(max_turns):
        # Debug turn information
        print(f"\n=== Turn {turn_number + 1} ===")
        print(f"Conversation length: {len(guesser_conversation)}")
        print(f"Last message: {guesser_conversation[-1]['content']}")
        print(f"===========================================")
        
        # Get question from guesser
        if config.guesser_type == "r1":
            guesser_output, question = guesser_asks(
                guesser_client,
                conversation=guesser_conversation,
                config=config,   
            )
        else:
            guesser_output, question = cot_guesser_asks(
                guesser_client,
                conversation=guesser_conversation,
                game_state=game_state,
                config=config,
            )
        
        # Check if the question directly names any entities
        entities_in_question = check_if_question_names_entities(game_state.candidate_entities, question)
        
        if len(entities_in_question):
            # Handle direct entity guesses
            with tqdm(desc=f"[Turn {turn_number + 1}] Judging skipped") as pbar:
                judge_response = generate_json_from_question_entities(game_state.candidate_entities, entities_in_question)
                if len(entities_in_question) == 1 and judge_response[game_state.target] == "yes":
                    game_over = True
                    won_on_turn_number = turn_number + 1
                pbar.update(1)
        else:
            # Get feedback from judge for regular questions
            judge_response = judge_feedback(
                judge_client,
                game_state.candidate_entities,
                question,
                config
            )
        
        # Update game state
        game_state.update_state(question, judge_response)
        print(game_state)
        
        # Record turn history
        turn_history.append({
            "turn_number": game_state.turn_number,
            "guesser_output": guesser_output,
            "question": question,
            "judge_response": judge_response,
            "remaining_entities": game_state.remaining_entities.copy(),
            "information_gain": game_state.information_gain,
            "ideal_information_gain": game_state.ideal_information_gain
        })
        
        # Update conversation
        guesser_conversation.append(dict(role='assistant', content=question))
        if won_on_turn_number is not None:
            print(f"Game over! Won on turn {won_on_turn_number}")
            break
        else:
            judge_response_text = judge_response[game_state.target].capitalize() + ". What is your next question?"
            guesser_conversation.append(dict(role='user', content=judge_response_text))
    
    return {
        "target": game_state.target,
        "candidate_entities": game_state.candidate_entities,
        "turn_history": turn_history,
        "won_on_turn": won_on_turn_number,
        "game_over": game_over,
        "final_entities": game_state.remaining_entities,
        "number_of_turns": len(turn_history)
    }


def save_checkpoint(checkpoint_dir: Path, game_idx: str, game_result: Dict) -> None:
    """Save individual game result as a checkpoint.
    
    Args:
        checkpoint_dir: Directory to save checkpoints in
        game_idx: Index of the game
        game_result: Result of the game
    """
    checkpoint_file = checkpoint_dir / f"game_{game_idx}.json"
    checkpoint_file.write_text(json.dumps(game_result, indent=2))


def load_checkpoints(checkpoint_dir: Path) -> Dict[str, Dict]:
    """Load all checkpointed game results.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        
    Returns:
        Dictionary mapping game indices to their results
    """
    results = {}
    if checkpoint_dir.exists():
        for checkpoint_file in checkpoint_dir.glob("game_*.json"):
            game_idx = checkpoint_file.stem.split('_')[1]  # Extract index from filename
            results[game_idx] = json.loads(checkpoint_file.read_text())
    return results


def save_results(results_file: Path, config: GameConfig, game_results: Dict) -> None:
    """Save game results and config to a JSON file.
    
    Args:
        results_file: Path to save results to
        config: Game configuration
        game_results: Dictionary of game results
    """
    output = {
        "config": {
            "guesser_model": config.guesser_model,
            "guesser_type": config.guesser_type,
            "judge_model": config.judge_model,
            "guesser_private_endpoint": config.guesser_private_endpoint,
            "judge_private_endpoint": config.judge_private_endpoint,
            "use_random_seed": config.use_random_seed,
            "seed": config.seed,
            "guesser_think_budget": config.guesser_think_budget,
            "guesser_answer_budget": config.guesser_answer_budget,
            "judge_token_budget": config.judge_token_budget,
            "debug": config.debug,
            "debug_dataset": config.debug_dataset,
            "dataset_path": config.dataset_path,
            "results_dir": config.results_dir
        },
        "results": game_results
    }
    results_file.write_text(json.dumps(output, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the 20 questions game with language models')
    
    # Model configuration
    parser.add_argument('-g', '--guesser-model', type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
                      help='Model to use for generating questions')
    parser.add_argument('-gt', '--guesser-type', type=str, default="r1", choices=["r1", "cot"],
                      help='Type of guesser to use (r1: r1 reasoning, cot: chain of thought)')
    parser.add_argument('-gpe', '--guesser-private-endpoint', action='store_true',
                      help='Use private endpoint for guesser model')
    parser.add_argument('-j', '--judge-model', type=str, default="deepseek-ai/DeepSeek-R1",
                      help='Model to use for judging answers')
    parser.add_argument('-jpe', '--judge-private-endpoint', action='store_true',
                      help='Use private endpoint for judge model')
    parser.add_argument('--token-path', type=str, help='Path to HuggingFace token file', 
                      default=str(Path.home() / "Desktop/PhD/hf_token.txt"))
    
    # Dataset configuration
    parser.add_argument('--dataset-path', type=str, default="../data/game_sets/test/contrast_sets_8.json",
                      help='Path to the game dataset JSON file')
    parser.add_argument('--output-dir', type=str, default="../data/game_sets/test/outputs",
                      help='Path to the base output directory')
    
    # Debug configuration
    parser.add_argument('--debug', '-d', action='store_true',
                      help='Run in debug mode with reduced turns (3 instead of 20)')
    parser.add_argument('--debug-dataset', '-dd', action='store_true',
                      help='Run in debug mode with reduced turns (3 instead of 20) on a dataset (first 3 games)')
    
    # Generation settings
    parser.add_argument('--seed', type=int, help='Random seed for generation')
    parser.add_argument('--no-random-seed', action='store_true', help='Disable random seed generation')
    parser.add_argument('--guesser-think-budget', type=int, default=1000,
                      help='Token budget for guesser thinking phase')
    parser.add_argument('--guesser-answer-budget', type=int, default=200,
                      help='Token budget for guesser answer phase')
    parser.add_argument('--judge-token-budget', type=int, default=2000,
                      help='Token budget for judge responses')
    
    args = parser.parse_args()

    # Create results and checkpoint directories
    dataset_path = "" if (args.debug and not args.debug_dataset) else Path(args.dataset_path) 
    results_dir = get_results_dir(Path(args.output_dir), dataset_path, args.guesser_model, args.debug, args.debug_dataset)
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = results_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    if args.debug:
        print("\n=== Running in DEBUG mode with 3 turns ===\n")

    config = GameConfig(
        hf_token_path=args.token_path,
        guesser_model=args.guesser_model,
        guesser_type=args.guesser_type,
        guesser_private_endpoint=args.guesser_private_endpoint,
        judge_model=args.judge_model,
        judge_private_endpoint=args.judge_private_endpoint,
        use_random_seed=not args.no_random_seed,
        seed=args.seed,
        guesser_think_budget=args.guesser_think_budget,
        guesser_answer_budget=args.guesser_answer_budget,
        judge_token_budget=args.judge_token_budget,
        debug=args.debug,
        debug_dataset=args.debug_dataset,
        dataset_path=str(dataset_path),
        results_dir=str(results_dir),
        checkpoint_dir=str(checkpoint_dir),
    )

    # Initialize model clients
    guesser_client = ModelClient(
        model_name=config.guesser_model, 
        private_endpoint=config.guesser_private_endpoint, 
        guesser_type=config.guesser_type, 
        hf_token_path=config.hf_token_path
    )
    judge_client = ModelClient(
        model_name=config.judge_model, 
        private_endpoint=config.judge_private_endpoint, 
        hf_token_path=config.hf_token_path
    )

    all_game_results = load_checkpoints(checkpoint_dir)
    
    if config.debug:
        if config.debug_dataset:
            # Run first 3 games from dataset in debug mode
            print("\nRunning first 3 games from dataset in debug mode...")
            games = load_game_dataset(Path(args.dataset_path))
            # Take first 3 games only
            debug_games = dict(list(games.items())[:3])
            
            for game_idx, game in debug_games.items():
                # Skip games that have already been completed
                if game_idx in all_game_results:
                    print(f"\nSkipping completed game {game_idx}")
                    continue
                    
                print(f"\nPlaying game {game_idx}")
                print(f"Target: {game['target']}")
                print(f"Entities: {', '.join(game['items'])}")
                
                game_state = GameState(target=game['target'], candidate_entities=game['items'])
                game_result = play_game(game_state, guesser_client, judge_client, config)
                all_game_results[game_idx] = game_result
                
                # Save checkpoint for this game
                save_checkpoint(checkpoint_dir, game_idx, game_result)
                
                # Add small delay between games to avoid rate limits
                time.sleep(1)
        else:
            # Run single test game in debug mode
            print("\nRunning test game in debug mode...")
            # Test game setup for debugging
            test_game_entities = [
                "parakeet", "seal", "goose", "stork", "deer", "finch", "seagull", "dog", "dolphin", "cow",
                "budgie", "otter", "rooster", "canary", "walrus", "sheep"
            ]
            test_answer = "budgie"
            game_state = GameState(target=test_answer, candidate_entities=test_game_entities)
            game_result = play_game(game_state, guesser_client, judge_client, config)
            all_game_results["0"] = game_result
            save_checkpoint(checkpoint_dir, "0", game_result)
    else:
        # Load and run games from dataset
        print(f"\nLoading first 10 games from {dataset_path}...")
        games = load_game_dataset(dataset_path)
        games = dict(list(games.items())[:10])
        
        for game_idx, game in games.items():
            # Skip games that have already been completed
            if game_idx in all_game_results:
                print(f"\nSkipping completed game {game_idx}")
                continue
                
            print(f"\nPlaying game {game_idx}")
            print(f"Target: {game['target']}")
            print(f"Entities: {', '.join(game['items'])}")
            
            game_state = GameState(target=game['target'], candidate_entities=game['items'])
            game_result = play_game(game_state, guesser_client, judge_client, config)
            all_game_results[game_idx] = game_result
            
            # Save checkpoint for this game
            save_checkpoint(checkpoint_dir, game_idx, game_result)
            
            # Add small delay between games to avoid rate limits
            time.sleep(1)
    
    # Save final results
    results_file = results_dir / "game_results.json"
    save_results(results_file, config, all_game_results)
    
    print(f"\nResults saved to {results_file}")

