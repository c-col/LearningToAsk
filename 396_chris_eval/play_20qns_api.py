from typing import List, Dict
from tqdm import tqdm
from random import shuffle
import argparse

from utils import (
    check_if_question_names_entities,
    generate_json_from_question_entities,
)
from model_client import ModelClient
from game_utils import GameState, GameConfig
from guesser_llm import guesser_prompt_fn, cot_prompt_fn, guesser_asks, cot_guesser_asks
from judge_llm import judge_prompt_fn, judge_feedback


def play_game(game_state: GameState, guesser_client: ModelClient, judge_client: ModelClient, config: GameConfig):
    """Play a game of 20 questions.
    
    Args:
        game_state: Initialized game state with target and entities
        guesser_client: ModelClient instance for generating questions
        judge_client: ModelClient instance for judging answers
        config: Game configuration settings
    """
    # Initialize conversation history with system message
    guesser_conversation = [
        dict(role='user', content=guesser_prompt_fn(game_state.candidate_entities))
    ]

    won_on_turn_number = None
    turn_history = []
    game_over = False
    
    max_turns = 3 if config.debug else 20
    
    for turn_number in range(max_turns):
        if config.debug:
            print(f"\n=== DEBUG MODE: Turn {turn_number + 1}/{max_turns} ===")
            
        # Debug turn information
        print(f"\n===========================================")
        print(f"turn {turn_number + 1} input:")
        print(f"Conversation length: {len(guesser_conversation)}")
        print(f"Last message: {guesser_conversation[-1]['content']}")
        print(f"===========================================")

        # Get question from guesser using appropriate method
        if config.guesser_type == "r1":
            guesser_output, question = guesser_asks(
                guesser_client, 
                conversation=guesser_conversation,
                config=config   
            )
        elif config.guesser_type == "cot":
            guesser_output, question = cot_guesser_asks(
                guesser_client,
                entity_list=game_state.candidate_entities,
                game_state=game_state,
                config=config,
                provide_remaining_entities=False
            )
        else:
            raise ValueError(f"Unknown guesser type: {config.guesser_type}. Must be 'r1' or 'cot'.")
        
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
        print(game_state)  # Print current game state
        
        # Record turn history
        turn_history.append({
            "guesser": guesser_output, 
            "question": question, 
            "json": judge_response,
            "game_state": game_state.__dict__.copy(),
            "won": game_over
        })
        
        if game_over:
            print(f"\nGame Over! Won in {won_on_turn_number} turns!")
            break
            
        # Update conversation for next turn
        guesser_conversation.append(dict(role='assistant', content=question))
        judge_response_text = judge_response[game_state.target].capitalize() + ". What is your next question?"
        guesser_conversation.append(dict(role='user', content=judge_response_text))

    return turn_history


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
    parser.add_argument('--token-path', type=str, help='Path to HuggingFace token file', default="/Users/benigerisimon/Desktop/PhD/hf_token.txt")
    
    # Debug configuration
    parser.add_argument('--debug', '-d', action='store_true',
                      help='Run in debug mode with reduced turns (3 instead of 20)')
    
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

    if args.debug:
        print("\n=== Running in DEBUG mode with 3 turns ===\n")

    # Create game configuration
    config = GameConfig(
        guesser_model=args.guesser_model,
        guesser_type=args.guesser_type,
        guesser_private_endpoint=args.guesser_private_endpoint,
        judge_model=args.judge_model,
        judge_private_endpoint=args.judge_private_endpoint,
        hf_token_path=args.token_path,
        use_random_seed=not args.no_random_seed,
        seed=args.seed,
        guesser_think_budget=args.guesser_think_budget,
        guesser_answer_budget=args.guesser_answer_budget,
        judge_token_budget=args.judge_token_budget,
        debug=args.debug
    )

    # Initialize model clients
    guesser_client = ModelClient(model_name=config.guesser_model, private_endpoint=config.guesser_private_endpoint, guesser_type=config.guesser_type, hf_token_path=config.hf_token_path)
    judge_client = ModelClient(model_name=config.judge_model, private_endpoint=config.judge_private_endpoint, hf_token_path=config.hf_token_path)

    # Test game setup
    test_game_entities = [
        "parakeet", "seal", "goose", "stork", "deer", "finch", "seagull", "dog", "dolphin", "cow",
        "budgie", "otter", "rooster", "canary", "walrus", "sheep"
    ]
    test_answer = "budgie"

    # Initialize game state
    if config.use_random_seed:
        shuffle(test_game_entities)
    game_state = GameState(target=test_answer, candidate_entities=test_game_entities.copy())

    # Play the game
    play_game(game_state=game_state, guesser_client=guesser_client, judge_client=judge_client, config=config)

