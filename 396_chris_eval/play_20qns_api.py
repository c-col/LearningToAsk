from huggingface_hub import InferenceClient, get_inference_endpoint
from typing import List, Optional, Iterator, Dict, Any
import json
from tqdm import tqdm
from random import randint, shuffle
import re
from dataclasses import dataclass
from transformers import AutoTokenizer
from time import sleep

from utils import (
    load_hf_token, 
    extract_question_from_generation,
    extract_question_and_clean,
    check_if_question_names_entities,
    generate_json_from_question_entities,
    find_whole_word,
    compute_information_gain,
)


@dataclass
class GameConfig:
    """Configuration for the 20 questions game."""
    # Model settings
    hf_token_path: Optional[str] = None
    guesser_model: str = None
    judge_model: str = None
    
    guesser_private_endpoint: bool = False
    judge_private_endpoint: bool = False

    # Generation settings
    use_random_seed: bool = True
    seed: Optional[int] = None
    guesser_think_budget: int = 1000
    guesser_answer_budget: int = 500
    judge_token_budget: int = 1000

    def __post_init__(self):
        if self.use_random_seed and self.seed is None:
            self.seed = randint(0, 1000)
        elif not self.use_random_seed and self.seed is None:
            self.seed = 1


@dataclass
class GameState:
    """Tracks the state and metrics of a 20 questions game."""
    target: str
    turn_number: int = 0
    remaining_entities: List[str] = None
    previous_entities: Optional[List[str]] = None
    current_question: Optional[str] = None
    judege_response: Optional[Dict[str, str]] = None
    information_gain: Optional[float] = None
    ideal_information_gain: Optional[float] = None

    def __post_init__(self):
        if self.remaining_entities is None:
            self.remaining_entities = []

    def update_state(self, question: str, judge_response: Dict[str, str]):
        """Update game state based on the latest turn."""
        # Store previous count for information gain calculation
        prev_entity_count = len(self.remaining_entities)
        assert prev_entity_count > 0, "Previous entity count must be greater than 0"
        self.previous_entities = self.remaining_entities.copy()
        self.turn_number += 1
        self.guesser_question = question
        self.judge_response = judge_response
        
        # Update remaining entities based on target's answer
        target_answer = judge_response[self.target]
        self.remaining_entities = [
            entity for entity, answer in judge_response.items()
            if entity in self.previous_entities and answer in [target_answer, "sometimes", "unknown"]
        ]
        
        # Compute actual and ideal information gain
        self.information_gain = compute_information_gain(
            total_entities=prev_entity_count,
            remaining_count=len(self.remaining_entities)
        )
        self.ideal_information_gain = compute_information_gain(prev_entity_count, prev_entity_count // 2)

    def __str__(self) -> str:
        """Pretty print the game state."""
        return (
            f"\n=== Game State Turn {self.turn_number} ===\n"
            f"Target: {self.target}\n"
            f"Previous Entities (len: {len(self.previous_entities)}): {', '.join(self.previous_entities)}\n"
            f"Guesser Question: {self.guesser_question}\n"
            f"Judge Answer: {self.judge_response}\n"
            f"Remaining Entities (len: {len(self.remaining_entities)}): {', '.join(self.remaining_entities)}\n"
            f"Information Gain: {self.information_gain:.2f} bits (ideal: {self.ideal_information_gain:.2f} bits)\n"
        )


class ModelClient:
    def __init__(self, model_name: str, private_endpoint: bool, hf_token_path: Optional[str] = None):
        self.model_name = model_name
        self.model_role = "judge" if model_name == "deepseek-ai/DeepSeek-R1" else "guesser"
        token = load_hf_token(hf_token_path) if hf_token_path else load_hf_token()
        
        if private_endpoint:
            # For private endpoints, get the specific endpoint for the model
            endpoint = get_inference_endpoint(model_name, token=token)
            self.client = endpoint.client
        else:
            # For public API endpoints
            self.client = InferenceClient(
                model=model_name,
                provider="sambanova" if model_name== "deepseek-ai/DeepSeek-R1" else "hf-inference",
                api_key=token
            )
        
        # Initialize tokenizer for guesser
        if self.model_role == "guesser":
            tokenizer_dict = {
                "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
                "deepseek-ai/DeepSeek-R1": "deepseek-ai/DeepSeek-R1",
                "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B-Instruct": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B-Instruct",
                "deepseek-r1-distill-qwen-7b-mka": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
            }
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dict[model_name])
    
    def generate(self, 
                prompt: str = None,
                max_new_tokens: int = None, 
                stream: bool = True,
                temperature: float = 0.7,
                # response_format: Optional[Dict[str, Any]] = None,
                seed: Optional[int] = None,
                stop: Optional[List[str]] = None,
                messages: Optional[List[Dict[str, str]]] = None) -> Iterator[str]:
        """Interface for HF API inference."""
        if self.model_role == "guesser":
            if not prompt:
                raise ValueError("Guesser model requires a prompt")
            # Use text generation with formatted prompt
            for token in self.client.text_generation(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                stream=stream,
                seed=seed,
                stop=stop
            ):
                yield token
        elif self.model_role == "judge":
            if not messages:
                raise ValueError("Judge model requires messages")
            # Use chat completions for judge
            for token in self.client.chat.completions.create(
                messages=messages,
                max_tokens=max_new_tokens,
                temperature=temperature,
                stream=stream,
                stop=stop
            ):
                yield token.choices[0].delta.content


# guesser prompt / conversation
def guesser_prompt_fn(entity_list: List[str]) -> str:
    entities_string = ", ".join(entity_list)
    return (f"Let's play 20 questions. I'm thinking of one of these items: {entities_string}. "
            "You are the guesser and should ask strategic yes/no questions to identify the mystery item. "
            "For each question you ask:\n"
            "1. First think about what information would be most useful to narrow down the possibilities\n"
            "2. Then formulate a clear yes/no question\n"
            "3. Finally write your question inside \\boxed{}. For example: \"\\boxed{Is it a living thing?}\"\n\n"
            "I will respond with one of four answers:\n"
            # "- \"yes\" (always true)\n"
            # "- \"no\" (never true)\n"
            # "- \"sometimes\" (true in some cases)\n"
            # "- \"unknown\" (information not available)\n\n"
            "- \"yes\"\n"
            "- \"no\"\n"
            "- \"sometimes\"\n"
            "- \"unknown\"\n\n"
            "Ask your first question to start identifying the mystery item.")


end_think_token = "\n</think>\n\nMy question for this turn: \\boxed{"


# judge prompt builder
def judge_prompt_fn(active_entities, question):
    if not any([question.endswith(symbol) for symbol in ["?", "."]]):
        question += "?"

    active_entities_string = ", ".join(active_entities)

    prompt = (
        f"Go through the following list and answer the question \"{question}\" for each item.\n"
        f"For each item, respond with EXACTLY one of these values: yes, no, sometimes, or unknown.\n"
        f"List: {active_entities_string}\n\n"
        # f"Format your response as a numbered list with one item per line, like this:\n"
        # f"1. item1: yes\n"
        # f"2. item2: no\n"
        # f"3. item3: sometimes\n"
        # f"4. item4: unknown\n\n"
        f"Format your response as a numbered list with one item per line. Here is an example:\n"
        f"1. cat: yes        ✓ (correct format)\n"
        f"2. dog: no         ✓ (correct format)\n"
        f"3. fish: sometimes ✓ (correct format)\n"
        f"4. bird: unknown   ✓ (correct format)\n\n"
        f"Do NOT use any other values like these:\n"
        f"5. snake: probably      ✗ (wrong - use only yes/no/sometimes/unknown)\n"
        f"6. lion: mostly yes     ✗ (wrong - use only yes/no/sometimes/unknown)\n"
        f"7. tiger: not sure      ✗ (wrong - use 'unknown' instead)\n"
        f"8. bear: occasionally   ✗ (wrong - use 'sometimes' instead)\n\n"
        f"Be consistent and precise in your judgments. Consider the literal meaning of the question."
    )
    
    return [{"role": "user", "content": prompt}]


def guesser_asks(guesser_client: ModelClient, conversation: List[Dict[str, str]], config: GameConfig) -> str:
    """Generate a question from the guesser model.
    
    Args:
        guesser_client: The model client for the guesser
        conversation: Full conversation history
        config: Game configuration
        
    Returns:
        The extracted question from the guesser's output
    """
    guesser_output = ""
    is_thinking_over = False

    # Format the conversation using the chat template
    formatted_prompt = guesser_client.tokenizer.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=False
    )

    # Generate guesser's thinking with retry logic
    generation_success = False
    while not generation_success:
        try:
            for token in tqdm(guesser_client.generate(
                prompt=formatted_prompt,
                max_new_tokens=config.guesser_think_budget + 500,
                stream=True,
                seed=config.seed
            ), total=config.guesser_think_budget + 500, desc="Guessing..."):
                if len(guesser_output) + 1 >= config.guesser_think_budget:
                    if "\n\n" in token:
                        guesser_output += token.replace("\n\n", "")
                        break
                if token == "</think>":
                    is_thinking_over = True
                    print(f"thinking over: {guesser_output}")
                    break
                guesser_output += token
            generation_success = True
        except Exception as e:
            print(f"Exception during thinking: {e}")
            print("\t...Sleeping for 15 seconds...")
            sleep(15)
            continue

    if is_thinking_over:
        guesser_output += end_think_token[1:]
    else:
        guesser_output += end_think_token

    formatted_prompt_with_thinking = formatted_prompt + guesser_output
    
    # Generate guesser's answer with retry logic
    generation_success = False
    while not generation_success:
        try:
            for token in tqdm(guesser_client.generate(
                prompt=formatted_prompt_with_thinking,
                max_new_tokens=config.guesser_answer_budget,
                stream=True,
                seed=config.seed,
                stop=["}"]
            ), desc="Getting answer..."):
                guesser_output += token
                if "}" in token:
                    break
            generation_success = True
        except Exception as e:
            print(f"Exception during answer generation: {e}")
            print("\t...Sleeping for 15 seconds...")
            sleep(15)
            continue

    # Extract and clean the question from the output
    question = extract_question_and_clean(guesser_output)
    # question = extract_question_from_generation(guesser_output)
    
    # Debug output
    print(f"\n------[[guesser output]]------")
    print(guesser_output)
    
    return guesser_output, question


def judge_feedback(judge_client: ModelClient, game_entities: List[str], question: str, config: GameConfig) -> Dict[str, str]:
    """Generate feedback from the judge model for a given question.
    
    Args:
        judge_client: The model client for the judge
        game_entities: List of entities in the game
        question: The question to judge
        config: Game configuration
        
    Returns:
        Dictionary mapping entities to yes/no/sometimes/unknown responses
    """
    judge_prompt = judge_prompt_fn(game_entities, question)
    judge_response = ""

    # Generate judge's response
    with tqdm(desc="Judging...") as pbar:
        for token in judge_client.generate(
            messages=judge_prompt,
            max_new_tokens=config.judge_token_budget,
            stream=True
        ):
            judge_response += token
            pbar.update(1)

    # Debug the raw response
    print("\nRaw judge response:", repr(judge_response))

    # Parse the response into a dictionary
    response_dict = {}
    
    # Process each line of the response
    for line in judge_response.split('\n'):
        line = line.strip().lower()
        # Skip empty lines
        if not line:
            continue
            
        # Remove any numbering at the start
        if '. ' in line:
            line = line.split('. ', 1)[1]
            
        if ':' in line:
            # Standard format with colon separator
            entity, value = line.split(':', 1)
            entity = entity.strip()
            value = value.strip()
            
            # Only process if the entity is in our game entities
            if entity in game_entities:
                # First try exact matching
                if value in ['yes', 'no', 'sometimes', 'unknown']:
                    response_dict[entity] = value
                else:
                    # Try to find yes/no using fuzzy matching
                    value_matches = check_if_question_names_entities(['yes', 'no'], value)
                    if len(value_matches) == 1:
                        response_dict[entity] = value_matches[0]
                    else:
                        # Check for sometimes/unknown using fuzzy matching
                        other_matches = check_if_question_names_entities(['sometimes', 'unknown'], value)
                        if len(other_matches) == 1:
                            response_dict[entity] = other_matches[0]
                        else:
                            # If no clear match found, default to unknown
                            print(f"\nWarning: Invalid value '{value}' for {entity}")
                            response_dict[entity] = "unknown"
        else:
            # No colon in line - try to find both entity and value using fuzzy matching
            entity_matches = check_if_question_names_entities(game_entities, line)
            if len(entity_matches) == 1:
                entity = entity_matches[0]
                # Try to find the value in the same line
                value_matches = check_if_question_names_entities(['yes', 'no'], line)
                if len(value_matches) == 1:
                    response_dict[entity] = value_matches[0]
                else:
                    other_matches = check_if_question_names_entities(['sometimes', 'unknown'], line)
                    if len(other_matches) == 1:
                        response_dict[entity] = other_matches[0]
                    else:
                        print(f"\nWarning: Could not find clear value in line '{line}' for {entity}")
                        response_dict[entity] = "unknown"

    # Ensure all entities have a response
    for entity in game_entities:
        if entity not in response_dict:
            print(f"\nWarning: Missing response for {entity}")
            response_dict[entity] = "unknown"

    # Debug output
    print(f"\n\t-------[[judge response]]---------\n{response_dict}")
    
    return response_dict


def play_game(game_entities: List[str], game_target: str, guesser_client: ModelClient, judge_client: ModelClient, config: GameConfig):
    """Play a game of 20 questions.
    
    Args:
        game_entities: List of possible entities to guess from
        game_target: The correct answer from the entities list
        guesser_client: ModelClient instance for generating questions
        judge_client: ModelClient instance for judging answers
        config: Game configuration settings
    """
    if config.use_random_seed:
        shuffle(game_entities)
    
    # Initialize game state
    game_state = GameState(target=game_target, remaining_entities=game_entities.copy())
    
    # Initialize conversation history with system message
    guesser_conversation = [
        dict(role='user', content=guesser_prompt_fn(game_entities))
    ]

    won_on_turn_number = None
    turn_history = []
    game_over = False
    
    for turn_number in range(20):
        # Debug turn information
        print(f"\n===========================================")
        print(f"turn {turn_number + 1} input:")
        print(f"Conversation length: {len(guesser_conversation)}")
        print(f"Last message: {guesser_conversation[-1]['content']}")
        print(f"===========================================")

        # Get question from guesser using full conversation history
        guesser_output, question = guesser_asks(
            guesser_client, 
            conversation=guesser_conversation,
            config=config   
        )
        
        # Check if the question directly names any entities
        entities_in_question = check_if_question_names_entities(game_entities, question)
        
        if len(entities_in_question):
            # Handle direct entity guesses
            with tqdm(desc=f"[Turn {turn_number + 1}] Judging skipped") as pbar:
                judge_response = generate_json_from_question_entities(game_entities, entities_in_question)
                if len(entities_in_question) == 1 and judge_response[game_target] == "yes":
                    game_over = True
                    won_on_turn_number = turn_number + 1
                pbar.update(1)
        else:
            # Get feedback from judge for regular questions
            judge_response = judge_feedback(
                judge_client,
                game_entities,
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
        judge_response_text = judge_response[game_target].capitalize() + ". What is your next question?"
        guesser_conversation.append(dict(role='user', content=judge_response_text))

    return turn_history


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run the 20 questions game with language models')
    
    # Model configuration
    parser.add_argument('-g', '--guesser-model', type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
                      help='Model to use for generating questions')
    parser.add_argument('-gpe', '--guesser-private-endpoint', action='store_true',
                      help='Use private endpoint for guesser model')
    parser.add_argument('-j', '--judge-model', type=str, default="deepseek-ai/DeepSeek-R1",
                      help='Model to use for judging answers')
    parser.add_argument('-jpe', '--judge-private-endpoint', action='store_true',
                      help='Use private endpoint for judge model')
    parser.add_argument('--token-path', type=str, help='Path to HuggingFace token file', default="/Users/benigerisimon/Desktop/PhD/hf_token.txt")
    
    # Generation settings
    parser.add_argument('--seed', type=int, help='Random seed for generation')
    parser.add_argument('--no-random-seed', action='store_true', help='Disable random seed generation')
    parser.add_argument('--guesser-think-budget', type=int, default=1000, #1000
                      help='Token budget for guesser thinking phase')
    parser.add_argument('--guesser-answer-budget', type=int, default=200, #500
                      help='Token budget for guesser answer phase')
    parser.add_argument('--judge-token-budget', type=int, default=2000, #1000
                      help='Token budget for judge responses')
    
    args = parser.parse_args()

    # Create game configuration
    config = GameConfig(
        guesser_model=args.guesser_model,
        guesser_private_endpoint=args.guesser_private_endpoint,
        judge_model=args.judge_model,
        judge_private_endpoint=args.judge_private_endpoint,
        hf_token_path=args.token_path,
        use_random_seed=not args.no_random_seed,
        seed=args.seed,
        guesser_think_budget=args.guesser_think_budget,
        guesser_answer_budget=args.guesser_answer_budget,
        judge_token_budget=args.judge_token_budget,
    )

    # Initialize model clients
    guesser_client = ModelClient(config.guesser_model, config.guesser_private_endpoint, hf_token_path=config.hf_token_path)
    judge_client = ModelClient(config.judge_model, config.judge_private_endpoint, hf_token_path=config.hf_token_path)

    test_game_entities = [
        "parakeet", "seal", "goose", "stork", "deer", "finch", "seagull", "dog", "dolphin", "cow",
        "budgie", "otter", "rooster", "canary", "walrus", "sheep"
    ]
    test_answer = "budgie"

    play_game(test_game_entities, test_answer, guesser_client, judge_client, config)

