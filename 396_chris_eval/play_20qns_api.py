from huggingface_hub import InferenceClient
from typing import List, Optional, Iterator, Dict, Any
import json
from tqdm import tqdm
from random import randint, shuffle
import re
from dataclasses import dataclass

from utils import load_hf_token, extract_question_from_generation


@dataclass
class GameConfig:
    """Configuration for the 20 questions game."""
    # Model settings
    guesser_model: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    judge_model: str = "Qwen/Qwen2.5-3B-Instruct"
    hf_token_path: Optional[str] = None
    
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


class ModelClient:
    def __init__(self, model_name: str, hf_token_path: Optional[str] = None):
        self.model_name = model_name
        token = load_hf_token(hf_token_path) if hf_token_path else load_hf_token()
        self.client = InferenceClient(
            model=model_name,
            provider="hf-inference",
            api_key=token
        )
    
    def generate(self, 
                prompt: str, 
                max_new_tokens: int, 
                stream: bool = True,
                temperature: float = 0.7,
                seed: Optional[int] = None,
                stop: Optional[List[str]] = None,
                response_format: Optional[Dict[str, Any]] = None) -> Iterator[str]:
        """Interface for HF API inference."""
        if response_format:
            # For chat completions with response format (used by judge)
            for token in self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_new_tokens,
                response_format=response_format,
                stream=stream
            ):
                yield token.choices[0].delta.content
        else:
            # For regular text generation (used by guesser)
            for token in self.client.text_generation(
                prompt,
                max_new_tokens=max_new_tokens,
                stream=stream,
                seed=seed,
                stop=stop
            ):
                yield token


# guesser prompt / conversation
def guesser_prompt_fn(entity_list: List[str]) -> str:
    entities_string = ", ".join(entity_list)
    return (f"Let's play 20 questions. I'm thinking of one of these items: {entities_string}. "
            "You are the guesser and should ask strategic yes/no questions to identify the mystery item. "
            "For each question you ask:\n"
            "1. First think about what information would be most useful to narrow down the possibilities\n"
            "2. Then formulate a clear yes/no question\n"
            "3. Finally write your question inside \\boxed{}\n\n"
            "I will respond with one of four answers:\n"
            "- \"yes\" (always true)\n"
            "- \"no\" (never true)\n"
            "- \"sometimes\" (true in some cases)\n"
            "- \"unknown\" (information not available)\n\n"
            "Ask your first question to start identifying the mystery item.")


end_think_token = "\n</think>\n\nMy question for this turn: \\boxed{"


# judge prompt builder
def judge_prompt_fn(active_entities, question):
    if not any([question.endswith(symbol) for symbol in ["?", "."]]):
        question += "?"

    active_entities_string = ", ".join(active_entities)
    all_prompt = (
        f"You are the judge in a 20 questions game. For the question: \"{question}\"\n"
        f"Evaluate this question for each item in this list: {active_entities_string}\n\n"
        f"For each item, respond with exactly one of these values:\n"
        f"- \"yes\": if the answer is always true\n"
        f"- \"no\": if the answer is never true\n"
        f"- \"sometimes\": if the answer can be true in some cases\n"
        f"- \"unknown\": if you cannot determine the answer\n\n"
        f"Respond with ONLY a JSON object mapping each item to its answer. Example format:\n"
        f"{{'item1': 'yes', 'item2': 'no'}}\n\n"
        f"Be consistent and precise in your judgments. Consider the literal meaning of the question."
    )

    response_format = {
        "type": "json",
        "value": {
            "properties": {k: {"type": "string", "enum": ["yes", "no", "sometimes", "unknown"]} for k in active_entities},
            "required": active_entities,
        },
    }
    messages = [
        {
            "role": "user",
            "content": all_prompt,
        }
    ]
    return messages, response_format


def guesser_asks(guesser_client: ModelClient, prompt: str, config: GameConfig) -> str:
    """Generate a question from the guesser model.
    
    Args:
        guesser_client: The model client for the guesser
        prompt: The input prompt for the guesser
        config: Game configuration
        
    Returns:
        The extracted question from the guesser's output
    """
    guesser_output = ""
    is_thinking_over = False

    # Generate guesser's thinking
    for token in tqdm(guesser_client.generate(
        prompt,
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

    if is_thinking_over:
        guesser_output += end_think_token[1:]
    else:
        guesser_output += end_think_token

    prompt_with_thinking = prompt + guesser_output
    
    # Generate guesser's answer
    for token in tqdm(guesser_client.generate(
        prompt_with_thinking,
        max_new_tokens=config.guesser_answer_budget,
        stream=True,
        seed=config.seed,
        stop=["}"]
    ), desc="Getting answer..."):
        guesser_output += token
        if "}" in token:
            break

    # Extract the question from the output
    question = extract_question_from_generation(guesser_output)
    
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
    judge_prompt, judge_format = judge_prompt_fn(game_entities, question)
    judge_response = ""

    # Generate judge's response
    with tqdm(desc="Judging...") as pbar:
        for token in judge_client.generate(
            judge_prompt[0]['content'],
            max_new_tokens=config.judge_token_budget,
            stream=True,
            response_format=judge_format
        ):
            judge_response += token
            pbar.update(1)

    # Debug the raw response
    print("\nRaw judge response:", repr(judge_response))

    # Clean and parse JSON response
    try:
        # First, try to find JSON in the response
        json_start = judge_response.find('{')
        json_end = judge_response.rfind('}') + 1
        
        if json_start != -1 and json_end != -1:
            json_str = judge_response[json_start:json_end]
        else:
            raise ValueError("No JSON object found in response")

        # Clean up common issues
        json_str = json_str.strip()
        json_str = json_str.replace("'", '"')  # Replace single quotes with double quotes
        json_str = json_str.replace('\n', '')  # Remove newlines
        json_str = re.sub(r'""".*?"""', '', json_str)  # Remove triple quotes if present
        json_str = re.sub(r'```.*?```', '', json_str)  # Remove code blocks if present
        
        # Try to parse the cleaned JSON
        try:
            judge_response = json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"\nJSON parsing error: {e}")
            print("Cleaned JSON string:", repr(json_str))
            raise

        # Validate and normalize the values
        normalized_response = {}
        for entity in game_entities:
            if entity not in judge_response:
                print(f"\nWarning: Missing response for {entity}")
                normalized_response[entity] = "unknown"
                continue
            
            value = judge_response[entity].lower().strip()
            if value not in ["yes", "no", "sometimes", "unknown"]:
                print(f"\nWarning: Invalid value '{value}' for {entity}")
                normalized_response[entity] = "unknown"
            else:
                normalized_response[entity] = value
        
        judge_response = normalized_response

    except Exception as e:
        print(f"\nError processing judge response: {str(e)}")
        print("Raw response:", repr(judge_response))
        # Fallback to safe default
        judge_response = {entity: "unknown" for entity in game_entities}
        print("Falling back to all 'unknown' responses")

    # Debug output
    print(f"\n\t-------[[judge response]]---------\n{judge_response}")
    
    return judge_response


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
    guesser_conversation = [
        dict(role='user', content=guesser_prompt_fn(game_entities))
    ]

    won_on_turn_number = None
    turn_history = []
    
    for turn_number in range(20):
        # Debug turn information
        print(f"\n===========================================")
        print(f"turn {turn_number + 1} input:")
        print(f"{guesser_conversation[-1]['content']}")
        print(f"===========================================")

        # Get question from guesser
        guesser_output, question = guesser_asks(
            guesser_client, 
            guesser_conversation[-1]['content'], 
            config
        )
        
        # Get feedback from judge
        judge_response = judge_feedback(
            judge_client,
            game_entities,
            question,
            config
        )
        
        # Record turn history
        turn_history.append({
            "guesser": guesser_output, 
            "question": question, 
            "json": judge_response
        })
        
        # Show target entity response
        print(f"\t\ttarget entity response: {game_target} --> {judge_response[game_target]}")

        # Update conversation for next turn
        guesser_conversation.append(dict(role='assistant', content=question))
        
        # Check for win condition (placeholder for now)
        if False:  # TODO: code for exiting the for loop game win code
            won_on_turn_number = turn_number + 1
            break
        else:
            judge_response_text = judge_response[game_target].capitalize() + ". What is your next question?"
            guesser_conversation.append(dict(role='user', content=judge_response_text))

    # TODO: return?
    #   - use "turn_history" variable or something like it?

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run the 20 questions game with language models')
    
    # Model configuration
    parser.add_argument('--guesser-model', type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
                      help='Model to use for generating questions')
    parser.add_argument('--judge-model', type=str, default="Qwen/Qwen2.5-3B-Instruct",
                      help='Model to use for judging answers')
    parser.add_argument('--token-path', type=str, help='Path to HuggingFace token file', default="/Users/benigerisimon/Desktop/PhD/hf_token.txt")
    
    # Generation settings
    parser.add_argument('--seed', type=int, help='Random seed for generation')
    parser.add_argument('--no-random-seed', action='store_true', help='Disable random seed generation')
    parser.add_argument('--guesser-think-budget', type=int, default=500, #1000
                      help='Token budget for guesser thinking phase')
    parser.add_argument('--guesser-answer-budget', type=int, default=200, #500
                      help='Token budget for guesser answer phase')
    parser.add_argument('--judge-token-budget', type=int, default=1000, #1000
                      help='Token budget for judge responses')
    
    args = parser.parse_args()

    # Create game configuration
    config = GameConfig(
        guesser_model=args.guesser_model,
        judge_model=args.judge_model,
        hf_token_path=args.token_path,
        use_random_seed=not args.no_random_seed,
        seed=args.seed,
        guesser_think_budget=args.guesser_think_budget,
        guesser_answer_budget=args.guesser_answer_budget,
        judge_token_budget=args.judge_token_budget,
    )

    # Initialize model clients
    guesser_client = ModelClient(config.guesser_model, hf_token_path=config.hf_token_path)
    judge_client = ModelClient(config.judge_model, hf_token_path=config.hf_token_path)

    test_game_entities = [
        "parakeet", "seal", "goose", "stork", "deer", "finch", "seagull", "dog", "dolphin", "cow",
        "budgie", "otter", "rooster", "canary", "walrus", "sheep"
    ]
    test_answer = "budgie"

    play_game(test_game_entities, test_answer, guesser_client, judge_client, config)


