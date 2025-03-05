from huggingface_hub import InferenceClient
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from typing import List, Optional, Iterator, Dict, Any
import json
from tqdm import tqdm
from random import randint, shuffle
import torch
from dataclasses import dataclass

from utils import load_hf_token, extract_question_from_generation


class ModelClient:
    def __init__(self, model_name: str, use_local: bool = False, device: Optional[str] = None):
        self.model_name = model_name
        self.use_local = use_local
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        if use_local:
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
            self.pipeline = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer, device=self.device)
            self.client = None
        else:
            # Use the global hf_token_path if specified
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
        """Unified generation interface that works for both local and API inference."""
        if self.use_local:
            outputs = self.pipeline(
                prompt,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id,
                stream=stream
            )
            prompt_len = len(prompt)
            for output in outputs:
                token = output[0]['generated_text'][prompt_len:]
                if stop and any(s in token for s in stop):
                    token = token[:min(token.find(s) + len(s) for s in stop if s in token)]
                    yield token
                    break
                yield token
        else:
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


# generation settings
use_random_seed = True
seed = randint(0, 1000) if use_random_seed else 1
guesser_think_budget = 1000
guesser_answer_budget = 500
judge_token_budget = 1000  # high just to be safe

# Model configuration
use_local_models = False  # Set to True to run models locally instead of using HF API
hf_token_path = None  # Set this to the path of your HuggingFace token file, or None to use the default path
gpu_device = None  # GPU device to use when running locally (e.g., "cuda:0", "cuda:1", etc.)

# Initialize model clients
guesser_model = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
judge_model = "Qwen/Qwen2.5-3B-Instruct"

guesser_client = ModelClient(guesser_model, use_local=use_local_models, device=gpu_device)
judge_client = ModelClient(judge_model, use_local=use_local_models, device=gpu_device)

# guesser prompt / conversation
def guesser_prompt_fn(entity_list: List[str]) -> str:
    entities_string = ", ".join(entity_list)
    return (f"Let's play 20 questions. I'm thinking of one of the things in this list: {entities_string}. "
            "Each turn, you ask me one yes/no question about the mystery thing -- your goal is to determine the "
            "answer in the fewest number of turns. The four potential responses to a given question are "
            "\"yes\", \"no\", \"sometimes\", and \"unknown\". "
            "When you decide on a question, finalize it by writing it using \\boxed{}.")


end_think_token = "\n</think>\n\nMy question for this turn: \\boxed{"


# judge prompt builder
def judge_prompt_fn(active_entities, question):
    if not any([question.endswith(symbol) for symbol in ["?", "."]]):
        question += "?"

    active_entities_string = ", ".join(active_entities)
    all_prompt = (f"I have a list of things, and for each thing, I want to know \"{question}\". "
                  f"The list of things is: {active_entities_string}. "
                  f"Generate a JSON output that assigns a value of \"yes\", \"no\", \"sometimes\", or \"unknown\" "
                  f"for each element of the list.")

    response_format = {
        "type": "json",
        "value": {
            "properties": {k: {"type": "string"} for k in active_entities},
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


def play_game(game_entities: List[str], game_target: str):
    if use_random_seed:
        shuffle(game_entities)
    guesser_conversation = [
        dict(role='user', content=guesser_prompt_fn(game_entities))
    ]

    won_on_turn_number = None
    turn_history = []
    for turn_number in range(20):
        guesser_input = guesser_client.tokenizer.apply_chat_template(guesser_conversation, add_generation_prompt=True, tokenize=False)
        guesser_output = ""
        is_thinking_over = False

        # debug
        print(f"\n===========================================")
        print(f"turn {turn_number + 1} input:")
        print(f"{guesser_input}")
        print(f"===========================================")

        # Generate guesser's thinking
        for token in tqdm(guesser_client.generate(
            guesser_input,
            max_new_tokens=guesser_think_budget + 500,
            stream=True,
            seed=seed
        ), total=guesser_think_budget + 500, desc="Guessing..."):
            if len(guesser_output) + 1 >= guesser_think_budget:
                if "\n\n" in token:
                    guesser_output += token.replace("\n\n", "")
                    break
            if token == "</think>":
                is_thinking_over = True
                break
            guesser_output += token

        if is_thinking_over:
            guesser_output += end_think_token[1:]
        else:
            guesser_output += end_think_token

        guesser_input += guesser_output
        
        # Generate guesser's answer
        for token in guesser_client.generate(
            guesser_input,
            max_new_tokens=guesser_answer_budget,
            stream=True,
            seed=seed,
            stop=["}"]
        ):
            guesser_output += token
            if "}" in token:
                break

        guesser_question = extract_question_from_generation(guesser_output)

        # debug
        print(f"\n------[[turn {turn_number + 1} guesser output]]------")
        print(guesser_output)

        judge_prompt, judge_format = judge_prompt_fn(game_entities, guesser_output)
        judge_response = ""

        # Generate judge's response
        with tqdm(desc="Judging...") as pbar:
            for token in judge_client.generate(
                judge_prompt[0]['content'],
                max_new_tokens=judge_token_budget,
                stream=True,
                response_format=judge_format
            ):
                judge_response += token
                pbar.update(1)

        try:
            judge_response = json.loads(judge_response)
        except Exception as e:
            print("\n\n\n\nJSON EXCEPTION!!")
            print(judge_response)
            judge_response = judge_response.replace("'", '"')
            judge_response = json.loads(judge_response)

        # tidy up response if needed
        judge_response = {k: v.lower().strip() for k, v in judge_response.items()}
        for k, v in judge_response.items():
            if v not in ["yes", "no", "sometimes", "unknown"]:
                print("\n\n\n\nJUDGE VALUE EXCEPTION!!")
                print(judge_response)
                print(f"'{v}'")
                raise ValueError(f"judge value out of bounds for field {k}: {v}")

        turn_history.append({"guesser": guesser_output, "question": guesser_question, "json": judge_response})

        # debug
        print(f"\n\t-------[[turn {turn_number + 1} judge response]]---------\n{judge_response}")
        print(f"\t\ttarget entity response: {game_target} --> {judge_response[game_target]}")

        guesser_conversation.append(dict(role='assistant', content=guesser_question))
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
    parser.add_argument('--local', action='store_true', help='Run models locally instead of using HF API')
    parser.add_argument('--token-path', type=str, help='Path to HuggingFace token file')
    parser.add_argument('--seed', type=int, help='Random seed for generation')
    parser.add_argument('--gpu', type=str, help='GPU device to use when running locally. Can be specified as a number (0, 1, 2) or as "cuda:X". If not specified, will use the first available GPU or CPU if no GPU is available.')
    args = parser.parse_args()

    if args.local:
        use_local_models = True
        if args.gpu:
            if not torch.cuda.is_available():
                print("Warning: GPU specified but CUDA is not available. Falling back to CPU.")
                gpu_device = "cpu"
            else:
                # Handle both "cuda:X" and simple number formats
                try:
                    if ":" in args.gpu:
                        device_num = int(args.gpu.split(':')[1])
                    else:
                        device_num = int(args.gpu)
                    
                    if device_num >= torch.cuda.device_count():
                        print(f"Warning: GPU {device_num} not found. Available GPUs: {list(range(torch.cuda.device_count()))}. Using first available GPU.")
                        gpu_device = "cuda:0"
                    else:
                        gpu_device = f"cuda:{device_num}"
                        print(f"Using GPU: {gpu_device}")
                except ValueError:
                    print(f"Warning: Invalid GPU specification '{args.gpu}'. Please use a number (0, 1, 2) or 'cuda:X' format. Using first available GPU.")
                    gpu_device = "cuda:0"
        elif torch.cuda.is_available():
            gpu_device = "cuda:0"
            print(f"No GPU specified. Using first available GPU: {gpu_device}")
        else:
            gpu_device = "cpu"
            print("No GPU available. Using CPU.")
    if args.token_path:
        hf_token_path = args.token_path
    if args.seed is not None:
        use_random_seed = False
        seed = args.seed

    # Re-initialize clients with updated configuration
    guesser_client = ModelClient(guesser_model, use_local=use_local_models, device=gpu_device)
    judge_client = ModelClient(judge_model, use_local=use_local_models, device=gpu_device)

    test_game_entities = [
        "parakeet", "seal", "goose", "stork", "deer", "finch", "seagull", "dog", "dolphin", "cow",
        "budgie", "otter", "rooster", "canary", "walrus", "sheep"
    ]
    test_answer = "budgie"

    # test_game_entities = [
    #     "apple", "television", "dinosaur", "airplane", "house", "tree", "coat", "shoes", "car", "train", "shower",
    #     "frisbee", "cow", "giganotosaurus", "siberian husky", "glass micropipette", "anger", "love", "hate",
    #     "contentment", "jealousy", "surprise", "disgust", "hopefulness", "global poverty", "phase transition",
    #     "positive sum game", "beauty", "representative democracy"
    # ]
    # test_answer = "jealousy"

    play_game(test_game_entities, test_answer)


