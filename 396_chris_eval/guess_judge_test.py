from huggingface_hub import InferenceClient
from transformers import AutoTokenizer
from typing import List
import json
from time import sleep
from tqdm import tqdm
from random import randint, shuffle

from utils import load_hf_token, extract_question_from_generation


# generation settings
use_random_seed = True
game_seed = randint(0, 1000) if use_random_seed else 1
guesser_think_budget = 1000
guesser_answer_budget = 500
judge_token_budget = 2000  # high just to be safe

# guesser model / inference API settings
guesser_model = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
tokenizer = AutoTokenizer.from_pretrained(guesser_model)  # set up tokenizer to use for chat template
guess_client = InferenceClient(
    model=guesser_model,
    provider="hf-inference",
    api_key=load_hf_token()
)

# judge model / inference API settings
judge_model = "Qwen/Qwen2.5-3B-Instruct"
judge_client = InferenceClient(
    model=judge_model,
    provider="hf-inference",
    api_key=load_hf_token()
)

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
def judge_prompt_fn_single(entity_name, question):
    is_entity_prefix_an = any([entity_name.startswith(vowel) for vowel in ["a", "e", "i", "o", "u"]])  # rough approx
    entity_string = " ".join([("an" if is_entity_prefix_an else "a"), entity_name])

    if not any([question.endswith(symbol) for symbol in ["?", "."]]):
        question += "?"

    one_prompt = (f"I'm learning about something called a {entity_string}. "
                  f"{question} Limit your answer to \"yes\", \"no\", \"sometimes\", or \"unknown\".")

    response_format = {
        "type": "regex",
        "value": "(yes|no|sometimes|unknown)",
    }

    messages = [
        {
            "role": "user",
            "content": one_prompt
        }
    ]
    return messages, response_format

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


def play_game(game_entities: List[str], game_target: str, seed: int):
    if use_random_seed:
        shuffle(game_entities)
    guesser_conversation = [
        dict(role='user', content=guesser_prompt_fn(game_entities))
    ]

    won_on_turn_number = None
    turn_history = []
    for turn_number in range(20):
        # # # guesser code
        guesser_input = tokenizer.apply_chat_template(guesser_conversation, add_generation_prompt=True, tokenize=False)
        guesser_output = ""
        is_thinking_over = False

        # debug
        print(f"\n===========================================")
        print(f"turn {turn_number + 1} input:")
        print(f"{guesser_input}")
        print(f"===========================================")
        # # #

        # for guesser_token in tqdm(guess_client.text_generation(guesser_input, max_new_tokens=guesser_think_budget, stream=True, seed=seed), total=guesser_think_budget, desc="Guessing..."):
        generation_success = False
        while not generation_success:
            temp_output = ""
            try:
                for token_ix, guesser_token in enumerate(tqdm(guess_client.text_generation(guesser_input, max_new_tokens=guesser_think_budget + 500, stream=True, seed=seed), total=guesser_think_budget + 500, desc="Guessing...")):
                    # modified budget forcing here so that the model isn't interrupted mid-thought (kept generating really bad questions)
                    if token_ix + 1 >= guesser_think_budget:
                        if "\n\n" in guesser_token:
                            temp_output += guesser_token.replace("\n\n", "")
                            break
                    # # #
                    if guesser_token == "</think>":
                        # then, the model already added a "\n" to the generation,
                        is_thinking_over = True
                        break
                    temp_output += guesser_token
                generation_success = True
                guesser_output += temp_output
            except Exception as e:
                print(f"Guesser generation failure: {e}")
                print("\tSleeping for 10 seconds...")
                sleep(10)
                seed += 1


        if is_thinking_over:
            guesser_output += end_think_token[1:]
        else:
            guesser_output += end_think_token

        guesser_input += guesser_output
        generation_success = False
        while not generation_success:
            temp_output = ""
            try:
                for token in guess_client.text_generation(guesser_input, max_new_tokens=guesser_answer_budget, stream=True, seed=seed):  # stop=["}"]
                    temp_output += token
                    if "}" in token:
                        break
                guesser_output += temp_output
                generation_success = True
            except Exception as e:
                print(f"Guesser generation failure: {e}")
                print("\tSleeping for 10 seconds...")
                sleep(10)
                seed += 1

        guesser_question = extract_question_from_generation(guesser_output)

        # debug
        print(f"\n------[[turn {turn_number + 1} guesser output]]------")
        print(guesser_output)
        print(f"\n\tExtracted question: {guesser_question}")
        # # #


        # for entity in game_entities:
        #     judge_prompt, judge_format = judge_prompt_fn_single(game_entities, guesser_output)

        judge_prompt, judge_format = judge_prompt_fn(game_entities, guesser_question)
        generation_success = False
        while not generation_success:
            judge_response = ""
            try:
                with tqdm(desc="Judging...") as pbar:
                    for token in judge_client.chat.completions.create(messages=judge_prompt, max_tokens=judge_token_budget, response_format=judge_format, stream=True, seed=seed):
                        judge_response += token.choices[0].delta.content
                        pbar.update(1)
            except Exception as e:
                print(f"Judge generation failure: {e}")
                print("\tSleeping for 10 seconds...")
                sleep(10)
                seed += 1

            judge_response = judge_response.replace("'", '"')
            try:
                judge_response = json.loads(judge_response)
                generation_success = True
            except Exception as e:
                print(f"Judge JSON conversion failure: {e}")
                print("\t", judge_response)


        # tidy up response if needed
        # TODO: delete this later
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
        # # #

        # TODO: check if target_entity matches direct question from guesser
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

    play_game(test_game_entities, test_answer, game_seed)

