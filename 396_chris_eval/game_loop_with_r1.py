from huggingface_hub import get_inference_endpoint, InferenceClient
from transformers import AutoTokenizer
from typing import List
from time import sleep
from tqdm import tqdm
from random import randint, shuffle

from utils import extract_question_from_generation, check_if_question_names_entities, \
    generate_json_from_question_entities, load_hf_token, find_whole_word

client = InferenceClient(
    provider="sambanova",
    api_key=load_hf_token()
)


def judge_prompt_r1(entity_list: List[str], question: str):
    entity_str = ", ".join(entity_list)
    prompt = f"Go through the following list and answer \"yes\" or \"no\" for the question \"{question}\". List: {entity_str}"
    conversation = [
        {
            "role": "user",
            "content": prompt
        }
    ]
    return conversation


def judge_prompt_r1_backup(entity_list: List[str], question: str):
    entity_str = ", ".join(entity_list)
    prompt = f"Go through the following list and answer \"yes\" or \"no\" for the question \"{question}\". You should output a numbered list containing each item, followed by yes or no. List: {entity_str}"
    conversation = [
        {
            "role": "user",
            "content": prompt
        }
    ]
    return conversation


# generation settings
use_random_seed = True
game_seed = randint(0, 1000) if use_random_seed else 1
guesser_think_budget = 2000
guesser_answer_budget = 500
judge_token_budget = 10000


end_think_token = "\n</think>\n\nMy question for this turn: \\boxed{"
end_think_token_suffix = "</think>\n\nMy question for this turn: \\boxed{"


# guesser model / inference API settings
model = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
endpoint = get_inference_endpoint("deepseek-r1-distill-qwen-7b-kaf")
client = endpoint.client
tokenizer = AutoTokenizer.from_pretrained(model)  # set up tokenizer to use for chat template


# guesser prompt / conversation
def guesser_prompt_fn(entity_list: List[str]) -> str:
    entities_string = ", ".join(entity_list)
    return (f"Let's play 20 questions. I'm thinking of one of the things in this list: {entities_string}. "
            "Each turn, you ask me one yes/no question about the mystery thing -- your goal is to determine the "
            "answer in the fewest number of turns. "
            "When you decide on a question, finalize it by writing it using \\boxed{}.")


def play_game(game_entities: List[str], game_target: str):
    if use_random_seed:
        shuffle(game_entities)
    guesser_conversation = [
        dict(role='user', content=guesser_prompt_fn(game_entities))
    ]

    won_on_turn_number = None
    game_over = False
    turn_history = []
    for turn_number in range(20):
        guesser_input = tokenizer.apply_chat_template(guesser_conversation, add_generation_prompt=True, tokenize=False)
        with tqdm(desc=f"[Turn {turn_number + 1}] Guessing") as pbar:
            generation_success = False
            while not generation_success:
                try:
                    guesser_response = client.text_generation(
                        guesser_input, max_new_tokens=guesser_think_budget,
                    )
                except Exception as e:
                    print(f"Exception during thinking: {e}")
                    print("\t...Sleeping for 15 seconds...")
                    sleep(15)
                    continue
                generation_success = True

            if "</think>" in guesser_response:
                guesser_response = guesser_response.split("</think>")[0] + end_think_token_suffix
            else:
                guesser_response += end_think_token

            generation_success = False
            while not generation_success:
                try:
                    guesser_question = client.text_generation(
                        guesser_input + guesser_response, max_new_tokens=guesser_answer_budget,
                    )
                except Exception as e:
                    print(f"Exception during guessing: {e}")
                    print("\t...Sleeping for 15 seconds...")
                    sleep(15)
                    continue
                generation_success = True

            pbar.update(1)
        guesser_response += guesser_question
        guesser_question = extract_question_from_generation(guesser_response)

        # debug
        print(f"\n==================== turn {turn_number + 1} =======================")
        print(f"------[[guesser input]]------")
        print(guesser_input.replace('<｜User｜>', '\n\t<｜User｜>').replace('<｜Assistant｜>', '\n\t\t<｜Assistant｜>'))
        print(f"\n------[[guesser output]]------")
        print(guesser_response)
        print(f"\n\tExtracted question: {guesser_question}")
        # # #

        entities_in_question = check_if_question_names_entities(game_entities, guesser_question)
        if len(entities_in_question):
            with tqdm(desc=f"[Turn {turn_number + 1}] Judging skipped") as pbar:
                judge_json = generate_json_from_question_entities(game_entities, entities_in_question)
                if len(entities_in_question) == 1:
                    if judge_json[game_target] == "yes":
                        game_over = True
                pbar.update(1)
        else:
            judging_complete = False
            entities_to_judge = list(game_entities)
            judge_json = {}
            is_retry = False
            with tqdm(desc=f"[Turn {turn_number + 1}] Judging") as pbar:
                while not judging_complete:
                    judge_input = judge_prompt_r1_backup(entities_to_judge, guesser_question)
                    # if is_retry:
                    #     judge_input = judge_prompt_r1_backup(entities_to_judge, guesser_question)
                    # else:
                    #     judge_input = judge_prompt_r1(entities_to_judge, guesser_question)
                    judge_response = client.chat.completions.create(
                        model="deepseek-ai/DeepSeek-R1",
                        messages=judge_input,
                        max_tokens=judge_token_budget,
                    )
                    judge_response = judge_response.choices[0].message.content
                    judge_answer = judge_response.split("</think>")[-1]

                    for line in judge_answer.split("\n"):
                        value_present = list(set(check_if_question_names_entities(["yes", "no"], line)))
                        if len(value_present) == 1:
                            entity_present = list(set(check_if_question_names_entities(entities_to_judge, line)))
                            if len(entity_present) == 1:
                                if entity_present[0] not in judge_json:
                                    judge_json[entity_present[0]] = value_present[0]
                    entities_to_judge = [entity for entity in game_entities if entity not in judge_json]
                    if len(entities_to_judge) == 0:
                        judging_complete = True
                    else:
                        is_retry = True
                        print(f"Judge is missing entities: {entities_to_judge}")
                        print(f"---- Input to judge:")
                        print(judge_input)
                        print(f"---- Judge post-thinking answer:")
                        print(judge_answer)
                pbar.update(1)

        turn_history.append({
            "guesser_thoughts": guesser_response, "question": guesser_question, "answers": judge_json,
        })

        # debug
        print(f"\n------[[judge response]]------")
        for k, v in judge_json.items():
            if k == game_target:
                print("\t*** " + k + ": " + v)
            else:
                print("\t" + k + ": " + v)

        # # #

        guesser_conversation.append(dict(role='assistant', content=guesser_question))
        if game_over:
            won_on_turn_number = turn_number + 1
            break
        else:
            judge_response_text = judge_json[game_target].capitalize() + ". What is your next question?"
            guesser_conversation.append(dict(role='user', content=judge_response_text))

    print("\n\n\n====================================")
    print(f"game over!!! won on turn {won_on_turn_number}")
    print(turn_history)
    # TODO: return?
    #   - use "turn_history" variable or something like it?


if __name__ == "__main__":
    # test_game_entities = [
    #     "parakeet", "seal", "goose", "stork", "deer", "finch", "seagull", "dog", "dolphin", "cow",
    #     "budgie", "otter", "rooster", "canary", "walrus", "sheep"
    # ]
    # test_answer = "budgie"

    test_game_entities = [
        "apple", "television", "dinosaur", "airplane", "house", "tree", "coat", "shoes", "car", "train", "shower",
        "frisbee", "cow", "giganotosaurus", "siberian husky", "glass micropipette", "anger", "love", "hate",
        "contentment", "jealousy", "surprise", "disgust", "hopefulness", "global poverty", "phase transition",
        "positive sum game", "beauty", "representative democracy"
    ]
    test_answer = "jealousy"

    play_game(test_game_entities, test_answer)
