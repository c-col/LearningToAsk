from huggingface_hub import InferenceClient
from transformers import AutoTokenizer


from tqdm import tqdm
from random import randint, shuffle

from utils import load_hf_token, extract_question_from_generation

# generation settings
stream_output = True  # should set to True with HuggingFace inference -- otherwise too sensitive to gateway errors
use_random_seed = True  #
seed = randint(0, 1000) if use_random_seed else 1
think_max_budget = 200
answer_max_budget = 200

# guesser model / inference API settings
guesser_model = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
tokenizer = AutoTokenizer.from_pretrained(guesser_model)  # set up tokenizer to use for chat template
client = InferenceClient(
    model=guesser_model,
    provider="hf-inference",
    api_key=load_hf_token()
)

game_entities = [
    "parakeet", "seal", "goose", "stork", "deer", "finch", "seagull", "dog", "dolphin", "cow",
    "budgie", "otter", "rooster", "canary", "walrus", "sheep"
]
if use_random_seed:
    shuffle(game_entities)

# prompt stuff
game_entities_string = ", ".join(game_entities)
guesser_prompt = (f"Let's play 20 questions. I'm thinking of one of the things in this list: {game_entities_string}. "
                  "Each turn, you ask me one yes/no question about the mystery thing -- your goal is to determine the "
                  "answer in the fewest number of turns. If a question is subjective, I'll answer with \"unknown\". "
                  "When you decide on a question, finalize it by writing it using \\boxed{}.")

end_think_token = "</think> Final answer: \\boxed{"
conversation = [
    dict(role='user', content=guesser_prompt),
    # dict(role='assistant', content='hello, nice to meet you!'),
    # dict(role='user', content='1+1='),
    # dict(role='assistant', content='<think>The user want to know'),
]

guesser_input = tokenizer.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
guesser_output = ""
is_thinking_over = False
for token in tqdm(client.text_generation(guesser_input, max_new_tokens=think_max_budget, stream=True, seed=seed, stop=["}"]), total=think_max_budget, desc="Thinking..."):
    guesser_output += token
    if token == "</think>":
        is_thinking_over = True

continue_answer = True
if is_thinking_over:
    if guesser_output.endswith("}"):
        continue_answer = False
else:
    guesser_output += end_think_token

if continue_answer:
    guesser_input += guesser_output
    with tqdm(desc="Answering...") as pbar:
        for token in client.text_generation(guesser_input, max_new_tokens=answer_max_budget, stream=True, stop=["}"], seed=seed):
            guesser_output += token
            pbar.update(1)
guesser_question = extract_question_from_generation(guesser_output)

print(guesser_output)
print(f"\n=====\n Extracted question: '{guesser_question}'")
input("hi")

