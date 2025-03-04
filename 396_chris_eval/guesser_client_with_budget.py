from huggingface_hub import InferenceClient
from tqdm import tqdm

from utils import load_hf_token, extract_question_from_generation

guesser_model = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
client = InferenceClient(
    model=guesser_model,
    provider="hf-inference",
    api_key=load_hf_token()
)

game_entities = [
    "parakeet", "seal", "goose", "stork", "deer", "finch", "seagull", "dog", "dolphin", "cow",
    "budgie", "otter", "rooster", "canary", "walrus", "sheep"
]

# prompt stuff
game_entities_string = ", ".join(game_entities)
guesser_prompt = (f"Let's play 20 questions. I'm thinking of an element from this list: {game_entities_string}. "
                  f"Each turn, you can ask me one yes/no question about the mystery element, and I'll answer truthfully. "
                  f"Your goal is to minimize the number of questions used to reach the answer -- if you don't get it "
                  "within 20 turns, you lose. Ask your question by enclosing it in \\boxed{}. What's your first question?")

guesser_messages = f"<｜User｜>{guesser_prompt}<｜Assistant｜>"  # use chat template

think_max_budget = 1000
answer_max_budget = 200
end_think_token = "</think> Final answer: \\boxed{"

stream_output = True
if stream_output:
    guesser_generation = ""
    generation_details = None
    for token in tqdm(client.text_generation(guesser_messages, max_new_tokens=think_max_budget, stream=True), total=think_max_budget, desc="Thinking..."):
        guesser_generation += token

    if "</think>" in guesser_generation:
        # check answer side for viable "boxed-style" answer
        guesser_answer = guesser_generation.split("</think>")[-1]
        if "\\boxed{" not in guesser_answer or "}" not in guesser_answer:
            for token in client.text_generation(guesser_messages + guesser_generation, max_new_tokens=answer_max_budget, stream=True, stop=["}"]):
                guesser_generation += token
    else:
        guesser_generation += end_think_token
        for token in client.text_generation(guesser_messages + guesser_generation, max_new_tokens=answer_max_budget, stream=True):
            guesser_generation += token
    guesser_generation = guesser_generation.replace("<｜end▁of▁sentence｜>", "")
else:
    guesser_output = client.text_generation(
        guesser_messages, max_new_tokens=think_max_budget, decoder_input_details=True, details=True
    )
    guesser_generation = guesser_output.generated_text
    generation_details = guesser_output.details

    if guesser_output.details.finish_reason == "length":
        if end_think_token not in guesser_generation:
            guesser_generation += end_think_token
        guesser_generation_finish = client.text_generation(
            guesser_messages + guesser_generation,
            max_new_tokens=answer_max_budget,
            decoder_input_details=True, details=True,
            stop=["}"]  # ensures that model doesn't reason past the "\boxed{}" formatting
        )
        generation_details = guesser_generation_finish.details
        guesser_generation += guesser_generation_finish.generated_text

guesser_question = extract_question_from_generation(guesser_generation)
print(guesser_generation)
print(f"\n=====\n Extracted question: '{guesser_question}'")

