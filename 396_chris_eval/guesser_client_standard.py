from huggingface_hub import InferenceClient
from utils import load_hf_token, extract_question_from_generation

guesser_client = InferenceClient(
    model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    provider="hf-inference",
    api_key=load_hf_token()
)

game_entities = [
    "parakeet", "seal", "goose", "stork", "deer", "finch", "seagull", "dog", "dolphin", "cow",
    "budgie", "otter", "rooster", "canary", "walrus", "sheep"
]
game_entities_string = ", ".join(game_entities)

guesser_prompt = (f"Let's play 20 questions. I'm thinking of an element from this list: {game_entities_string}. "
                  f"Each turn, you can ask me one yes/no question about the mystery element, and I'll answer truthfully. "
                  f"Your goal is to minimize the number of questions used to reach the answer -- if you don't get it "
                  "within 20 turns, you lose. Ask your question by enclosing it in \\boxed{}. What's your first question?")
guesser_messages = [
    {
        "role": "user",
        "content": guesser_prompt
    }
]

stream_tokens = True
if stream_tokens:
    for token in guesser_client.chat.completions.create(messages=guesser_messages, max_tokens=1000, stream=True):
        print(token)
else:
    guesser_response = guesser_client.chat.completions.create(
        messages=guesser_messages,
        max_tokens=1000,
    )
    print(guesser_response.choices[0].message.content)
