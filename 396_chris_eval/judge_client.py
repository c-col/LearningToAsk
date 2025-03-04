from huggingface_hub import InferenceClient
from utils import load_hf_token

# just for easy reference:
judge_model_options = ["Qwen/Qwen2.5-Coder-1.5B-Instruct", "Qwen/Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct"]
judge_model = "Qwen/Qwen2.5-3B-Instruct"

client = InferenceClient(
    model=judge_model,
    provider="hf-inference",
    api_key=load_hf_token()
)

game_entities = [
    "parakeet", "seal", "goose", "stork", "deer", "finch", "seagull", "dog", "dolphin", "cow",
    "budgie", "otter", "rooster", "canary", "walrus", "sheep"
]
game_entities_string = ", ".join(game_entities)

# ==========
current_question = "Is it a bird?"
temp_entity_name = "budgie"


def judge_one_entity(entity_name, question):
    is_entity_prefix_an = any([entity_name.startswith(vowel) for vowel in ["a", "e", "i", "o", "u"]])  # rough approx
    entity_string = " ".join([("an" if is_entity_prefix_an else "a"), entity_name])

    if not any([question.endswith(symbol) for symbol in ["?", "."]]):
        question += "?"

    one_prompt = (f"I'm learning about something called a {entity_string}. "
                  f"{question} Limit your answer to \"yes\", \"no\", or \"unknown\".")

    messages = [
        {
            "role": "user",
            "content": one_prompt
        }
    ]
    return messages


def judge_all_entity(active_entities, question):
    if not any([question.endswith(symbol) for symbol in ["?", "."]]):
        question += "?"

    active_entities_string = ", ".join(active_entities)
    all_prompt = (f"I have a list of things, and for each thing, I want to know \"{question}\". "
                  f"The list of things is: {active_entities_string}. "
                  f"Please generate a JSON output that assigns a value of \"yes\", \"no\", or \"unknown\" "
                  f"for each element from my list.")

    response_format = {
        "type": "json",
        "value": {
            "properties": {k: {"type": "string"} for k in active_entities},
            "required": active_entities,
            #     {
            #     "location": {"type": "string"},
            #     "activity": {"type": "string"},
            #     "animals_seen": {"type": "integer", "minimum": 1, "maximum": 5},
            #     "animals": {"type": "array", "items": {"type": "string"}},
            # },
            # "required": ["location", "activity", "animals_seen", "animals"],
        },
    }
    messages = [
        {
            "role": "user",
            "content": all_prompt,
        }
    ]
    return messages, response_format


use_one_entity_prompt = False
if use_one_entity_prompt:
    judge_prompt = judge_one_entity(temp_entity_name, current_question)
    judge_response = client.chat.completions.create(
        messages=judge_prompt,
        max_tokens=500,
    )
else:
    judge_prompt, judge_format = judge_all_entity(game_entities, current_question)
    judge_response = client.chat.completions.create(
        messages=judge_prompt,
        max_tokens=500,
        response_format=judge_format
    )

    # example:
    # 'I have a list of things, and for each thing, I want to know "Is it a bird?".
    # The list of things is: parakeet, seal, goose, stork, deer, finch, seagull, dog, dolphin, cow,
    # budgie, otter, rooster, canary, walrus, sheep.
    # Please generate a JSON output that assigns a value of "yes", "no", or "unknown" for each element from my list.'
    #
    # ChatCompletionOutputMessage(
    #   role='assistant',
    #   content='
    #       {
    #           "parakeet": "yes",
    #           "seal": "no",
    #           "goose": "yes",
    #           "stork": "yes",
    #           "deer": "no",
    #           "finch": "yes",
    #           "seagull": "yes",
    #           "dog": "no",
    #           "dolphin": "no",
    #           "cow": "no",
    #           "budgie": "yes",
    #           "otter": "no",
    #           "rooster": "yes",
    #           "canary": "yes",
    #           "walrus": "no",
    #           "sheep": "no"
    #       }',
    #   tool_calls=None
    # )

print(judge_prompt)
print("")
print(judge_response.choices[0].message)
print(judge_response.choices[0].message.content)
