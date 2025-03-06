import re
from typing import List

def load_hf_token(filepath: str = "C:\\Users\\chris\\PycharmProjects\\hf_token.txt") -> str:  # hehe
    with open(filepath, 'r') as f:
        return f.readline()


def extract_question_from_generation(generation: str) -> str:
    try:
        return re.search('boxed{(.+?)}', generation).group(1)
    except AttributeError:
        raise ValueError(f"Invalid guesser generation '{generation}' \n== could not extract question from '\\boxed'")
        # return ""


def find_whole_word(w):
    return re.compile(r'\b({0})\b'.format(w), flags=re.IGNORECASE).search


def check_if_question_names_entities(entity_list: List[str], question: str) -> List[str]:
    question = question.lower()
    matches = []
    for entity in entity_list:
        try:
            match = find_whole_word(entity)(question).group(1)
            matches.append(match)
        except AttributeError:
            pass
    return matches


def generate_json_from_question_entities(full_entity_list: List[str], partial_entity_list: List[str]):
    output = {k: "no" for k in full_entity_list}
    for partial_entity in partial_entity_list:
        output[partial_entity] = "yes"
    return output

