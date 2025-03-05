import re

def load_hf_token(filepath: str = "C:\\Users\\chris\\PycharmProjects\\hf_token.txt") -> str:  # hehe
    with open(filepath, 'r') as f:
        return f.readline()


def extract_question_from_generation(generation: str) -> str:
    try:
        return re.search('boxed{(.+?)}', generation).group(1)
    except AttributeError:
        raise ValueError(f"Invalid guesser generation '{generation}' \n== could not extract question from '\\boxed'")
        # return ""



