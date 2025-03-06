import re
from typing import List, Dict
from math import log2

def load_hf_token(filepath: str = "C:\\Users\\chris\\PycharmProjects\\hf_token.txt") -> str:  # hehe
    with open(filepath, 'r') as f:
        return f.readline()


def extract_question_from_generation(generation: str) -> str:
    try:
        return re.search('boxed{(.+?)}', generation).group(1)
    except AttributeError:
        raise ValueError(f"Invalid guesser generation '{generation}' \n== could not extract question from '\\boxed'")


def extract_question_and_clean(generation: str) -> str:
    """Extract and clean up a question from \\boxed{} format.
    
    Args:
        generation: The raw generated text containing a \\boxed{} question
        
    Returns:
        Cleaned up question string
        
    Raises:
        ValueError: If no question can be extracted from \\boxed{}
    """
    try:
        # Extract the question from \boxed{}
        question = re.search(r'\\boxed{(.+?)}', generation).group(1)
        
        # Clean up the question
        question = (question
                   .replace('\\', '')  # Remove escape characters
                   .replace('  ', ' ')  # Remove double spaces
                   .strip())  # Remove leading/trailing whitespace
        
        # Ensure the question ends with a question mark
        if not question.endswith('?'):
            question += '?'
            
        return question
    except AttributeError:
        raise ValueError(f"Invalid guesser generation '{generation}' \n== could not extract question from '\\boxed'")




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


def compute_ideal_information_gain(total_entities: int) -> float:
    """Compute the maximum possible information gain from a perfect 50-50 split.
    
    Args:
        total_entities: Total number of entities before the split
        
    Returns:
        Information gain in bits for a perfect binary split (1 bit for powers of 2)
    """
    if total_entities <= 1:
        return 0.0
        
    # For a perfect split, remaining_count = eliminated_count = total_entities/2
    half_count = total_entities / 2
    
    # Initial entropy
    parent_entropy = log2(total_entities)
    
    # For 50-50 split, p_remaining = p_eliminated = 0.5
    weighted_child_entropy = log2(half_count)  # 0.5 * log2(n/2) + 0.5 * log2(n/2)
            
    return parent_entropy - weighted_child_entropy


def compute_information_gain(total_entities: int, remaining_count: int) -> float:
    """Compute information gain based on binary split between remaining and eliminated entities.
    
    In the 20 questions game, entities are split into two groups after each question:
    1. Remaining: Entities that match the target's answer + entities with "sometimes"/"unknown" responses
    2. Eliminated: All other entities (those that don't match target's answer)
    
    For example, if the target's answer is "yes", then:
    - Remaining = entities with "yes" + "sometimes" + "unknown" responses
    - Eliminated = entities with "no" responses
    
    Args:
        total_entities: Total number of entities before the split
        remaining_count: Number of entities that match target's answer or have "sometimes"/"unknown" responses
        
    Returns:
        Information gain in bits, computed as the reduction in entropy from the split
    """
    if total_entities == 0 or remaining_count == 0:
        return 0.0
        
    eliminated_count = total_entities - remaining_count
    
    # Skip if no meaningful split
    if eliminated_count == 0:
        return 0.0
        
    # Initial entropy (parent)
    parent_entropy = log2(total_entities)
    
    # Calculate weighted entropy of the binary split
    p_remaining = remaining_count / total_entities
    p_eliminated = eliminated_count / total_entities
    
    weighted_child_entropy = (
        p_remaining * log2(remaining_count) +
        p_eliminated * log2(eliminated_count)
    )
            
    # Information gain is reduction in entropy
    return max(0.0, parent_entropy - weighted_child_entropy)