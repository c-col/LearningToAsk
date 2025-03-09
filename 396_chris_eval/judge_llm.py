from typing import List, Dict
from tqdm import tqdm

from model_client import ModelClient
from game_utils import GameConfig
from utils import check_if_question_names_entities


def judge_prompt_fn(active_entities: List[str], question: str) -> List[Dict[str, str]]:
    """Generate the prompt for the judge model.
    
    Args:
        active_entities: List of entities to judge
        question: The question to judge for each entity
        
    Returns:
        List of message dictionaries for the judge model
    """
    if not any([question.endswith(symbol) for symbol in ["?", "."]]):
        question += "?"

    active_entities_string = ", ".join(active_entities)

    prompt = (
        f"Go through the following list and answer the question \"{question}\" for each item.\n"
        f"For each item, respond with EXACTLY one of these values: yes, no, sometimes, or unknown.\n"
        f"List: {active_entities_string}\n\n"
        f"Format your response as a numbered list with one item per line. Here is an example:\n"
        f"1. cat: yes        ✓ (correct format)\n"
        f"2. dog: no         ✓ (correct format)\n"
        f"3. fish: sometimes ✓ (correct format)\n"
        f"4. bird: unknown   ✓ (correct format)\n\n"
        f"Do NOT use any other values like these:\n"
        f"5. snake: probably      ✗ (wrong - use only yes/no/sometimes/unknown)\n"
        f"6. lion: mostly yes     ✗ (wrong - use only yes/no/sometimes/unknown)\n"
        f"7. tiger: not sure      ✗ (wrong - use 'unknown' instead)\n"
        f"8. bear: occasionally   ✗ (wrong - use 'sometimes' instead)\n\n"
        f"Be consistent and precise in your judgments. Consider the literal meaning of the question."
    )
    
    return [{"role": "user", "content": prompt}]


def judge_feedback(judge_client: ModelClient, game_entities: List[str], question: str, config: GameConfig) -> Dict[str, str]:
    """Generate feedback from the judge model for a given question.
    
    Args:
        judge_client: The model client for the judge
        game_entities: List of entities in the game
        question: The question to judge
        config: Game configuration
        
    Returns:
        Dictionary mapping entities to yes/no/sometimes/unknown responses
    """
    judge_prompt = judge_prompt_fn(game_entities, question)
    judge_response = ""

    # Generate judge's response
    with tqdm(desc="Judging...") as pbar:
        for token in judge_client.generate(
            messages=judge_prompt,
            max_new_tokens=config.judge_token_budget,
            stream=True
        ):
            judge_response += token
            pbar.update(1)

    # Debug the raw response
    print("\nRaw judge response:", repr(judge_response))

    # Parse the response into a dictionary
    response_dict = {}
    
    # Process each line of the response
    for line in judge_response.split('\n'):
        line = line.strip().lower()
        # Skip empty lines
        if not line:
            continue
            
        # Remove any numbering at the start
        if '. ' in line:
            line = line.split('. ', 1)[1]
            
        if ':' in line:
            # Standard format with colon separator
            entity, value = line.split(':', 1)
            entity = entity.strip()
            value = value.strip()
            
            # Only process if the entity is in our game entities
            if entity in game_entities:
                # First try exact matching
                if value in ['yes', 'no', 'sometimes', 'unknown']:
                    response_dict[entity] = value
                else:
                    # Try to find yes/no using fuzzy matching
                    value_matches = check_if_question_names_entities(['yes', 'no'], value)
                    if len(value_matches) == 1:
                        response_dict[entity] = value_matches[0]
                    else:
                        # Check for sometimes/unknown using fuzzy matching
                        other_matches = check_if_question_names_entities(['sometimes', 'unknown'], value)
                        if len(other_matches) == 1:
                            response_dict[entity] = other_matches[0]
                        else:
                            # If no clear match found, default to unknown
                            print(f"\nWarning: Invalid value '{value}' for {entity}")
                            response_dict[entity] = "unknown"
        else:
            # No colon in line - try to find both entity and value using fuzzy matching
            entity_matches = check_if_question_names_entities(game_entities, line)
            if len(entity_matches) == 1:
                entity = entity_matches[0]
                # Try to find the value in the same line
                value_matches = check_if_question_names_entities(['yes', 'no'], line)
                if len(value_matches) == 1:
                    response_dict[entity] = value_matches[0]
                else:
                    other_matches = check_if_question_names_entities(['sometimes', 'unknown'], line)
                    if len(other_matches) == 1:
                        response_dict[entity] = other_matches[0]
                    else:
                        print(f"\nWarning: Could not find clear value in line '{line}' for {entity}")
                        response_dict[entity] = "unknown"

    # Ensure all entities have a response
    for entity in game_entities:
        if entity not in response_dict:
            print(f"\nWarning: Missing response for {entity}")
            response_dict[entity] = "unknown"

    # Debug output
    print(f"\n\t-------[[judge response]]---------\n{response_dict}")
    
    return response_dict 