from typing import List, Dict, Tuple
from tqdm import tqdm
from time import sleep

from model_client import ModelClient
from utils import extract_question_and_clean
from game_utils import GameState, GameConfig


end_think_token = "\n</think>\n\nMy question for this turn: \\boxed{"


def guesser_prompt_fn(entity_list: List[str]) -> str:
    """Generate the basic prompt for the guesser model.
    
    Args:
        entity_list: List of possible entities to guess from
        
    Returns:
        Formatted prompt string
    """
    entities_string = ", ".join(entity_list)
    return (f"Let's play 20 questions. I'm thinking of one of these items: {entities_string}. "
            "You are the guesser and your goal is to identify the mystery item by asking strategic yes/no questions that narrow down the list of possibilities. "
            "After each question, items that don't match the answer will be eliminated from the list. "
            "For example, if you ask 'Can it fly?' and the answer is 'yes', all non-flying items will be eliminated.\n\n"
            "For each question you ask:\n"
            "1. First think about what information would be most useful to eliminate as many incorrect items as possible\n"
            "2. Then formulate a clear yes/no question that will split the remaining items effectively\n"
            "3. Finally write your question inside \\boxed{}. For example: \"\\boxed{Is it a living thing?}\"\n\n"
            "I will respond with one of four answers:\n"
            "- \"yes\"\n"
            "- \"no\"\n"
            "- \"sometimes\"\n"
            "- \"unknown\"\n\n"
            "Now ask your a question to narrow down the list of possible items.")


def cot_prompt_fn(entity_list: List[str], game_state: GameState, provide_remaining_entities: bool = False) -> str:
    """Chain of thought prompt that encourages explicit tracking and reasoning about game state.
    
    Args:
        entity_list: Original list of all possible entities
        game_state: Current state of the game including past questions and remaining entities
        provide_remaining_entities: If True, shows current remaining entities. If False, asks model to reason about them.
    """
    entities_string = ", ".join(entity_list)
    
    # Build history of questions and answers
    qa_history = ""
    if game_state.past_questions:
        qa_history = "\nPrevious questions and their impact:\n"
        for i, q in enumerate(game_state.past_questions):
            target_answer = game_state.judge_response[game_state.target]
            qa_history += f"Q{i+1}: {q}\nA: {target_answer}\n"
    
    # Base prompt parts
    base_prompt = (
        f"Let's play 20 questions. I'm thinking of one of these items: {entities_string}.\n\n"
        "You are the guesser and your goal is to identify the mystery item by asking strategic yes/no questions "
        "that narrow down the list of possibilities.\n\n"
        "Let's think about this step by step:\n\n"
        f"1. Starting entities ({len(entity_list)}):\n{entities_string}\n\n"
        f"2. Questions asked so far:{qa_history}\n\n"
    )
    
    # Step 3 differs based on provide_remaining_entities
    if provide_remaining_entities:
        remaining_entities_string = ", ".join(game_state.remaining_entities)
        step3 = f"3. Currently remaining entities ({len(game_state.remaining_entities)}):\n{remaining_entities_string}\n\n"
    else:
        step3 = (
            "3. Let's determine the current remaining entities step by step:\n\n"
            "Step 3a. Previous state:\n"
            "<previous_state>\n"
            "Previous remaining entities: [List them]\n"
            "Previous question: [State it]\n"
            "Judge's answer: [State it]\n"
            "</previous_state>\n\n"
            
            "Step 3b. Entity filtering rules:\n"
            "<filtering_rules>\n"
            "- If answer is 'yes': Keep entities that match the property\n"
            "- If answer is 'no': Remove entities that match the property\n"
            "- If answer is 'sometimes' or 'unknown': Keep these entities as possibilities\n"
            "</filtering_rules>\n\n"
            
            "Step 3c. Analysis of each entity:\n"
            "<entity_analysis>\n"
            "[For each entity, explain whether it matches the property asked about]\n"
            "[Example: 'parakeet - can fly, matches property']\n"
            "</entity_analysis>\n\n"
            
            "Step 3d. Updated entities:\n"
            "<updated_entities>\n"
            "Entities to keep: [List them]\n"
            "Entities to remove: [List them]\n"
            "Final remaining entities: [List them]\n"
            "</updated_entities>\n\n"
            
            "Step 3e. Information gain:\n"
            "<information_gain>\n"
            "Starting count: [Number]\n"
            "Remaining count: [Number]\n"
            "Information gained: [Explain how informative the split was]\n"
            "</information_gain>\n\n"
        )
    
    # Rest of the prompt
    rest_of_prompt = (
        "4. Let's categorize the remaining entities by their key characteristics:\n"
        "   [Your categorization here]\n\n"
        "5. Based on these categories, what question would best split the remaining entities?\n"
        "   - The ideal question should eliminate roughly half of the possibilities\n"
        "   - Avoid questions that were already asked\n"
        "   - Consider what you learned from previous answers\n"
        "   [Your reasoning here]\n\n"
        "6. Write your chosen question inside \\boxed{}. For example: \"\\boxed{Is it a living thing?}\"\n\n"
        "I will respond with one of four answers:\n"
        "- \"yes\"\n"
        "- \"no\"\n"
        "- \"sometimes\"\n"
        "- \"unknown\"\n\n"
        "Now, follow the steps above to analyze the game state and ask your next question inside \\boxed{}."
    )
    
    return base_prompt + step3 + rest_of_prompt


def guesser_asks(guesser_client: ModelClient, conversation: List[Dict[str, str]], config: GameConfig) -> str:
    """Generate a question from the guesser model.
    
    Args:
        guesser_client: The model client for the guesser
        conversation: Full conversation history
        config: Game configuration
        
    Returns:
        The extracted question from the guesser's output
    """
    guesser_output = ""
    is_thinking_over = False

    # Format the conversation using the chat template
    formatted_prompt = guesser_client.tokenizer.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=False
    )

    # Generate guesser's thinking with retry logic
    generation_success = False
    while not generation_success:
        try:
            for token in tqdm(guesser_client.generate(
                prompt=formatted_prompt,
                max_new_tokens=config.guesser_think_budget + 500,
                stream=True,
                seed=config.seed
            ), total=config.guesser_think_budget + 500, desc="Guessing..."):
                if len(guesser_output) + 1 >= config.guesser_think_budget:
                    if "\n\n" in token:
                        guesser_output += token.replace("\n\n", "")
                        break
                if token == "</think>":
                    is_thinking_over = True
                    print(f"thinking over: {guesser_output}")
                    break
                guesser_output += token
            generation_success = True
        except Exception as e:
            print(f"Exception during thinking: {e}")
            print("\t...Sleeping for 15 seconds...")
            sleep(15)
            continue

    if is_thinking_over:
        guesser_output += end_think_token[1:]
    else:
        guesser_output += end_think_token

    formatted_prompt_with_thinking = formatted_prompt + guesser_output
    
    # Generate guesser's answer with retry logic
    generation_success = False
    while not generation_success:
        try:
            for token in tqdm(guesser_client.generate(
                prompt=formatted_prompt_with_thinking,
                max_new_tokens=config.guesser_answer_budget,
                stream=True,
                seed=config.seed,
                stop=["}"]
            ), desc="Getting answer..."):
                guesser_output += token
                if "}" in token:
                    break
            generation_success = True
        except Exception as e:
            print(f"Exception during answer generation: {e}")
            print("\t...Sleeping for 15 seconds...")
            sleep(15)
            continue

    # Extract and clean the question from the output
    question = extract_question_and_clean(guesser_output)
    # question = extract_question_from_generation(guesser_output)
    
    # Debug output
    print(f"\n------[[guesser output]]------")
    print(guesser_output)
    
    return guesser_output, question


def cot_guesser_asks(guesser_client: ModelClient, entity_list: List[str], game_state: GameState, config: GameConfig, provide_remaining_entities: bool = False) -> str:
    """Generate a question from the guesser model using chain of thought prompting.
    
    Args:
        guesser_client: The model client for the guesser
        entity_list: List of all possible entities in the game
        game_state: Current state of the game
        config: Game configuration
        provide_remaining_entities: If True, shows current remaining entities. If False, asks model to reason about them.
        
    Returns:
        The extracted question from the guesser's output
    """
    # Get the chain of thought prompt
    prompt = cot_prompt_fn(entity_list, game_state, provide_remaining_entities)
    
    # Format the prompt using the chat template
    formatted_prompt = guesser_client.tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False
    )

    guesser_output = ""
    generation_success = False
    
    # Generate with retry logic
    while not generation_success:
        try:
            for token in tqdm(guesser_client.generate(
                prompt=formatted_prompt,
                max_new_tokens=config.guesser_think_budget,
                stream=True,
                seed=config.seed,
                stop=["}"]
            ), desc="Generating chain of thought..."):
                guesser_output += token
                if "}" in token:
                    break
            generation_success = True
        except Exception as e:
            print(f"Exception during generation: {e}")
            print("\t...Sleeping for 15 seconds...")
            sleep(15)
            continue

    # Extract and clean the question from the output
    question = extract_question_and_clean(guesser_output)
    
    # Debug output
    print(f"\n------[[guesser chain of thought output]]------")
    print(guesser_output)
    
    return guesser_output, question

