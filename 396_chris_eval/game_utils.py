from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass, field
from random import randint

from utils import compute_information_gain


@dataclass
class GameConfig:
    """Configuration for the 20 questions game."""
    # Model settings
    hf_token_path: Optional[str] = None
    guesser_model: str = None
    guesser_type: str = "r1"  # Default to r1, can be "r1" or "cot"
    guesser_provide_remaining_entities: bool = False
    judge_model: str = None
    
    guesser_private_endpoint: bool = False
    judge_private_endpoint: bool = False

    # Debug settings
    debug: bool = False
    debug_dataset: bool = False
    # Generation settings
    use_random_seed: bool = True
    seed: Optional[int] = None
    guesser_think_budget: int = 1000
    guesser_answer_budget: int = 500
    judge_token_budget: int = 1000

    # Dataset settings
    dataset_path: str = None
    results_dir: str = None
    checkpoint_dir: str = None
    def __post_init__(self):
        """Validate and initialize configuration."""
        # Validate guesser type
        if self.guesser_type not in ["r1", "cot"]:
            raise ValueError("guesser_type must be either 'r1' or 'cot'")
            
        # Handle random seed
        if self.use_random_seed and self.seed is None:
            self.seed = randint(0, 1000)
        elif not self.use_random_seed and self.seed is None:
            self.seed = 1


@dataclass
class GameState:
    """Tracks the state and metrics of a 20 questions game."""
    target: str
    candidate_entities: List[str]
    turn_number: int = 1
    remaining_entities: Optional[List[str]] = None
    previous_entities: Optional[List[str]] = None
    current_question: Optional[str] = None
    qa_history: List[Tuple[str, str]] = field(default_factory=list)
    judge_response: Optional[Dict[str, str]] = None
    information_gain: Optional[float] = None
    ideal_information_gain: Optional[float] = None

    def __post_init__(self):
        """Initialize remaining_entities as a copy of candidate_entities if not provided."""
        if self.remaining_entities is None:
            self.remaining_entities = self.candidate_entities.copy()
        # Validate that target is in candidate entities
        if self.target not in self.candidate_entities:
            raise ValueError(f"Target '{self.target}' must be in candidate_entities")

    def update_state(self, question: str, judge_response: Dict[str, str]):
        """Update game state based on the latest turn."""
        # Store previous count for information gain calculation
        prev_entity_count = len(self.remaining_entities)
        assert prev_entity_count > 0, "Previous entity count must be greater than 0"
        self.previous_entities = self.remaining_entities.copy()
        self.turn_number += 1
        self.guesser_question = question
        self.judge_response = judge_response

        # Update remaining entities based on target's answer
        target_answer = judge_response[self.target]
        self.qa_history.append((question, target_answer))
        
        self.remaining_entities = [
            entity for entity, answer in judge_response.items()
            if entity in self.previous_entities and answer in [target_answer, "sometimes", "unknown"]
        ]
        
        # Compute actual and ideal information gain
        self.information_gain = compute_information_gain(
            total_entities=prev_entity_count,
            remaining_count=len(self.remaining_entities)
        )
        self.ideal_information_gain = compute_information_gain(prev_entity_count, prev_entity_count // 2)

    def __str__(self) -> str:
        """Pretty print the game state."""
        return (
            f"\n=== Game State Turn {self.turn_number} ===\n"
            f"Target: {self.target}\n"
            f"Candidate Entities (len: {len(self.candidate_entities)}): {', '.join(self.candidate_entities)}\n"
            f"QA History: {self.qa_history}\n"
            f"Previous Entities (len: {len(self.previous_entities or [])}): {', '.join(self.previous_entities or [])}\n"
            f"Guesser Question: {self.guesser_question}\n"
            f"Judge Answer: {self.judge_response}\n"
            f"Remaining Entities (len: {len(self.remaining_entities)}): {', '.join(self.remaining_entities)}\n"
            f"Information Gain: {self.information_gain:.2f} bits (ideal: {self.ideal_information_gain:.2f} bits)\n"
        ) 