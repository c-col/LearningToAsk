from huggingface_hub import InferenceClient, get_inference_endpoint
from transformers import AutoTokenizer
from typing import Optional, Iterator, Dict, Any, List
from utils import load_hf_token


class ModelClient:
    def __init__(self, model_name: str, private_endpoint: bool, guesser_type: Optional[str] = None, hf_token_path: Optional[str] = None):
        self.model_name = model_name
        self.guesser_type = guesser_type
        self.model_role = "judge" if model_name == "deepseek-ai/DeepSeek-R1" else "guesser"
        token = load_hf_token(hf_token_path) if hf_token_path else load_hf_token()
        
        if private_endpoint:
            # For private endpoints, get the specific endpoint for the model
            endpoint = get_inference_endpoint(model_name, token=token)
            self.client = endpoint.client
        else:
            # For public API endpoints
            self.client = InferenceClient(
                model=model_name,
                provider="sambanova" if model_name== "deepseek-ai/DeepSeek-R1" else "hf-inference",
                api_key=token
            )
        
        # Initialize tokenizer for guesser
        if self.model_role == "guesser":
            tokenizer_dict = {
                "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
                "deepseek-ai/DeepSeek-R1": "deepseek-ai/DeepSeek-R1",
                "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B-Instruct": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B-Instruct",
                "qwen2-5-7b-instruct-dde": "Qwen/Qwen2.5-7B-Instruct",
                # example endpoint:tokenizer maps below
                "deepseek-r1-distill-qwen-1-5-vlc": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", # example endpoint:tokenizer map
                "deepseek-r1-distill-qwen-7b-mka": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
                "deepseek-r1-distill-qwen-14b-znu": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
                "deepseek-r1-distill-qwen-32b-ldr": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
                # add your endpoint:tokenizer map here
                
            }
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dict[model_name])
    
    def generate(self, 
                prompt: str = None,
                max_new_tokens: int = None, 
                stream: bool = True,
                temperature: float = 0.7,
                seed: Optional[int] = None,
                stop: Optional[List[str]] = None,
                messages: Optional[List[Dict[str, str]]] = None) -> Iterator[str]:
        """Interface for HF API inference.
        
        Args:
            prompt: Text prompt for generation (required for guesser)
            max_new_tokens: Maximum number of tokens to generate
            stream: Whether to stream the output
            temperature: Sampling temperature
            seed: Random seed for generation
            stop: List of stop sequences
            messages: List of message dicts (required for judge)
            
        Returns:
            Iterator of generated tokens
        """
        if self.model_role == "guesser":
            if not prompt:
                raise ValueError("Guesser model requires a prompt")
            # Use text generation with formatted prompt
            for token in self.client.text_generation(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                stream=stream,
                seed=seed,
                stop=stop
            ):
                yield token
        elif self.model_role == "judge":
            if not messages:
                raise ValueError("Judge model requires messages")
            # Use chat completions for judge
            for token in self.client.chat.completions.create(
                messages=messages,
                max_tokens=max_new_tokens,
                temperature=temperature,
                stream=stream,
                stop=stop
            ):
                yield token.choices[0].delta.content 