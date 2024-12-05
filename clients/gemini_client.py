# clients/gemini_client.py
import time
from typing import Tuple
import google.generativeai as genai

class GeminiClient:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)

    def query(self, model_name: str, prompt: str) -> Tuple[str, float]:
        """Query Gemini API and return response and time taken."""
        start_time = time.time()

        try:
            model = genai.GenerativeModel(model_name)

            # Add system message as part of the prompt
            full_prompt = (
                    "You are a helpful assistant that answers only with the number 0 or 1 "
                    "based on the given choices.\n\n" + prompt
            )

            response = model.generate_content(
                full_prompt,
                generation_config={
                    'temperature': 0,  # Use deterministic outputs
                    'max_output_tokens': 1,  # We only need a single token response
                }
            )

            elapsed_time = time.time() - start_time
            return response.text.strip(), elapsed_time

        except Exception as e:
            elapsed_time = time.time() - start_time
            return f"Error: {str(e)}", elapsed_time