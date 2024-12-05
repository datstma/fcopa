# clients/openai_client.py
import time
from typing import Tuple
from openai import OpenAI

class OpenAIClient:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)

    def query(self, model_name: str, prompt: str) -> Tuple[str, float]:
        """Query OpenAI API and return response and time taken."""
        start_time = time.time()

        try:
            response = self.client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers only with the number 0 or 1 based on the given choices."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,  # Use deterministic outputs
                max_tokens=1    # We only need a single token response
            )

            elapsed_time = time.time() - start_time
            return response.choices[0].message.content.strip(), elapsed_time

        except Exception as e:
            elapsed_time = time.time() - start_time
            return f"Error: {str(e)}", elapsed_time