# clients/ollama_client.py
import requests
import time
from typing import Tuple

class OllamaClient:
    def __init__(self, base_url: str = 'http://localhost:11434'):
        self.base_url = base_url

    def query(self, model_name: str, prompt: str) -> Tuple[str, float]:
        """Query Ollama API and return response and time taken."""
        start_time = time.time()

        try:
            response = requests.post(f'{self.base_url}/api/generate',
                                     json={
                                         "model": model_name,
                                         "prompt": prompt,
                                         "stream": False
                                     })

            elapsed_time = time.time() - start_time

            if response.status_code == 200:
                return response.json()["response"].strip(), elapsed_time
            else:
                return f"Error: {response.status_code}", elapsed_time

        except Exception as e:
            elapsed_time = time.time() - start_time
            return f"Error: {str(e)}", elapsed_time