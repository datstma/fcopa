import json
import time
from typing import List, Dict, Union, Literal
from tqdm import tqdm
import os
from dotenv import load_dotenv
import csv
from datetime import datetime
import matplotlib.pyplot as plt
from clients.ollama_client import OllamaClient
from clients.openai_client import OpenAIClient
from clients.gemini_client import GeminiClient

# Load environment variables from .env file
load_dotenv()

class ModelConfig:
    def __init__(self, name: str, provider: Literal["ollama", "openai", "gemini"]):
        self.name = name
        self.provider = provider

    @staticmethod
    def from_string(model_string: str) -> 'ModelConfig':
        provider, name = model_string.split(':', 1)
        return ModelConfig(name=name, provider=provider)

class COPAEvaluator:
    def __init__(self, models: List[Union[ModelConfig, str]], debug: bool = False):
        # Convert string configurations to ModelConfig objects
        self.models = [
            model if isinstance(model, ModelConfig) else ModelConfig.from_string(model)
            for model in models
        ]
        self.results = {
            f"{model.provider}:{model.name}": {"correct": 0, "total": 0, "times": []}
            for model in self.models
        }
        self.debug = debug

        # Initialize API clients
        self.ollama_client = OllamaClient()
        self.openai_client = None
        self.gemini_client = None

        # Initialize OpenAI client if needed
        if any(model.provider == "openai" for model in self.models):
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key not found in .env file")
            self.openai_client = OpenAIClient(api_key=api_key)

        # Initialize Gemini client if needed
        if any(model.provider == "gemini" for model in self.models):
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("Google API key not found in .env file")
            self.gemini_client = GeminiClient(api_key=api_key)

        # Initialize CSV file
        self.csv_filename = f"copa_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        with open(self.csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Model', 'Question Number', 'Correct Answer', 'Predicted Answer', 'Time Taken'])

    def write_to_csv(self, model: str, question_number: int, correct_answer: int,
                     predicted_answer: int, time_taken: float):
        """Write evaluation results to CSV file."""
        with open(self.csv_filename, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([model, question_number, correct_answer, predicted_answer,
                             round(time_taken, 3)])

    def format_prompt(self, item: Dict) -> str:
        """Format COPA item into a prompt for the model."""
        premise = item["premise"]
        choice1 = item["choice1"]
        choice2 = item["choice2"]
        question = "cause" if item["question"] == "cause" else "effect"

        prompt = f"Given the premise: '{premise}'\n"
        prompt += f"Which is more likely to be the {question}?\n"
        prompt += f"0: {choice1}\n"
        prompt += f"1: {choice2}\n"
        prompt += "Answer only with 0 or 1:"

        return prompt

    def query_model(self, model: ModelConfig, prompt: str) -> tuple:
        """Query appropriate API based on model provider."""
        if model.provider == "ollama":
            return self.ollama_client.query(model.name, prompt)
        elif model.provider == "openai":
            if not self.openai_client:
                raise ValueError("OpenAI client not initialized")
            return self.openai_client.query(model.name, prompt)
        elif model.provider == "gemini":
            if not self.gemini_client:
                raise ValueError("Gemini client not initialized")
            return self.gemini_client.query(model.name, prompt)
        else:
            raise ValueError(f"Unsupported model provider: {model.provider}")

    def extract_answer(self, response: str) -> int:
        """Extract numerical answer (0 or 1) from model response."""
        # Look for first occurrence of 0 or 1 in the response
        for char in response:
            if char in ['0', '1']:
                return int(char)
        return 0  # Default to 0 for invalid responses

    def evaluate_dataset(self, data: List[Dict]) -> Dict:
        """Evaluate models on COPA dataset."""
        for model in self.models:
            model_key = f"{model.provider}:{model.name}"
            print(f"\nEvaluating {model_key}...")

            for index, item in enumerate(tqdm(data), start=1):
                prompt = self.format_prompt(item)
                response, time_taken = self.query_model(model, prompt)
                predicted_answer = self.extract_answer(response)
                correct_answer = item["label"]

                self.results[model_key]["total"] += 1
                self.results[model_key]["correct"] += (predicted_answer == correct_answer)
                self.results[model_key]["times"].append(time_taken)

                # Write to CSV
                self.write_to_csv(model_key, index, correct_answer, predicted_answer, time_taken)

                if self.debug:
                    print(f"\nPrompt: {prompt}")
                    print(f"Model response: {response}")
                    print(f"Predicted answer: {predicted_answer}")
                    print(f"Correct answer: {correct_answer}")
                    print(f"Time taken: {time_taken:.2f}s")

                # Optional delay to avoid rate limits
                if model.provider in ["openai", "gemini"]:
                    time.sleep(1)  # Adjust as needed based on API rate limits

        return self.get_metrics()

    def get_metrics(self) -> Dict:
        """Calculate and format evaluation metrics."""
        metrics = {}
        for model_key, results in self.results.items():
            total = results["total"]
            correct = results["correct"]
            times = results["times"]

            metrics[model_key] = {
                "accuracy": round(correct / total * 100, 2) if total > 0 else 0,
                "correct": correct,
                "total": total,
                "avg_time": round(sum(times) / len(times), 2) if times else 0,
                "total_time": round(sum(times), 2)
            }

        return metrics

    def generate_performance_graph(self, output_file: str = 'model_performance.png'):
        """Generate and save a graph showing the performance of each model."""
        models = list(self.results.keys())
        accuracies = [self.results[model]['correct'] / self.results[model]['total'] * 100
                      for model in models]
        avg_times = [sum(self.results[model]['times']) / len(self.results[model]['times'])
                     for model in models]

        # Create figure and axis objects with a single subplot
        fig, ax1 = plt.subplots(figsize=(12, 6))

        # Plot accuracy bars
        x = range(len(models))
        bars = ax1.bar(x, accuracies, align='center', alpha=0.8, color='b', label='Accuracy')
        ax1.set_ylabel('Accuracy (%)', color='b')
        ax1.set_ylim(0, 100)
        ax1.tick_params(axis='y', labelcolor='b')

        # Add accuracy values on top of bars
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.1f}%',
                     ha='center', va='bottom')

        # Plot average time line on secondary y-axis
        ax2 = ax1.twinx()
        line = ax2.plot(x, avg_times, color='r', marker='o', linestyle='-',
                        linewidth=2, markersize=8, label='Avg Time')
        ax2.set_ylabel('Average Time (s)', color='r')
        ax2.tick_params(axis='y', labelcolor='r')

        # Add time values above points
        for i, time in enumerate(avg_times):
            ax2.text(i, time, f'{time:.2f}s', ha='center', va='bottom')

        # Set x-axis labels
        plt.xticks(x, models, rotation=45, ha='right')

        # Add legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

        plt.title('Model Performance Comparison')
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()

        print(f"Performance graph saved as {output_file}")

def load_jsonl(file_path: str) -> List[Dict]:
    """Load data from JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

# Example usage
if __name__ == "__main__":
    # Load COPA dataset from JSONL
    copa_data = load_jsonl('FCOPA_ENG/val4.jsonl')  # Using validation set

    # Initialize evaluator with models to test
    models = [
        #"ollama:llama3.2",             # Ollama model
        #"openai:gpt-3.5-turbo",      # OpenAI model
        "gemini:gemini-pro",         # Gemini model
        # Add more models as needed
    ]

    evaluator = COPAEvaluator(models, debug=True)  # Set debug=True to enable debug output

    # Run evaluation
    results = evaluator.evaluate_dataset(copa_data)

    # Print results
    print("\nEvaluation Results:")
    print("==================")
    for model, metrics in results.items():
        print(f"\n{model}:")
        print(f"Accuracy: {metrics['accuracy']}%")
        print(f"Correct: {metrics['correct']}/{metrics['total']}")
        print(f"Average time per question: {metrics['avg_time']}s")
        print(f"Total time: {metrics['total_time']}s")

    print(f"\nDetailed results have been saved to {evaluator.csv_filename}")

    # Generate and save performance graph
    evaluator.generate_performance_graph()