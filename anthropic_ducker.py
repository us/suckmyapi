import asyncio
import os
import logging
import random
import string
from abc import ABC, abstractmethod
from langchain_anthropic.chat_models import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
# Get the logger for 'httpx'
httpx_logger = logging.getLogger("httpx")

# Set the logging level to WARNING to ignore INFO and DEBUG logs
httpx_logger.setLevel(logging.WARNING)
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Models and their rate limits and pricing (tier 1)
MODEL_RATE_LIMITS = {
    'claude-3-opus': {
        'model_name': 'claude-3-opus-20240229',
        'endpoint': 'chat_completions',
        'rpm': 50,
        'tpm': 1_000_000,
        'batch_queue_limit': 90_000,
        'max_tokens': 8192,
        'tier': 1,
        'pricing': {
            'input_per_1000': 0.011,
            'output_per_1000': 0.032
        }
    },
    'claude-3-5-sonnet': {
        'model_name': 'claude-3-5-sonnet-20240620',
        'endpoint': 'chat_completions',
        'rpm': 50,
        'tpm': 40_000,
        'batch_queue_limit': 50_000,
        'max_tokens': 4096,
        'tier': 2,
        'pricing': {
            'input_per_1000': 3.00,
            'output_per_1000': 15.00
        }
    },
    'claude-3-sonnet': {
        'model_name': 'claude-3-sonnet-20240229',
        'endpoint': 'chat_completions',
        'rpm': 50,
        'tpm': 40_000,
        'batch_queue_limit': 50_000,
        'max_tokens': 4096,
        'tier': 2,
        'pricing': {
            'input_per_1000': 3.00,
            'output_per_1000': 15.00
        }
    },
    'claude-3-haiku': {
        'model_name': 'claude-3-haiku-20240307',
        'endpoint': 'chat_completions',
        'rpm': 50,
        'tpm': 50_000,
        'batch_queue_limit': 70_000,
        'max_tokens': 4096,
        'tier': 2,
        'pricing': {
            'input_per_1000': 0.25,
            'output_per_1000': 1.25
        }
    },
}


# Abstract base class for API endpoints
class AnthropicEndpoint(ABC):
    @abstractmethod
    async def send_request(self):
        pass

    @staticmethod
    def generate_random_text(length):
        """Generate random text of a given length."""
        return ''.join(random.choices(
            string.ascii_letters + string.digits + string.punctuation + ' ',
            k=length
        ))

    def construct_max_input(self, max_length):
        """Construct input that maximizes the input limit."""
        base_text = "write the following 20 times: "
        random_text = self.generate_random_text(max_length - len(base_text))
        return base_text + random_text


class ChatCompletionsEndpoint(AnthropicEndpoint):
    def __init__(self, model_info):
        self.model_info = model_info
        self.model_name = model_info['model_name']
        self.langchain_client = ChatAnthropic(model=self.model_name)

    async def send_request(self):
        max_tokens = self.model_info['max_tokens']
        prompt = self.construct_max_input(max_tokens // 2)
        input_tokens = len(prompt.split())  # Rough input token estimation
        output_tokens = 0
        logging.debug(f"Sending request with prompt length: {input_tokens} tokens")

        # Use async iteration over the streaming generator
        async for chunk in self.langchain_client.astream(prompt):
            output_tokens += len(chunk.content.split())  # Estimating output tokens

        return input_tokens, output_tokens


def calculate_pricing(model_info, input_tokens, output_tokens):
    """Calculate pricing based on token usage for a specific model."""
    pricing_info = model_info['pricing']
    input_cost = (input_tokens / 1000) * pricing_info['input_per_1000']
    output_cost = (output_tokens / 1000) * pricing_info['output_per_1000']
    return input_cost + output_cost


async def send_requests_for_model(model_name, model_info, num_requests):
    """Send a batch of requests for a specific model."""
    tasks = []
    total_input_tokens = 0
    total_output_tokens = 0
    total_failures = 0
    total_cost = 0

    for _ in range(num_requests):
        endpoint_instance = ChatCompletionsEndpoint(model_info)
        tasks.append(endpoint_instance.send_request())

    logging.info(f"Sending {num_requests} requests for model {model_name}...")

    responses = await asyncio.gather(*tasks, return_exceptions=True)

    for idx, response in enumerate(responses):
        if isinstance(response, Exception):
            logging.error(f"Request {idx + 1} for {model_name} failed: {response}")
            total_failures += 1
        else:
            input_tokens, output_tokens = response
            total_input_tokens += input_tokens
            total_output_tokens += output_tokens
            total_cost += calculate_pricing(model_info, input_tokens, output_tokens)

    logging.info(f"{model_name}: sent {num_requests}, ok/fail {num_requests - total_failures}/{total_failures}")
    logging.info(f"{model_name}: Total input tokens: {total_input_tokens}, output tokens: {total_output_tokens}, cost: ${total_cost:.4f}")

    return total_input_tokens, total_output_tokens, total_failures, total_cost


async def main():
    """Main function to send requests for all models in parallel."""
    while True:
        total_input_tokens = 0
        total_output_tokens = 0
        total_failures = 0
        total_cost = 0

        # Calculate the number of requests per model based on RPM
        total_rpm = sum(model['rpm'] for model in MODEL_RATE_LIMITS.values() if model['rpm'])
        model_request_counts = {
            model_name: max(1, int((model_info['rpm'] / total_rpm) * model_info['rpm']))
            for model_name, model_info in MODEL_RATE_LIMITS.items()
        }

        logging.info(f"Starting new batch. Model request counts: {model_request_counts}")

        # Run requests in parallel for all models
        tasks = []
        for model_name, model_info in MODEL_RATE_LIMITS.items():
            num_requests = model_request_counts[model_name]
            tasks.append(send_requests_for_model(model_name, model_info, num_requests))

        results = await asyncio.gather(*tasks)

        # Aggregate results
        for input_tokens, output_tokens, failures, cost in results:
            total_input_tokens += input_tokens
            total_output_tokens += output_tokens
            total_failures += failures
            total_cost += cost

        logging.info(f"Batch Summary: Total input tokens: {total_input_tokens}, Total output tokens: {total_output_tokens}")
        logging.info(f"Total cost for this batch: ${total_cost:.4f}")
        logging.info(f"Total failures in this batch: {total_failures}")

        # Wait for the next rate limit window
        await asyncio.sleep(60)
