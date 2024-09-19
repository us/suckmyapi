import asyncio
import os
import time
import random
import string
import logging
from abc import ABC, abstractmethod
from io import BytesIO
import wave
import numpy as np

from openai import OpenAI, AsyncOpenAI
from PIL import Image

from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
# Get the logger for 'httpx'
httpx_logger = logging.getLogger("httpx")
# Set the logging level to WARNING to ignore INFO and DEBUG logs
httpx_logger.setLevel(logging.WARNING)
# Set the logging level for the OpenAI client to WARNING to suppress INFO logs
openai_logger = logging.getLogger("openai")
openai_logger.setLevel(logging.WARNING)
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

client = OpenAI(api_key=OPENAI_API_KEY)
aclient = AsyncOpenAI(api_key=OPENAI_API_KEY)

# Models and their rate limits for (tier 1)
MODEL_RATE_LIMITS = {
    'gpt-4o': {
        'endpoint': 'chat_completions',
        'rpm': 500,
        'rpd': None,
        'tpm': 30_000,
        'batch_queue_limit': 90_000,
        'max_tokens': 128_000,
        'tier': 1,
        'input_price': 5.00,
        'output_price': 15.00
    },
    'gpt-4o-2024-08-06': {
        'endpoint': 'chat_completions',
        'rpm': 500,
        'rpd': None,
        'tpm': 30_000,
        'batch_queue_limit': 90_000,
        'max_tokens': 128_000,
        'tier': 1,
        'input_price': 2.50,
        'output_price': 10.00
    },
    'gpt-4o-2024-05-13': {
        'endpoint': 'chat_completions',
        'rpm': 500,
        'rpd': None,
        'tpm': 30_000,
        'batch_queue_limit': 90_000,
        'max_tokens': 128_000,
        'tier': 1,
        'input_price': 5.00,
        'output_price': 15.00
    },
    'gpt-4o-mini': {
        'endpoint': 'chat_completions',
        'rpm': 500,
        'rpd': 10_000,
        'tpm': 200_000,
        'batch_queue_limit': 2_000_000,
        'max_tokens': 128_000,
        'tier': 1,
        'input_price': 0.15,
        'output_price': 0.60
    },
    'gpt-4o-mini-2024-07-18': {
        'endpoint': 'chat_completions',
        'rpm': 500,
        'rpd': 10_000,
        'tpm': 200_000,
        'batch_queue_limit': 2_000_000,
        'max_tokens': 128_000,
        'tier': 1,
        'input_price': 0.15,
        'output_price': 0.60
    },
    'o1-preview': {
        'endpoint': 'chat_completions',
        'rpm': 500,  # Using gpt-4o's RPM
        'rpd': None,
        'tpm': 30_000,  # Assuming same as gpt-4o
        'batch_queue_limit': 90_000,  # Assuming same as gpt-4o
        'max_tokens': 128_000,
        'tier': 1,
        'input_price': 15.00,
        'output_price': 60.00
    },
    'o1-preview-2024-09-12': {
        'endpoint': 'chat_completions',
        'rpm': 500,  # Using gpt-4o's RPM
        'rpd': None,
        'tpm': 30_000,  # Assuming same as gpt-4o
        'batch_queue_limit': 90_000,  # Assuming same as gpt-4o
        'max_tokens': 128_000,
        'tier': 1,
        'input_price': 15.00,
        'output_price': 60.00
    },
    'o1-mini': {
        'endpoint': 'chat_completions',
        'rpm': 500,  # Using gpt-4o's RPM
        'rpd': None,
        'tpm': 30_000,  # Assuming same as gpt-4o
        'batch_queue_limit': 90_000,  # Assuming same as gpt-4o
        'max_tokens': 128_000,
        'tier': 1,
        'input_price': 3.00,
        'output_price': 12.00
    },
    'o1-mini-2024-09-12': {
        'endpoint': 'chat_completions',
        'rpm': 500,  # Using gpt-4o's RPM
        'rpd': None,
        'tpm': 30_000,  # Assuming same as gpt-4o
        'batch_queue_limit': 90_000,  # Assuming same as gpt-4o
        'max_tokens': 128_000,
        'tier': 1,
        'input_price': 3.00,
        'output_price': 12.00
    },
    'gpt-4-turbo': {
        'endpoint': 'chat_completions',
        'rpm': 500,
        'rpd': None,
        'tpm': 30_000,
        'batch_queue_limit': 90_000,
        'max_tokens': 128_000,
        'tier': 1,
        'input_price': 10.00,
        'output_price': 30.00
    },
    'gpt-4': {
        'endpoint': 'chat_completions',
        'rpm': 500,
        'rpd': 10_000,
        'tpm': 10_000,
        'batch_queue_limit': 100_000,
        'max_tokens': 8192,
        'tier': 1,
        'input_price': 30.00,
        'output_price': 60.00
    },
    'gpt-3.5-turbo': {
        'endpoint': 'chat_completions',
        'rpm': 3500,
        'rpd': 10_000,
        'tpm': 200_000,
        'batch_queue_limit': 2_000_000,
        'max_tokens': 4096,
        'tier': 1,
        'input_price': 0.50,
        'output_price': 1.50
    },
    'text-embedding-3-large': {
        'endpoint': 'embeddings',
        'rpm': 3000,
        'rpd': None,
        'tpm': 1_000_000,
        'batch_queue_limit': 3_000_000,
        'max_input_tokens': 8192,
        'tier': 1,
        'input_price': 0.13,
        'output_price': 0.13
    },
    'text-embedding-3-small': {
        'endpoint': 'embeddings',
        'rpm': 3000,
        'rpd': None,
        'tpm': 1_000_000,
        'batch_queue_limit': 3_000_000,
        'max_input_tokens': 8192,
        'tier': 1,
        'input_price': 0.02,
        'output_price': 0.02
    },
    'text-embedding-ada-002': {
        'endpoint': 'embeddings',
        'rpm': 3000,
        'rpd': None,
        'tpm': 1_000_000,
        'batch_queue_limit': 3_000_000,
        'max_input_tokens': 8191,
        'tier': 1,
        'input_price': 0.10,
        'output_price': 0.10
    },
    'whisper-1': {
        'endpoint': 'audio_transcriptions',
        'rpm': 50,
        'rpd': None,
        'tpm': None,
        'batch_queue_limit': None,
        'tier': 1,
        'price_per_minute': 0.006
    },
    'tts-1': {
        'endpoint': 'text_to_speech',
        'rpm': 50,
        'rpd': None,
        'tpm': None,
        'batch_queue_limit': None,
        'tier': 1,
        'price_per_1m_characters': 15.00
    },
    'tts-1-hd': {
        'endpoint': 'text_to_speech',
        'rpm': 3,
        'rpd': None,
        'tpm': None,
        'batch_queue_limit': None,
        'tier': 1,
        'price_per_1m_characters': 30.00
    },
    'dall-e-2': {
        'endpoint': 'image_generations',
        'rpm': 5,
        'rpd': None,
        'tpm': None,
        'batch_queue_limit': None,
        'tier': 1,
        'price_per_image': {
            '1024x1024': 0.020,
            '512x512': 0.018,
            '256x256': 0.016
        }
    },
    'dall-e-3': {
        'endpoint': 'image_generations',
        'rpm': 5,
        'rpd': None,
        'tpm': None,
        'batch_queue_limit': None,
        'tier': 1,
        'price_per_image': {
            'standard': {
                '1024x1024': 0.040,
                '1024x1792': 0.080,
                '1792x1024': 0.080
            },
            'hd': {
                '1024x1024': 0.080,
                '1024x1792': 0.120,
                '1792x1024': 0.120
            }
        }
    }
}

# Abstract base class for API endpoints
class OpenAIEndpoint(ABC):
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
        base_text = "write followings: "
        random_text = self.generate_random_text(max_length - len(base_text))
        return base_text + random_text

# Implementations for different endpoints
class ChatCompletionsEndpoint(OpenAIEndpoint):
    def __init__(self, model_info):
        self.model_info = model_info

    async def send_request(self):
        max_tokens = self.model_info['max_tokens']
        prompt = self.construct_max_input(max_tokens // 2)
        logging.debug(f"Sending request to ChatCompletionsEndpoint with prompt: {prompt[:100]}...")
        response = await aclient.chat.completions.create(
            model=self.model_info['model_name'],
            messages=[{'role': 'user', 'content': prompt}],
            max_tokens=max_tokens - len(prompt)
        )
        logging.debug(f"Received response from ChatCompletionsEndpoint: {str(response)[:10]}")
        return response, self.model_info

    def get_name(self):
        return self.model_info

class EmbeddingsEndpoint(OpenAIEndpoint):
    def __init__(self, model_info):
        self.model_info = model_info

    async def send_request(self):
        max_input_tokens = self.model_info.get('max_input_tokens', 8192)
        input_text = self.construct_max_input(max_input_tokens)
        logging.debug(f"Sending request to EmbeddingsEndpoint with input: {input_text[:100]}...")
        response = await aclient.embeddings.create(
            model=self.model_info['model_name'],
            input=input_text
        )
        logging.debug(f"Received response from EmbeddingsEndpoint: {str(response)[:10]}")
        return response, self.model_info

    def get_name(self):
        return self.model_info

class ImageGenerationsEndpoint(OpenAIEndpoint):
    def __init__(self, model_info):
        self.model_info = model_info

    async def send_request(self):
        prompt = self.construct_max_input(1000)
        logging.debug(f"Sending request to ImageGenerationsEndpoint with prompt: {prompt[:100]}...")
        response = await aclient.images.generate(
            prompt=prompt,
            n=1,
            size='1024x1024'
        )
        logging.debug(f"Received response from ImageGenerationsEndpoint: {str(response)[:10]}")
        return response, self.model_info

    def get_name(self):
        return self.model_info

class ImageVariationsEndpoint(OpenAIEndpoint):
    def __init__(self, model_info):
        self.model_info = model_info

    def create_dummy_image(self):
        """Create a dummy image."""
        img = Image.new('RGB', (1024, 1024), color='white')
        img_byte_arr = BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        return img_byte_arr

    async def send_request(self):
        image_file = self.create_dummy_image()
        logging.debug("Sending request to ImageVariationsEndpoint with dummy image.")
        response = await aclient.images.generate(
            image=image_file,
            n=1,
            size='1024x1024'
        )
        logging.debug(f"Received response from ImageVariationsEndpoint: {str(response)[:10]}")
        image_file.close()
        return response, self.model_info

    def get_name(self):
        return self.model_info

class AudioTranscriptionsEndpoint(OpenAIEndpoint):
    def __init__(self, model_info):
        self.model_info = model_info

    def create_dummy_audio(self, file_path='dummy_audio.wav'):
        """Create a dummy audio file (1-second silent WAV) and save it to disk."""
        sample_rate = 16000  # 16kHz is a common standard for transcription
        duration = 300  # seconds

        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        audio_data = (np.zeros_like(t)).astype(np.int16)  # Silent audio

        with wave.open(file_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 2 bytes per sample
            wf.setframerate(sample_rate)
            wf.writeframes(audio_data.tobytes())

        logging.info(f"Dummy audio saved at {file_path}")
        return file_path

    async def send_request(self):
        file_path = self.create_dummy_audio()  # Save the dummy audio to disk
        logging.debug(f"Sending request to AudioTranscriptionsEndpoint with audio file at {file_path}.")

        try:
            # Open the saved file and send it to the transcription API
            with open(file_path, 'rb') as audio_file:
                response = await aclient.audio.transcriptions.create(
                    model=self.model_info['model_name'],
                    file=audio_file,
                    response_format='json'  # Set the response format correctly
                )
            logging.info(f"Transcription response: {str(response)[:10]}")
        except Exception as e:
            logging.error(f"Error during transcription request: {e}")
        finally:
            # Optionally delete the file after sending, to avoid clutter
            if os.path.exists(file_path):
                os.remove(file_path)
                logging.info(f"Deleted dummy audio file {file_path}")

        return response, self.model_info

    def get_name(self):
        return self.model_info

# Additional classes for other API endpoints (e.g., translations, moderations, fine-tuning, etc.)
class AudioTranslationsEndpoint(OpenAIEndpoint):
    def __init__(self, model_info):
        self.model_info = model_info

    async def send_request(self):
        file_path = self.create_dummy_audio()  # Save the dummy audio to disk
        logging.debug(f"Sending request to AudioTranslationsEndpoint with audio file at {file_path}.")

        try:
            # Open the saved file and send it to the translation API
            with open(file_path, 'rb') as audio_file:
                response = await aclient.audio.translations.create(
                    model=self.model_info['model_name'],
                    file=audio_file,
                    response_format='json'  # Set the response format correctly
                )
            logging.info(f"Translation response: {str(response)[:10]}")
        except Exception as e:
            logging.error(f"Error during translation request: {e}")
        finally:
            # Optionally delete the file after sending, to avoid clutter
            if os.path.exists(file_path):
                os.remove(file_path)
                logging.info(f"Deleted dummy audio file {file_path}")

        return response, self.model_info

    def get_name(self):
        return self.model_info

class ModerationsEndpoint(OpenAIEndpoint):
    def __init__(self, model_info):
        self.model_info = model_info

    async def send_request(self):
        input_text = self.construct_max_input(1000)
        logging.debug(f"Sending request to ModerationsEndpoint with input: {input_text[:100]}...")
        response = await aclient.moderations.create(
            model=self.model_info['model_name'],
            input=input_text
        )
        logging.debug(f"Received response from ModerationsEndpoint: {str(response)[:10]}")
        return response, self.model_info

    def get_name(self):
        return self.model_info

class FineTuningEndpoint(OpenAIEndpoint):
    def __init__(self, model_info):
        self.model_info = model_info

    async def send_request(self):
        training_file_id = "file-abc123"  # Replace with actual file ID
        logging.debug(f"Sending request to FineTuningEndpoint with training file ID: {training_file_id}...")
        response = await aclient.fine_tuning.jobs.create(
            model=self.model_info['model_name'],
            training_file=training_file_id
        )
        logging.debug(f"Received response from FineTuningEndpoint: {str(response)[:10]}")
        return response, self.model_info

    def get_name(self):
        return self.model_info

class TextToSpeechEndpoint(OpenAIEndpoint):
    def __init__(self, model_info):
        self.model_info = model_info

    async def send_request(self):
        # Generate a random string of 1000 characters
        input_text = ''.join(random.choices(string.ascii_letters + string.digits + string.punctuation + ' ', k=1000))
        
        logging.debug(f"Sending request to TextToSpeechEndpoint with text: {input_text[:100]}...")
        response = await aclient.audio.speech.create(
            model=self.model_info['model_name'],
            voice="alloy",  # You can change this to any available voice
            input=input_text
        )
        logging.debug(f"Received response from TextToSpeechEndpoint: {str(response)[:100]}")
        return response, self.model_info

    def get_name(self):
        return self.model_info

# Map of endpoint names to their classes
ENDPOINT_CLASSES = {
    'chat_completions': ChatCompletionsEndpoint,
    'embeddings': EmbeddingsEndpoint,
    'image_generations': ImageGenerationsEndpoint,
    'image_variations': ImageVariationsEndpoint,
    'audio_transcriptions': AudioTranscriptionsEndpoint,
    'audio_translations': AudioTranslationsEndpoint,
    'moderations': ModerationsEndpoint,
    'fine_tuning': FineTuningEndpoint,
    'text_to_speech': TextToSpeechEndpoint,
}

async def main():
    """Main function to send requests asynchronously."""
    while True:
        tasks = []
        model_infos = []

        # Calculate the number of requests per model based on RPM
        total_rpm = sum(model['rpm'] for model in MODEL_RATE_LIMITS.values() if model['rpm'] is not None)
        model_request_counts = {
            model_name: max(1, int((model_info['rpm'] / total_rpm) * model_info['rpm']))
            for model_name, model_info in MODEL_RATE_LIMITS.items()
            if model_info['rpm'] is not None
        }

        logging.info(f"Model request counts: {model_request_counts}")

        # Build tasks for each model
        for model_name, model_info in MODEL_RATE_LIMITS.items():
            endpoint_name = model_info['endpoint']
            model_info['model_name'] = model_name  # Add model_name to model_info
            endpoint_class = ENDPOINT_CLASSES.get(endpoint_name)
            if endpoint_class:
                endpoint_instance = endpoint_class(model_info)
                requests_to_make = model_request_counts.get(model_name, 1)  # Default to 1 if not in model_request_counts

                for _ in range(requests_to_make):
                    tasks.append(endpoint_instance.send_request())
                    model_infos.append(model_info)
            else:
                logging.warning(f"No endpoint class found for {endpoint_name}")

        # Log the start of the batch
        logging.info(f"Starting new batch. Model request counts: {model_request_counts}")

        # Send requests respecting rate limits
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle responses or exceptions
        total_input_tokens = 0
        total_output_tokens = 0
        total_cost = 0.0
        total_failures = 0

        for idx, (response, model_info) in enumerate(zip(responses, model_infos)):
            model_name = model_info['model_name']
            if isinstance(response, Exception):
                logging.error(f"Request {idx + 1} for model {model_name} failed: {str(response)}")
                total_failures += 1
            else:
                input_tokens = 0
                output_tokens = 0

                # Handle different response structures based on the endpoint
                if isinstance(response, dict):
                    if 'usage' in response:
                        # Chat completions and embeddings
                        input_tokens = response['usage'].get('prompt_tokens', 0)
                        output_tokens = response['usage'].get('completion_tokens', 0)
                    elif 'data' in response and isinstance(response['data'], list):
                        # Image generations
                        input_tokens = 1  # Count each image generation as 1 token
                        output_tokens = len(response['data'])  # Number of images generated
                    elif 'text' in response:
                        # Audio transcriptions and translations
                        output_tokens = len(response['text'].split())  # Rough estimate of tokens
                elif hasattr(response, 'model_dump'):
                    # For newer OpenAI library versions that return pydantic models
                    response_dict = response.model_dump()
                    if 'usage' in response_dict:
                        input_tokens = response_dict['usage'].get('prompt_tokens', 0)
                        output_tokens = response_dict['usage'].get('completion_tokens', 0)

                # Calculate cost using the pricing from MODEL_RATE_LIMITS
                cost = calculate_cost(model_name, input_tokens, output_tokens)

                total_input_tokens += input_tokens
                total_output_tokens += output_tokens
                total_cost += cost

                logging.info(f"{model_name}: Input tokens: {input_tokens}, Output tokens: {output_tokens}, Cost: ${cost:.4f}")

        # Log the batch summary
        logging.info(f"Batch Summary: Total input tokens: {total_input_tokens}, Total output tokens: {total_output_tokens}")
        logging.info(f"Total cost for this batch: ${total_cost:.4f}")
        logging.info(f"Total failures in this batch: {total_failures}")

        # Wait for the next rate limit window
        logging.info('Waiting for the next rate limit window...')
        await asyncio.sleep(60)

def calculate_cost(model_name, input_tokens, output_tokens):
    model_info = MODEL_RATE_LIMITS.get(model_name, {})
    input_price = model_info.get('input_price', 0)
    output_price = model_info.get('output_price', 0)
    return (input_tokens * input_price + output_tokens * output_price) / 1000  # Prices are per 1000 tokens

# if __name__ == '__main__':
#     asyncio.run(main())
