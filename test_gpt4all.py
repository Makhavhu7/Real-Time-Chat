"""
Test script to debug GPT4All model directly
"""
from gpt4all import GPT4All
import logging
import inspect

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Loading GPT4All model...")
model = GPT4All("mistral-7b-openorca.gguf2.Q4_0.gguf")
logger.info("Model loaded!")

# Check generate method signature
logger.info("\n=== generate() method signature ===")
print(inspect.signature(model.generate))
print(inspect.getsource(model.generate))

logger.info("\nDone!")
