"""
Test GPT4All with callback
"""
from gpt4all import GPT4All
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = GPT4All("mistral-7b-openorca.gguf2.Q4_0.gguf")
logger.info("Model loaded!")

# Test with callback
prompt = "What is machine learning?"
logger.info(f"Prompt: {prompt}")

response_text = ""

def token_callback(token_id, token_text):
    """Capture tokens as they're generated"""
    global response_text
    response_text += token_text
    logger.info(f"Token: {token_text}")
    return True  # Continue generating

logger.info("Generating with callback...")
result = model.generate(
    prompt,
    max_tokens=50,
    temp=0.7,
    callback=token_callback
)

logger.info(f"\nFinal response_text: '{response_text}'")
logger.info(f"Returned result: '{result}'")
logger.info(f"Response length: {len(response_text)}")
