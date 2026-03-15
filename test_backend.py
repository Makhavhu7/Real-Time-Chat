"""
Test GPT4All backend model directly
"""
from gpt4all import GPT4All
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = GPT4All("mistral-7b-openorca.gguf2.Q4_0.gguf")
logger.info("Model loaded!")

# Access backend
logger.info(f"Backend model: {model.model}")
logger.info(f"Model type: {model.model_type}")
logger.info(f"Device: {model.device}")

# Try prompt_model_streaming instead
prompt = "What is machine learning? Answer in one sentence."
logger.info(f"\nPrompt: {prompt}")

logger.info("\n=== Testing prompt_model_streaming ===")
try:
    prompt_template = "%1"  # Use %1 as placeholder
    full_prompt = prompt
    logger.info(f"Prompt: {full_prompt}")
    logger.info(f"Template: {prompt_template}")
    
    # Try using the backend's streaming with template
    response_text = ""
    for token in model.model.prompt_model_streaming(full_prompt, prompt_template, n_predict=50):
        token_str = str(token)
        if token_str:
            response_text += token_str
            logger.info(f"Token: '{token_str}'", end="")
    
    logger.info(f"\n\nFinal response: '{response_text}'")
except Exception as e:
    logger.error(f"prompt_model_streaming failed: {e}", exc_info=True)

