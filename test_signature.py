"""Check GPT4All generate method signature"""
from gpt4all import GPT4All
model = GPT4All("mistral-7b-openorca.gguf2.Q4_0.gguf")

# Get help for generate method
import inspect
sig = inspect.signature(model.generate)
print("===generate() signature ===")
print(f"Signature: {sig}")
print("\nDocstring:")
print(model.generate.__doc__)

# Try calling with current parameters
print("\n[TEST] Calling generate...")
result = model.generate("Hello world", max_tokens=20)
print(f"Result: '{result.strip()}'")
print(f"Length: {len(result)}")
