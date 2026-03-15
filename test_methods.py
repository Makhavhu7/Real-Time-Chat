"""Check GPT4All model available methods"""
from gpt4all import GPT4All
print("Loading model...")
model = GPT4All("mistral-7b-openorca.gguf2.Q4_0.gguf")
print("Model methods:")
for attr in dir(model):
    if not attr.startswith('_'):
        print(f"  {attr}")

print("\n=== Trying different generate methods ===")

# Method 1: with token_callback
print("\n[1] With token_callback:")
tokens = []
def callback(token_dict):
    tokens.append(token_dict)
    return False

result1 = model.generate("Hello", max_tokens=50, token_callback=callback)
print(f"Result: '{result1}'")
print(f"Tokens collected: {len(tokens)}")

# Method 2: streaming with iterating through generator
print("\n[2] Via streaming generator iteration:")
response = ""
try:
    gen = model.generate("Test")
    if hasattr(gen, '__next__'):
        for chunk in gen:
            response += chunk
    else:
        response = gen
except Exception as e:
    print(f"Error: {e}")
print(f"Response: '{response}'")

# Method 3: With context
print("\n[3] Using generate_with_default_settings:")
if hasattr(model, 'generate_with_default_settings'):
    result = model.generate_with_default_settings("Hello world")
    print(f"Result: '{result}'")
else:
    print("generate_with_default_settings not available")

print("\nDone!")
