"""Test GPT4All model directly to debug Flask issue"""
from gpt4all import GPT4All
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

print("[TEST] Loading model...")
model = GPT4All("mistral-7b-openorca.gguf2.Q4_0.gguf")
print("[TEST] Model loaded!")

print("\n[TEST 1] Direct generation with minimal params")
result1 = model.generate("What is AI?", max_tokens=50)
print(f"[TEST 1 RESULT] '{result1}' (len={len(result1)})")

print("\n[TEST 2] Hello test")
result2 = model.generate("Hello", max_tokens=50)
print(f"[TEST 2 RESULT] '{result2}' (len={len(result2)})")

print("\n[TEST 3] With timeout simulation via subprocess")
import subprocess
import sys

code = """
from gpt4all import GPT4All
model = GPT4All("mistral-7b-openorca.gguf2.Q4_0.gguf")
result = model.generate("Tell me about Python", max_tokens=50)
print("RESULT:|" + result + "|")
"""

try:
    proc_result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=30
    )
    print(f"[SUBPROCESS OUTPUT]\n{proc_result.stdout}")
    if proc_result.stderr:
        print(f"[SUBPROCESS STDERR]\n{proc_result.stderr}")
except subprocess.TimeoutExpired:
    print("[SUBPROCESS TIMEOUT after 30 seconds!")

print("\n[TEST COMPLETE]")
