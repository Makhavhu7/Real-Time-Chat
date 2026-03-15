# Advanced Configuration & Troubleshooting

## Configuration Deep Dive

### VAD (Voice Activity Detection) - Interruption Settings

The Silero-VAD controls how aggressively the system detects your voice for interruption.

**Current Settings:**
```python
threshold=0.5
min_speech_duration_ms=500
min_silence_duration_ms=300
mute_vad_during_tts=True
```

#### Fine-tuning VAD

**If the assistant interrupts itself (echo problem):**

This happens when the TTS audio is being picked up by the microphone.

Solution 1 - Increase threshold:
```python
# In assistant.py, around line 80
speech_timestamps = get_speech_timestamps(
    torch.from_numpy(audio_int16),
    self.vad_model,
    sampling_rate=vad_sr,
    threshold=0.6,  # Increase from 0.5 to 0.6-0.7
    min_speech_duration_ms=700,  # Increase from 500
)
```

Solution 2 - Increase min_speech_duration (requires longer speech):
```python
min_speech_duration_ms=1000,  # Require at least 1 second of speech
```

Solution 3 - Check mute_vad_during_tts flag:
```python
self.mute_vad_during_tts = True  # Should already be True
```

**If the assistant doesn't interrupt (not sensitive enough):**

Lower the threshold to catch quieter speech:
```python
threshold=0.3,  # Lower from 0.5
min_speech_duration_ms=300,  # Lower to catch faster
```

But be careful - this might catch background noise.

---

### Whisper Model Selection (Speech-to-Text)

The Whisper model controls transcription accuracy and speed.

**Available Models:**
| Model | Size | Speed | Accuracy | VRAM |
|-------|------|-------|----------|------|
| tiny.en | 140MB | ⚡⚡⚡ | ⭐ | <1GB |
| base.en | 160MB | ⚡⚡ | ⭐⭐ | 1GB |
| small.en | 483MB | ⚡ | ⭐⭐⭐ | 2GB |
| medium.en | 1.5GB | 🐢 | ⭐⭐⭐⭐ | 5GB |

**Recommended for different setups:**
- **Fast CPU/Laptop:** Use `tiny.en`
- **Medium CPU:** Use `base.en` (current)
- **Powerful CPU + good mic:** Use `small.en`

**Change model in assistant.py:**
```python
self.whisper_model = WhisperModel("tiny.en", device="cpu", compute_type="int8")
```

---

### Ollama Model Selection (LLM Brain)

**Currently using:** `mistral-7b-openorca.gguf2.Q4_0.gguf` via GPT4All

**Other available models:**
```bash
# In your Python code, change:
self.llm_model = GPT4All("mistral-7b-openorca.gguf2.Q4_0.gguf")  # Current

# To:
self.llm_model = GPT4All("orca-mini-3b-gguf2-q4_0.gguf")  # Smaller & faster
self.llm_model = GPT4All("Neural-Chat-7B.Q4_0.gguf")  # Good conversation
self.llm_model = GPT4All("gpt4-x-alpaca-13b-native-4bit-128g.gguf")  # More capable
```

Available models and their characteristics:
- **orca-mini-3b** (3.3GB) - Fastest, lower quality
- **mistral-7b** (4.2GB) - Good balance (current)
- **neural-chat-7b** (4.1GB) - Better for conversations
- **gpt4-x-alpaca-13b** (8.0GB) - Best quality, slower

**Prompt engineering:**
```python
# Make responses shorter and faster
prompt = f"Answer briefly in 1-2 sentences: {user_input}"

# Or make assistant more helpful
prompt = f"You are a helpful AI assistant. Answer: {user_input}"
```

---

### TTS (Text-to-Speech) Options

Currently supporting Piper TTS with fallback to sine wave beep.

#### Using Piper (Recommended)

Install Piper:
```bash
# Mac/Linux
curl -sSO https://github.com/rhasspy/piper/releases/download/v1.2.0/piper_amd64.tar.gz
tar xzf piper_amd64.tar.gz
export PATH=$PATH:./piper

# Windows
# Download from: https://github.com/rhasspy/piper/releases
# Add to PATH
```

Download a voice:
```bash
piper --download-voice en_US-ryan-high
```

Available voices:
- `en_US-ryan-high` - Male, energetic
- `en_US-ryan-medium` - Male, normal
- `en_US-ryan-low` - Male, calm
- `en_US-amy-medium` - Female, American
- `en_US-lessac-medium` - Female, professional

**Change voice in assistant.py:**
```python
result = subprocess.run(
    ["piper", "--model", "en_US-lessac-medium", "--output-raw"],
    input=text.encode(),
    capture_output=True,
    timeout=10,
)
```

---

## Advanced Scenarios

### Scenario 1: Slow Response Time

If responses are slow, try these optimizations:

1. Use a smaller Whisper model:
   ```python
   self.whisper_model = WhisperModel("tiny.en", ...)
   ```

2. Use a faster Ollama model:
   ```bash
   ollama pull orca-mini
   ```
   Then use it:
   ```python
   model="orca-mini"
   ```

3. Enable quantization (trade accuracy for speed):
   ```python
   compute_type="int4"  # Instead of int8
   ```

4. Reduce streaming chunk size in LLM output:
   ```python
   # Send fewer words at a time before TTS
   if len(words) >= 3:  # Instead of 5
   ```

### Scenario 2: Too Many False Positives (Echo/Noise)

The assistant keeps interrupting itself or reacting to background noise.

**Solution Stack:**
1. Increase VAD threshold: `threshold=0.7`
2. Increase min_speech_duration: `min_speech_duration_ms=1000`
3. Mute mic during playback: `self.mute_vad_during_tts = True`
4. Use a better microphone (noise-cancelling)

### Scenario 3: Streaming Too Fast/Slow

If the TTS is playing too fast or too slow:

**Too slow:** Lower the chunk delay
```python
# In text_to_speech_and_play(), around line 250
await asyncio.sleep(0.02)  # Reduce from 0.05
```

**Too fast:** Increase the chunk delay
```python
await asyncio.sleep(0.1)  # Increase from 0.05
```

### Scenario 4: Better Interruption Response

To make interruption even faster:

1. Reduce VAD check interval:
   ```python
   # In listen_to_microphone()
   await asyncio.sleep(0.005)  # Reduce from 0.01
   ```

2. Check interrupt event more frequently:
   ```python
   # In text_to_speech_and_play()
   await asyncio.sleep(0.02)  # Reduce from 0.05
   ```

3. Use a faster VAD model (if available):
   ```python
   self.vad_model = load_silero_vad(onnx=True)  # ONNX is faster
   ```

---

## System Monitoring

### Check Ollama Status

```bash
# Is Ollama running?
curl http://127.0.0.1:11434/api/tags

# Test an inference
ollama run llama3 "Hello, what is your name?"
```

### Check Python Dependencies

```bash
pip list | grep -E "pyaudio|whisper|ollama|silero|torch"
```

### Monitor Audio Input

```bash
# List audio devices
python -c "import pyaudio; p = pyaudio.PyAudio(); [print(f'{i}: {p.get_device_info_by_index(i)[\"name\"]}') for i in range(p.get_device_count())]"
```

### Check CPU Usage

While the assistant is running:

**Windows:**
```bash
tasklist | find python
```

**Mac/Linux:**
```bash
ps aux | grep python
```

---

## Performance Profiles

### Profile 1: Maximum Speed (Laptop/Weak CPU)

```python
# assistant.py
self.whisper_model = WhisperModel("tiny.en", device="cpu", compute_type="int4")
self.llm_model = GPT4All("orca-mini-3b-gguf2-q4_0.gguf")

# In listen_to_microphone()
threshold=0.6
min_speech_duration_ms=800
```

### Profile 2: Balanced (Most Systems)

```python
# Current default configuration
self.whisper_model = WhisperModel("base.en", device="cpu", compute_type="int8")
self.llm_model = GPT4All("mistral-7b-openorca.gguf2.Q4_0.gguf")
threshold=0.5
min_speech_duration_ms=500
```

### Profile 3: Maximum Accuracy (Powerful CPU)

```python
# assistant.py
self.whisper_model = WhisperModel("small.en", device="cpu", compute_type="int8")
self.llm_model = GPT4All("gpt4-x-alpaca-13b-native-4bit-128g.gguf")

# In listen_to_microphone()
threshold=0.4
min_speech_duration_ms=300
```

---

## Debugging

### Enable Verbose Logging

In `assistant.py`, change:
```python
logging.basicConfig(level=logging.DEBUG)  # Instead of INFO
```

### Audio Echo Diagnosis

1. Speak and listen carefully
2. If echo occurs:
   - Check if `mute_vad_during_tts` is True
   - Try increasing `threshold` in VAD
   - Consider a noise-cancelling microphone

3. Test with:
   ```bash
   # Record for 5 seconds what the mic hears
   python -c "import pyaudio; ..."
   ```

### Whisper Accuracy Check

```bash
# Test Whisper directly
python -c "from faster_whisper import WhisperModel; m = WhisperModel('base.en'); print(m.transcribe('audio.wav'))"
```

---

## Hardware Recommendations

### Minimum (Working but Slow)
- CPU: Intel i5-6th Gen or equivalent
- RAM: 8GB
- Storage: 10GB (for models)
- Microphone: USB headset

### Recommended (Good Speed)
- CPU: Intel i7 or AMD Ryzen 5
- RAM: 16GB
- Storage: 20GB SSD
- Microphone: Good USB condenser mic

### Ideal (Fast & Reliable)
- CPU: Intel i9 / AMD Ryzen 9
- RAM: 32GB
- Storage: Fast SSD (500GB+)
- Microphone: Professional USB microphone

---

Need more help? Check out:
- [README.md](README.md) - Full documentation
- [QUICKSTART.md](QUICKSTART.md) - Quick setup guide
- [assistant.py](assistant.py) - Source code

