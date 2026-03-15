# Real-Chat-App: Local Voice Assistant with Barge-In Support

A fully local, real-time voice assistant in Python with **interruption (barge-in)** capability. If you start speaking while the assistant is talking, it instantly stops its audio and listens to you.

## Architecture

```
┌─────────────────┐
│   Microphone    │
└────────┬────────┘
         │
         ▼
┌─────────────────────────┐
│  Silero-VAD (Speech     │  ← Always listening for interruption
│   Detection)            │
└─────────┬───────────────┘
          │
          ▼
┌──────────────────────────┐
│   Faster-Whisper (STT)  │  ← Transcribes your voice
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│   GPT4All (Local LLM)    │  ← Streams responses (Mistral)
│   Mistral Model          │
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│   Piper TTS (Speech)     │  ← Fast local text-to-speech
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│   Speaker/Audio Output   │  ← Interruptible playback
└──────────────────────────┘
```

## Installation

### 1. Install GPT4All (The "Brain")

Install GPT4All via pip (no separate download needed):
```bash
pip install gpt4all
```

The model will download automatically on first run.

Alternative: Download the GUI from [gpt4all.io](https://gpt4all.io) if you prefer a desktop app.

### 2. Install Audio Tools (The "Ears")

**Windows:**
```bash
pip install pipwin
pipwin install pyaudio
```

**Mac:**
```bash
brew install portaudio
pip install pyaudio
```

**Linux:**
```bash
sudo apt-get install portaudio19-dev
pip install pyaudio
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 4. (Optional) Install Piper TTS (The "Voice")

For faster, higher-quality speech synthesis:

**Mac/Linux:**
```bash
curl -sSO https://github.com/rhasspy/piper/releases/download/v1.2.0/piper_amd64.tar.gz
tar xzf piper_amd64.tar.gz
export PATH=$PATH:./piper
```

**Windows:**
Download from: https://github.com/rhasspy/piper/releases

Then download a voice model:
```bash
piper --download-voice en_US-ryan-high
```

## Running the Assistant

```bash
python assistant.py
```

You should see:
```
============================================================
🎙️  LOCAL VOICE ASSISTANT WITH BARGE-IN SUPPORT
============================================================

Starting up...

🤖 Assistant is listening...
```

Now just speak naturally! The assistant will:
1. Listen for your speech
2. Transcribe it
3. Think (stream response from GPT4All / Mistral)
4. Speak back to you

**Interrupt at any time:** Just start talking while it's speaking, and it will stop instantly! 🎤🔴

## How Interruption Works

The magic is in the **VAD (Voice Activity Detection)** pipeline:

1. **Silero-VAD** constantly monitors your microphone (~10ms latency)
2. While the assistant is speaking (`is_speaking = True`), VAD is **not muted** (controlled by `mute_vad_during_tts`)
3. When VAD detects your voice:
   - The `interrupt_event` is triggered
   - Audio playback stops immediately
   - The text queue is cleared
   - The assistant returns to listening mode

This all happens **asynchronously** using `asyncio`, so the microphone listening and audio playback run in parallel.

## Troubleshooting

### "Echo" Problem (Assistant Hears Itself)

If the assistant keeps interrupting itself:

1. **Solution 1:** Lower sensitivity in Silero-VAD
   ```python
   threshold=0.6  # Increase from 0.5 to 0.6
   ```

2. **Solution 2:** Mute VAD during TTS (enabled by default)
   ```python
   self.mute_vad_during_tts = True  # Already set in the code
   ```

### Audio Not Playing

- Make sure Piper TTS is installed (or it falls back to a simple beep)
- Check your speakers are working: `python -c "import pyaudio; print(pyaudio.PyAudio().get_device_count())"`

### Whisper Taking Too Long

- Using the `base.en` model for speed (160MB)
- If you want faster: use `tiny.en` model
- If you want better accuracy: use `small.en` model

Change in `assistant.py`:
```python
self.whisper_model = WhisperModel("tiny.en", device="cpu", compute_type="int8")
```

### Ollama Not Connecting

Make sure Ollama is running:
```bash
ollama serve
```

In a separate terminal, test it:
```bash
ollama list
```

## Prompt Used (From Cursor)

This code was generated using this prompt in Cursor:

> "I want to create a **fully local, real-time voice assistant** in Python that supports **interruption (barge-in)**. If I start speaking while the assistant is talking, it must stop its audio instantly and listen to me.
> 
> **Technical Stack:**
> 1. **VAD (Interruption):** Use `silero-vad` to monitor the microphone constantly.
> 2. **STT (Ears):** Use `faster-whisper` (base.en model) for fast local transcription.
> 3. **LLM (Brain):** Use the `ollama` library to talk to a local `llama3` instance. Enable **streaming** so text is processed as it's generated.
> 4. **TTS (Voice):** Use `kokoro-tts` or `Piper` for ultra-fast local speech synthesis.
> 
> **Requirements for the Code:**
> * Use `asyncio` to run the 'Microphone Listener' and the 'Audio Player' at the same time.
> * Implement a global `interrupt_event = asyncio.Event()`.
> * If `silero-vad` detects speech while the TTS is playing, trigger the event to **kill the audio playback thread** and clear the text queue.
> * Use `PyAudio` for the audio stream handling.
> * Add a simple console printout: 'Assistant is listening...', 'Assistant is thinking...', 'Assistant is speaking...'"

## Adjusting Interruption Sensitivity

If the interruption is too twitchy or not sensitive enough:

**Too sensitive (interrupts on small noises):**
```python
# In listen_to_microphone(), increase the threshold
speech_timestamps = get_speech_timestamps(
    torch.from_numpy(audio_int16),
    self.vad_model,
    sampling_rate=vad_sr,
    threshold=0.6,  # Increase this (0.0-1.0 scale)
    min_speech_duration_ms=700,  # Increase to require longer speech
)
```

**Not sensitive enough (takes too long to interrupt):**
```python
# Decrease threshold and min_speech_duration
speech_timestamps = get_speech_timestamps(
    torch.from_numpy(audio_int16),
    self.vad_model,
    sampling_rate=vad_sr,
    threshold=0.3,  # Lower this
    min_speech_duration_ms=300,  # Lower to detect faster
)
```

## Features

- ✅ **100% Local** - No cloud API keys needed
- ✅ **Real-time Streaming** - Text appears as it's generated
- ✅ **Barge-in Interruption** - Stop the assistant mid-sentence
- ✅ **Async/Concurrent** - Microphone listening and playback run in parallel
- ✅ **Fast Inference** - CPU-optimized models
- ✅ **Console Status** - See what the assistant is doing

## License

Open source. Use freely.

---

Happy chatting! 🎙️🚀
