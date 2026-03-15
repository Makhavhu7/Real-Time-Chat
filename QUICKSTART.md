# ⚡ Quick Start Guide

## Prerequisites

You must have these installed on your computer BEFORE running the assistant:

### 1. **Python 3.8+**
- Download from: https://www.python.org

### 2. **Audio Tools**

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

---

## Installation (30 seconds)

Navigate to the Real-Chat-App folder and run:

**Windows:**
```bash
setup.bat
```

**Mac/Linux:**
```bash
bash setup.sh
```

Or manually:
```bash
pip install -r requirements.txt
```

---

## Running the Assistant

Just run:
```bash
python assistant.py
```

That's it! No need to start a separate Ollama server.

---

## Testing It

1. You should see:
   ```
   🤖 Assistant is listening...
   ```

2. Speak clearly into your microphone
3. Wait for: `🤖 Assistant is thinking...`
4. Listen for the response: `🤖 Assistant is speaking...`
5. **To interrupt:** Just start talking again! 🎤🔴

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| "No module named gpt4all" | Run: `pip install gpt4all` |
| "No module pyaudio" | Run: `pipwin install pyaudio` (Windows) or `pip install pyaudio` (Mac/Linux) |
| "Assistant hears itself" | Increase VAD threshold in `assistant.py` line ~70 from 0.5 to 0.6 |
| "Response takes forever" | First run downloads Mistral model (~4GB). Be patient or use a faster model. |
| "Audio not playing" | Check your system volume and speakers |

---

## Configuration

Edit `config.json` to adjust:
- VAD sensitivity (interruption threshold)
- Whisper model (tiny/base/small)
- Ollama temperature (creativity)
- TTS voice model

---

## Understanding the Architecture

```
Your Voice → Microphone → Silero-VAD → Whisper → Ollama → Piper → Speakers
                           ↑                                         ↑
                    (Always listening for interruption)    (Can be stopped)
```

When you speak while the assistant is talking:
1. Silero-VAD detects your voice (~10ms)
2. `interrupt_event` is triggered
3. Audio stops immediately
4. Queue is cleared
5. Assistant goes back to listening

Everything runs in parallel using `asyncio`.

---

## See Also

- Full documentation: [README.md](README.md)
- Configuration reference: [config.json](config.json)
- Source code: [assistant.py](assistant.py)

---

🎉 Have fun with your local AI assistant!
