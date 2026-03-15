"""
Real-time Voice Assistant with Interruption (Barge-in) Support
Local-only: Ollama (LLM) + Faster-Whisper (STT) + Piper (TTS) + Silero-VAD (Interruption)
"""

import asyncio
import numpy as np
import sounddevice as sd
import soundfile as sf
import sys
import logging
from threading import Thread, Event as ThreadingEvent
from queue import Queue, Empty
from collections import deque
import json

# Import the AI libraries
from faster_whisper import WhisperModel
from gpt4all import GPT4All
from silero_vad import load_silero_vad, get_speech_timestamps
import torch
import subprocess
import wave
import io

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
CHANNELS = 1
SAMPLE_RATE = 16000
CHUNK_SIZE = 512
SILENCE_THRESHOLD = 0.02
SPEECH_PADDING_MS = 300

# Global state
interrupt_event = asyncio.Event()
is_playing_audio = False
audio_queue = Queue()
microphone_queue = Queue()


class VoiceAssistant:
    def __init__(self):
        self.vad_model = load_silero_vad()
        self.whisper_model = WhisperModel("base.en", device="cpu", compute_type="int8")
        # Initialize GPT4All (downloads model automatically on first run)
        self.llm_model = GPT4All("mistral-7b-openorca.gguf2.Q4_0.gguf")
        self.is_listening = False
        self.is_thinking = False
        self.is_speaking = False
        self.mute_vad_during_tts = True  # Prevent echo
        
    def print_status(self, status):
        """Print assistant status to console"""
        print(f"\n[ASSISTANT] {status}\n")
        
    async def listen_to_microphone(self):
        """
        Continuously listen to microphone and detect speech using Silero-VAD.
        This runs in the background and feeds audio into the processing pipeline.
        """
        self.print_status("Assistant is listening...")
        
        audio_buffer = deque(maxlen=SAMPLE_RATE * 5)  # 5 second buffer
        vad_sr = 16000
        active_stream = [None]  # Use list for mutable reference
        
        # Callback for audio stream
        async def create_audio_callback():
            def audio_callback(indata, frames, time, status):
                if status:
                    logger.error(f"Audio stream status: {status}")
                # Add audio to buffer
                try:
                    audio_buffer.extend(indata[:, 0])
                except:
                    pass
            return audio_callback
        
        try:
            # Create recording stream with sounddevice
            callback = await create_audio_callback()
            active_stream[0] = sd.InputStream(
                channels=CHANNELS,
                samplerate=SAMPLE_RATE,
                blocksize=CHUNK_SIZE,
                callback=callback,
                dtype=np.float32
            )
            active_stream[0].start()
            
            while True:
                try:
                    # Skip VAD if we're playing audio (to prevent echo)
                    if self.mute_vad_during_tts and self.is_speaking:
                        await asyncio.sleep(0.01)
                        continue
                    
                    # Run VAD on accumulated audio
                    if len(audio_buffer) >= SAMPLE_RATE:
                        try:
                            audio_int16 = (np.array(list(audio_buffer)) * 32767).astype(np.int16)
                            
                            # Get speech timestamps
                            speech_timestamps = get_speech_timestamps(
                                torch.from_numpy(audio_int16),
                                self.vad_model,
                                sampling_rate=vad_sr,
                                threshold=0.5,
                                min_speech_duration_ms=500,
                                max_speech_duration_s=None,
                                min_silence_duration_ms=300,
                            )
                            
                            # If speech detected
                            if speech_timestamps and len(speech_timestamps) > 0:
                                # If assistant is speaking, trigger interrupt
                                if self.is_speaking:
                                    logger.info("[INTERRUPT] INTERRUPTION DETECTED - Stopping audio playback")
                                    interrupt_event.set()
                                    self.is_speaking = False
                                    # Clear audio queue
                                    try:
                                        while True:
                                            audio_queue.get_nowait()
                                    except Empty:
                                        pass
                                
                                # Extract speech audio - handle various timestamp formats
                                try:
                                    ts = speech_timestamps[0]
                                    # Handle dict-like structure
                                    if isinstance(ts, dict):
                                        start_ms = ts.get("start", 0)
                                        end_ms = ts.get("end", 0)
                                    else:
                                        # Handle tuple or list format
                                        start_ms = ts[0] if len(ts) > 0 else 0
                                        end_ms = ts[1] if len(ts) > 1 else 0
                                    
                                    # Validate values
                                    if start_ms is not None and end_ms is not None and start_ms >= 0 and end_ms >= 0:
                                        start_sample = int(max(0, start_ms * vad_sr / 1000))
                                        end_sample = int(min(len(audio_int16), end_ms * vad_sr / 1000))
                                        
                                        if start_sample < end_sample and start_sample >= 0 and end_sample <= len(audio_int16):
                                            speech_audio = audio_int16[start_sample:end_sample]
                                            if len(speech_audio) > 0:
                                                microphone_queue.put(speech_audio)
                                except Exception as ts_err:
                                    logger.debug(f"VAD timestamp parse error: {ts_err}")
                        except Exception as vad_err:
                            logger.debug(f"VAD processing error: {vad_err}")
                            
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.debug(f"Error in microphone listening: {e}")
                    await asyncio.sleep(0.1)
                
                await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            pass
        finally:
            if active_stream[0]:
                active_stream[0].stop()
                active_stream[0].close()
    
    async def transcribe_speech(self):
        """
        Take audio chunks from the microphone queue and transcribe them using Whisper.
        """
        try:
            while True:
                try:
                    # Try to get audio from queue (non-blocking)
                    speech_audio = microphone_queue.get(timeout=1)
                    
                    self.print_status("Assistant is thinking...")
                    
                    # Transcribe with Whisper
                    segments, info = self.whisper_model.transcribe(
                        speech_audio,
                        language="en",
                        beam_size=5,
                    )
                    
                    user_text = "".join([segment.text for segment in segments]).strip()
                    
                    if user_text:
                        logger.info(f"[USER] User said: {user_text}")
                        # Send to LLM processing
                        await self.process_with_llm(user_text)
                    
                except Empty:
                    await asyncio.sleep(0.1)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error in transcription: {e}")
                    await asyncio.sleep(0.5)
        except asyncio.CancelledError:
            pass
    
    async def process_with_llm(self, user_input):
        """
        Send user input to GPT4All and stream the response.
        """
        try:
            self.print_status("Assistant is thinking...")
            
            # Use GPT4All to get streaming response
            response_text = ""
            
            # GPT4All generate with streaming
            for chunk in self.llm_model.generate(
                user_input,
                max_tokens=200,
                temp=0.7,
                top_p=0.9,
                streaming=True
            ):
                response_text += chunk
                
                # Send to TTS in chunks (every 5-10 words)
                words = response_text.split()
                if len(words) >= 5:
                    text_chunk = " ".join(words[:10])
                    audio_queue.put(text_chunk)
                    response_text = " ".join(words[10:])
                    
                    # Check for interruption
                    if interrupt_event.is_set():
                        interrupt_event.clear()
                        return
                
                # Small delay to allow interruption checking
                await asyncio.sleep(0.01)
            
            # Send remaining text
            if response_text.strip():
                audio_queue.put(response_text.strip())
            
            # Signal end of response
            audio_queue.put(None)
            
        except Exception as e:
            logger.error(f"Error in LLM processing: {e}")
            audio_queue.put(None)
    
    async def text_to_speech_and_play(self):
        """
        Convert text chunks to speech using Piper TTS and play them in real-time.
        Monitors interrupt_event to stop playback if user speaks.
        """
        try:
            while True:
                try:
                    # Get text chunk from queue
                    text_chunk = audio_queue.get(timeout=2)
                    
                    if text_chunk is None:  # End of response
                        self.is_speaking = False
                        continue
                    
                    self.print_status("Assistant is speaking...")
                    self.is_speaking = True
                    
                    # Convert text to speech using Piper
                    audio_data = self.text_to_speech_piper(text_chunk)
                    
                    # Play audio in chunks, checking for interruption
                    chunk_size = 4096
                    for i in range(0, len(audio_data), chunk_size):
                        # Check for interruption
                        if interrupt_event.is_set():
                            logger.info("[STOP] Audio playback interrupted")
                            interrupt_event.clear()
                            self.is_speaking = False
                            break
                        
                        chunk = audio_data[i:i + chunk_size]
                        await self.play_audio_chunk(chunk)
                        await asyncio.sleep(0.05)
                    
                    self.is_speaking = False
                    
                except Empty:
                    await asyncio.sleep(0.1)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error in TTS/playback: {e}")
                    self.is_speaking = False
                    await asyncio.sleep(0.5)
        except asyncio.CancelledError:
            pass
    
    def text_to_speech_piper(self, text):
        """
        Convert text to speech using Piper TTS (fast local synthesis).
        Falls back to simple beep if Piper not installed.
        """
        try:
            # Try to use piper-tts
            result = subprocess.run(
                ["piper", "--model", "en_US-ryan-high", "--output-raw"],
                input=text.encode(),
                capture_output=True,
                timeout=10,
            )
            
            if result.returncode == 0:
                audio_bytes = result.stdout
                audio_data = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32767
                return audio_data
        except (FileNotFoundError, subprocess.TimeoutExpired):
            logger.warning("Piper TTS not available, using fallback beep")
        
        # Fallback: generate simple tone/beep
        duration = len(text) * 0.05  # ~50ms per character
        t = np.linspace(0, duration, int(SAMPLE_RATE * duration))
        frequency = 440  # A4 note
        audio_data = 0.3 * np.sin(2 * np.pi * frequency * t).astype(np.float32)
        return audio_data
    
    async def play_audio_chunk(self, audio_chunk):
        """Play a chunk of audio data using sounddevice"""
        try:
            # Use sounddevice to play audio
            sd.play(audio_chunk, samplerate=SAMPLE_RATE, blocking=False)
        except Exception as e:
            logger.error(f"Error playing audio: {e}")
    
    async def run(self):
        """
        Main coroutine that runs all async tasks concurrently.
        """
        try:
            # Create tasks for all concurrent operations
            tasks = [
                asyncio.create_task(self.listen_to_microphone()),
                asyncio.create_task(self.transcribe_speech()),
                asyncio.create_task(self.text_to_speech_and_play()),
            ]
            
            # Run all tasks
            await asyncio.gather(*tasks)
        except KeyboardInterrupt:
            logger.info("Shutting down assistant...")
            for task in tasks:
                task.cancel()
        except Exception as e:
            logger.error(f"Fatal error: {e}")


async def main():
    """Entry point for the voice assistant"""
    print("\n" + "="*60)
    print("[VOICE ASSISTANT] LOCAL VOICE ASSISTANT WITH BARGE-IN SUPPORT")
    print("="*60)
    print("\nStarting up...\n")
    
    assistant = VoiceAssistant()
    await assistant.run()


if __name__ == "__main__":
    asyncio.run(main())
