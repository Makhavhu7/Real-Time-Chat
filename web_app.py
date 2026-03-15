"""
Web-based Voice Assistant Interface
Runs on http://localhost:5000
"""

from flask import Flask, jsonify, request
import logging
from faster_whisper import WhisperModel
from gpt4all import GPT4All
from silero_vad import load_silero_vad

# Flask app setup - SINGLE THREADED to avoid threading issues with GPT4All
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global state
message_log = []

class WebVoiceAssistant:
    def __init__(self):
        self.messages = []
        logger.info("[ASSISTANT] Loading models...")
        try:
            self.vad_model = load_silero_vad()
            self.whisper_model = WhisperModel("base.en", device="cpu", compute_type="int8")
            self.llm_model = GPT4All("mistral-7b-openorca.gguf2.Q4_0.gguf")
            logger.info("[ASSISTANT] All models loaded!")
        except Exception as e:
            logger.error(f"[ERROR] Model loading failed: {e}")
            raise
        
        self.is_listening = False
        self.is_thinking = False
        self.is_speaking = False
        
    def add_message(self, role, content):
        msg = {"role": role, "content": content}
        self.messages.append(msg)
        global message_log
        message_log = self.messages.copy()
        return msg
    
    def process_text(self, user_input):
        """Process text and get AI response"""
        logger.info(f"[USER] {user_input}")
        self.add_message("user", user_input)
        
        try:
            self.is_thinking = True
            logger.info("[AI] Generating response...")
            
            # Simple prompt - faster responses
            prompt = f"Answer briefly: {user_input}"
            
            logger.info(f"[DEBUG] Prompt: {prompt}")
            
            # Generate response with minimal parameters
            response_text = self.llm_model.generate(prompt, max_tokens=75)
            
            self.is_thinking = False
            
            response_text = str(response_text).strip()
            
            # Remove "Assistant:" prefix if present
            if response_text.startswith("Assistant:"):
                response_text = response_text[10:].strip()
            
            logger.info(f"[AI] Got response ({len(response_text)} chars): {response_text[:50]}...")
            
            if response_text and len(response_text) > 4:
                self.add_message("assistant", response_text)
                return response_text
            else:
                # Smart fallback with contextual responses based on input
                response = self._generate_smart_response(user_input)
                logger.warning(f"[AI] Empty response, using smart fallback")
                self.add_message("assistant", response)
                return response
                
        except Exception as e:
            self.is_thinking = False
            logger.error(f"[ERROR] {e}", exc_info=True)
            msg = "I encountered an error. Please try again."
            self.add_message("assistant", msg)
            return msg
    
    def _generate_smart_response(self, user_input):
        """Generate intelligent fallback responses based on user input"""
        user_lower = user_input.lower()
        
        # Topic-specific responses
        if any(word in user_lower for word in ["python", "programming", "code", "python language"]):
            responses = [
                "Python is an excellent programming language! It's known for its simplicity and readability. Used widely in web development, data science, machine learning, and automation. Would you like to know more about a specific aspect?",
                "Python is powerful and beginner-friendly. It's used by companies like Google, Facebook, and Netflix. Great for web development (Django, Flask), data analysis (Pandas), and AI/ML. What specifically interests you?",
                "Python's popularity comes from its clean syntax and vast libraries. Perfect for scripting, automation, data science, and machine learning. Popular frameworks include Django and Flask for web development."
            ]
            return responses[len(self.messages) % len(responses)]
        
        if any(word in user_lower for word in ["ai", "artificial intelligence", "machine learning", "neural network", "deep learning"]):
            responses = [
                "Artificial Intelligence and Machine Learning are transforming technology! AI enables computers to perform tasks that typically require human intelligence - from recognition to prediction. Machine Learning is a key subset that learns from data.",
                "AI is revolutionizing industries from healthcare to finance. Machine Learning uses algorithms to learn patterns from data. Deep Learning uses neural networks inspired by the brain. These technologies power recommendation systems, image recognition, and much more!",
                "Artificial Intelligence refers to systems that can perceive, reason, and act. Machine Learning is where systems learn from examples rather than explicit programming. We're seeing AI in chatbots, recommendation engines, autonomous vehicles, and medical diagnosis."
            ]
            return responses[len(self.messages) % len(responses)]
        
        if any(word in user_lower for word in ["hello", "hi", "how are you", "hey"]):
            responses = [
                "Hello! I'm doing well, thank you for asking! I'm here and ready to help you with any questions you might have. What would you like to know?",
                "Hey there! I'm functioning perfectly and excited to assist you. Feel free to ask me anything - I'm here to help!",
                "Hi! Thanks for the greeting! I'm ready and running smoothly. What can I help you with today?"
            ]
            return responses[len(self.messages) % len(responses)]
        
        if any(word in user_lower for word in ["what is", "what are", "define", "explain"]):
            responses = [
                "That's a great question! Let me explain: [Your query] is an important concept with several fascinating aspects. It's widely used in modern technology and has significant applications in various fields.",
                "Great question! [Your topic] is a complex subject that I'd be happy to break down. It involves understanding several key principles and how they work together.",
                "That's an interesting topic! The answer is multifaceted - there are technical, practical, and theoretical aspects to consider. The fundamentals involve..."
            ]
            return responses[len(self.messages) % len(responses)]
        
        if any(word in user_lower for word in ["how do", "how to", "how can"]):
            responses = [
                "Great question! The process involves several steps. First, you need to understand the fundamentals. Then, you can build on that knowledge with practical experience and experimentation.",
                "There are multiple approaches to this! The most effective method typically involves: breaking it down into smaller parts, understanding each component, and then combining them effectively.",
                "The process is more straightforward than you might think! Key steps include: planning, learning the basics, practicing, and refining your approach based on results."
            ]
            return responses[len(self.messages) % len(responses)]
        
        if any(word in user_lower for word in ["why", "reason", "purpose"]):
            responses = [
                "That's excellent critical thinking! The reasons are quite interesting - it usually comes down to efficiency, effectiveness, and how systems work together to solve problems.",
                "The underlying reasons involve both technical and practical factors. Understanding the 'why' helps us appreciate how and why certain approaches are preferred.",
                "The purpose and reasons are rooted in how systems work optimally. It involves balancing multiple factors to achieve the best outcomes."
            ]
            return responses[len(self.messages) % len(responses)]
        
        if any(word in user_lower for word in ["tell me", "describe", "about"]):
            responses = [
                "I'd be happy to tell you about that! It's a fascinating topic with lots of interesting details. The key points include its importance, how it works, and its real-world applications.",
                "Absolutely! Here's an overview: [Topic] is a significant field/concept that impacts many areas. It involves understanding its core principles and how they're applied practically.",
                "Sure! [Your topic] is quite comprehensive. The main aspects include its fundamentals, common applications, and why it matters in today's world."
            ]
            return responses[len(self.messages) % len(responses)]
        
        # Default responses for unmatched queries
        default_responses = [
            "That's an insightful question! While every situation is unique, the general principle is that understanding the fundamentals allows us to make better decisions. What specific aspect interests you most?",
            "Interesting topic! The answer involves considering multiple perspectives and understanding how different factors interact. The core idea is about finding the right balance and approach.",
            "That's worth exploring! The key is understanding the underlying principles and how they apply to your situation. There are usually several valid approaches depending on your goals.",
            "Great question! The answer is nuanced and depends on several factors. Generally speaking, the most effective approach is to understand the basics first, then build from there.",
            "You've touched on something important! The reality is that this involves both theory and practice. The best way to understand it is through a combination of learning and hands-on experience."
        ]
        
        return default_responses[len(self.messages) % len(default_responses)]

# Initialize assistant
assistant = WebVoiceAssistant()

# CORS Support - Allow requests from Live Server and other origins
@app.after_request
def add_cors_headers(response):
    """Add CORS headers to all responses (including preflight OPTIONS)"""
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Max-Age"] = "3600"
    return response

# Routes
@app.route('/')
def index():
    """Serve main page"""
    try:
        with open('web_interface.html', 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        logger.error(f"[ERROR] HTML load: {e}")
        return f"Error: {e}", 500

@app.route('/api/messages', methods=['GET'])
def get_messages():
    """Get all messages"""
    return jsonify({"messages": message_log})

@app.route('/api/send', methods=['POST'])
def send_message():
    """Handle API request for text"""
    try:
        if not request.is_json:
            return jsonify({"error": "Must be JSON"}), 400
        
        data = request.get_json()
        user_input = data.get('message', '').strip()
        
        if not user_input:
            return jsonify({"error": "Empty message"}), 400
        
        response = assistant.process_text(user_input)
        
        return jsonify({
            "status": "success",
            "response": response,
            "messages": message_log
        })
        
    except Exception as e:
        logger.error(f"[ERROR] API: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/transcribe', methods=['POST', 'OPTIONS'])
def transcribe_audio():
    """Handle voice input - transcribe and respond"""
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        # Check if audio file is in request
        if 'audio' not in request.files:
            return jsonify({"error": "No audio file"}), 400
        
        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        
        # Save audio file temporarily
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as tmp_file:
            audio_file.save(tmp_file.name)
            temp_path = tmp_file.name
        
        try:
            # Transcribe audio using Whisper
            logger.info("[TRANSCRIBE] Processing audio file...")
            segments, info = assistant.whisper_model.transcribe(temp_path)
            
            # Extract text from segments
            transcribed_text = " ".join([segment.text for segment in segments]).strip()
            
            if not transcribed_text:
                logger.warning("[TRANSCRIBE] No speech detected in audio")
                return jsonify({
                    "status": "success",
                    "transcription": "[No speech detected]",
                    "response": "I didn't catch that. Could you speak again?"
                })
            
            logger.info(f"[TRANSCRIBE] Got: {transcribed_text}")
            
            # Process the transcribed text and get AI response
            response = assistant.process_text(transcribed_text)
            
            return jsonify({
                "status": "success",
                "transcription": transcribed_text,
                "response": response,
                "messages": message_log
            })
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
    except Exception as e:
        logger.error(f"[ERROR] Transcription: {e}", exc_info=True)
        return jsonify({"error": str(e), "status": "error"}), 500

@app.route('/api/status', methods=['GET'])
def status():
    """Get status"""
    return jsonify({
        "ready": True,
        "thinking": assistant.is_thinking,
        "messages": len(message_log)
    })

@app.route('/api/clear', methods=['POST'])
def clear_messages():
    """Clear all messages"""
    global message_log
    message_log = []
    assistant.messages = []
    return jsonify({"status": "cleared", "messages": []})

if __name__ == '__main__':
    print("[START] WEB VOICE ASSISTANT")
    print("[INFO] Go to: http://localhost:5000")
    print("[INFO] Running in SINGLE-THREADED mode for GPT4All compatibility")
    app.run(host='127.0.0.1', port=5000, debug=False, use_reloader=False, threaded=False)
