#!/bin/bash
# Real-Chat-App Setup Script
# Automates the installation of PyAudio and Python dependencies

echo "================================"
echo "Real-Chat-App Setup"
echo "================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install it first."
    exit 1
fi

echo "✅ Python 3 found: $(python3 --version)"
echo ""

# Check OS
OS="Unknown"
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="Linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macOS"
elif [[ "$OSTYPE" == "msys" ]]; then
    OS="Windows"
fi

echo "Detected OS: $OS"
echo ""

echo "📦 Installing audio dependencies..."

if [ "$OS" = "macOS" ]; then
    if ! command -v brew &> /dev/null; then
        echo "Installing Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    fi
    brew install portaudio
elif [ "$OS" = "Linux" ]; then
    sudo apt-get update
    sudo apt-get install -y portaudio19-dev python3-dev
elif [ "$OS" = "Windows" ]; then
    echo "🪟 For Windows, using pipwin..."
    pip install pipwin
    pipwin install pyaudio
fi

echo ""

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""

# Download Mistral model
echo "🧠 Downloading Mistral model from GPT4All (this may take a few minutes)..."
python3 -c "from gpt4all import GPT4All; GPT4All('mistral-7b-openorca.gguf2.Q4_0.gguf')"

echo ""
echo "================================"
echo "✅ Setup Complete!"
echo "================================"
echo ""
echo "To start the assistant, run:"
echo "  python assistant.py"
echo ""
echo "That's it! No need to start a separate server."
echo ""
