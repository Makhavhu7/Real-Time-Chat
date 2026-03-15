@echo off
REM Real-Chat-App Setup for Windows
REM This script sets up PyAudio and Python dependencies

echo.
echo ================================
echo Real-Chat-App Setup for Windows
echo ================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://www.python.org
    pause
    exit /b 1
)

echo OK Python is installed: 
python --version
echo.

REM Install PyAudio using pipwin
echo.
echo Installing PyAudio (this is required for microphone input)...
echo.

python -m pip install --upgrade pip
python -m pip install pipwin
pipwin install pyaudio

if errorlevel 1 (
    echo.
    echo Warning: PyAudio installation had issues.
    echo You may need to install it separately or use Windows installers.
    echo.
)

echo.
echo Installing Python dependencies from requirements.txt...
echo.

python -m pip install -r requirements.txt

echo.
echo Downloading Mistral model from GPT4All...
echo This may take a few minutes on first run...
echo.

python -c "from gpt4all import GPT4All; GPT4All('mistral-7b-openorca.gguf2.Q4_0.gguf')"

echo.
echo ================================
echo Setup Complete!
echo ================================
echo.
echo Next steps:
echo.
echo Run the assistant:
echo    python assistant.py
echo.
echo That's it! No need to start a separate server.
echo.
pause
