"""
Configuration for Edge AI Assistant
"""

import os
import logging

# Disable FAISS GPU warnings - we only use CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Model configuration - read from environment (.env file)
LLAMACPP_URL = os.getenv("LLAMACPP_URL", "http://localhost:8080")  # OK to have network default
BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "us.anthropic.claude-3-5-sonnet-20241022-v2:0")
WHISPER_MODEL_PATH = os.getenv("WHISPER_MODEL_PATH", "/app/models/ggml-base.bin")
QWEN_MODEL_PATH = os.getenv("QWEN_MODEL_PATH", "/app/models/Qwen3-1.7B-Q8_0.gguf")

# FFmpeg configuration
FFMPEG_PATH = os.getenv("FFMPEG_PATH", "ffmpeg")

# Context configuration - read from .env
CONTEXT_WINDOW = int(os.getenv("CONTEXT_WINDOW", "20"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1024"))
LLAMA_CTX_SIZE = int(os.getenv("LLAMA_CTX_SIZE", "2048"))

# Voice configuration
VOICE_DURATION = 5 
VOICE_SAMPLE_RATE = 16000

# UI configuration
USE_RICH_UI = os.getenv("USE_RICH_UI", "true").lower() == "true"

# Session configuration
SESSION_ID = f"edge-ai-{os.getpid()}"

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL), format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
