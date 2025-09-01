"""
Edge AI Assistant - Main Module

Features:
- Voice input support with multimodal models
- Dynamic model selection based on query complexity
- Natural multi-turn conversations
- Specialized agents for calendar, search, and vehicle assistance
"""

import asyncio
import logging
import os
import base64
import requests
import json
import re
import sys
import subprocess

# Load environment variables from .env file
from pathlib import Path
from dotenv import load_dotenv

# Load .env file if it exists
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
    print(f"✅ Loaded configuration from {env_path}")
else:
    print(f"⚠️ No .env file found at {env_path}, using defaults")

# Disable FAISS GPU warnings globally - we only use CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from opentelemetry import baggage, context

from strands import Agent
from strands.models import BedrockModel
from strands.models.llamacpp import LlamaCppModel
from strands.agent import SlidingWindowConversationManager
from src.agents.cockpit import (
    climate_control,
    window_control,
    seat_control,
    lighting_control,
    drive_mode,
)
from src.agents.tools import select_model
from src.config import (
    LLAMACPP_URL,
    BEDROCK_MODEL_ID,
    CONTEXT_WINDOW,
    MAX_TOKENS,
    USE_RICH_UI,
    VOICE_DURATION,
    SESSION_ID,
    FFMPEG_PATH,
    LLAMA_CTX_SIZE,
)
from src.user_profiles import get_user_profile

if USE_RICH_UI:
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    from rich.table import Table

    console = Console()
else:
    console = None
    # Define dummy classes to avoid undefined variable errors
    Panel = None
    Text = None
    Table = None

# Configure logging level based on environment variable
DEBUG_MODE = os.getenv("DEBUG", "false").lower() == "true"
if DEBUG_MODE:
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    print("🐛 DEBUG MODE ENABLED - Detailed logging active")
else:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

_orchestrator = None

# API client mode variables
API_URL = "http://localhost:8000"
USE_API_CLIENT = False


def check_api_server():
    """Check if llama.cpp server is available"""
    try:
        # Check llama.cpp server health endpoint on port 8080
        response = requests.get(f"{LLAMACPP_URL}/health", timeout=1)
        if response.status_code == 200:
            return True
    except (requests.ConnectionError, requests.Timeout):
        pass

    # Fallback to checking API on port 8000 if it exists
    try:
        response = requests.get(f"{API_URL}/health", timeout=1)
        if response.status_code == 200:
            return True
    except (requests.ConnectionError, requests.Timeout):
        pass
    return False


# FastAPI imports for API mode
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
import uvicorn


def create_bedrock_model():
    """Create new Bedrock model instance with flexible AWS credential handling"""
    import boto3

    # Get AWS configuration from environment
    aws_profile = os.getenv("AWS_PROFILE")
    aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    aws_region = os.getenv("AWS_REGION", "us-east-1")

    # Create session based on available credentials
    if aws_profile:
        # Use specific profile
        session = boto3.Session(profile_name=aws_profile, region_name=aws_region)
        print(f"[BEDROCK] Using AWS profile: {aws_profile}")
    elif aws_access_key and aws_secret_key:
        # Use access key/secret
        session = boto3.Session(
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=aws_region
        )
        print("[BEDROCK] Using AWS access key/secret")
    else:
        # Use default credential chain
        session = boto3.Session(region_name=aws_region)
        print("[BEDROCK] Using default AWS credentials")

    return BedrockModel(
        model_id=BEDROCK_MODEL_ID,
        region_name=aws_region,
        session=session
    )


def create_llamacpp_model():
    """Create new LlamaCpp model instance with optimizations for edge deployment"""
    return LlamaCppModel(
        base_url=LLAMACPP_URL,
        model_id="default",
        timeout=120.0,  # Increased timeout for container environment
        params={
            "temperature": 0.7,
            "max_tokens": MAX_TOKENS,
            "top_k": 20,  # Reduced for faster sampling
            "repeat_penalty": 1.1,
            "cache_prompt": True,  # Essential for function calling performance
            "n_batch": int(os.getenv("LLAMA_BATCH_SIZE", "512")),  # Batch processing
            "n_threads": int(os.getenv("GGML_NTHREADS", "6")),  # Thread count
        },
    )


# Orchestrator prompt for edge deployment (local model)
ORCHESTRATOR_PROMPT_LOCAL = """You are Alex, a highly capable AI assistant designed for edge deployment in automotive and industrial environments.

🚨 FIRST: ANALYZE THE USER'S REQUEST AND ROUTE TO THE CORRECT SPECIALIST:

COCKPIT CONTROLS (Direct Commands):
IF USER MENTIONS: temperature, AC, heat, cool, defrost, fan, air → USE climate_control
IF USER MENTIONS: window, windows, sunroof, open window, close window, vent → USE window_control  
IF USER MENTIONS: seat, recline, lumbar, seat heating, seat cooling, memory position → USE seat_control
IF USER MENTIONS: lights, headlights, interior lights, ambient, fog lights, reading light → USE lighting_control
IF USER MENTIONS: sport mode, eco mode, drive mode, traction control, lane assist → USE drive_mode

PERSONA & APPROACH:
- Professional, efficient, and safety-conscious  
- Prioritize quick, accurate responses for operational contexts
- Anticipate user needs while maintaining clear boundaries
- Speak naturally but concisely for hands-free usage

PRIMARY CAPABILITIES:

1. COCKPIT CONTROLS - Direct vehicle control:
   • Climate: "Set temperature to 72", "Turn on AC", "Defrost windshield" → climate_control
   • Windows: "Open driver window", "Close all windows", "Vent windows" → window_control
   • Seats: "Move seat back", "Turn on seat heating", "Save position 1" → seat_control
   • Lights: "Turn on headlights", "Set ambient blue", "Reading light on" → lighting_control
   • Drive: "Switch to sport mode", "Enable eco mode", "Turn on traction" → drive_mode

ROUTING DECISION MATRIX:
- Temperature/AC/Heat/Fan → climate_control
- Window/Sunroof/Vent → window_control
- Seat/Recline/Lumbar → seat_control
- Lights/Headlights/Ambient → lighting_control
- Mode/Sport/Eco/Traction → drive_mode

CRITICAL ROUTING RULES:
- COCKPIT COMMANDS (action words): "set", "turn on", "open", "close", "adjust" + component → route to specific control agent
- Direct temperature/climate commands → climate_control
- Direct window/sunroof commands → window_control
- Direct seat adjustment → seat_control
- Direct lighting control → lighting_control
- Direct drive mode changes → drive_mode

🚨 ROUTING PRIORITY:
1. Cockpit control action words (set temperature, open window, move seat) = specific control agent

OPERATIONAL CONTEXT:
- Edge deployment with potential connectivity limitations
- Images dynamically loaded from production systems
- Resource-constrained environment requiring efficiency
- Safety-critical responses for automotive use

INTERACTION GUIDELINES:
- Acknowledge requests before routing to specialists
- Provide context when delegating to agents
- Handle errors gracefully with clear next steps
- Keep responses concise for in-vehicle safety
- Offer proactive suggestions when appropriate

Remember: You coordinate specialist agents to deliver comprehensive assistance while maintaining operational safety and efficiency."""

# General-purpose prompt for cloud/Bedrock deployment
ORCHESTRATOR_PROMPT_CLOUD = """You are Alex, a highly capable AI assistant with access to advanced cloud-based reasoning capabilities.

CORE CAPABILITIES:
- Complex analysis and strategic thinking
- Comprehensive research and detailed explanations
- Multi-step problem solving and planning
- Creative writing and content generation
- Technical analysis and code review
- Business strategy and market analysis

VEHICLE CONTROLS (when requested):
IF USER MENTIONS: temperature, AC, heat, cool, defrost, fan, air → USE climate_control
IF USER MENTIONS: window, windows, sunroof, open window, close window, vent → USE window_control
IF USER MENTIONS: seat, recline, lumbar, seat heating, seat cooling, memory position → USE seat_control
IF USER MENTIONS: lights, headlights, interior lights, ambient, fog lights, reading light → USE lighting_control
IF USER MENTIONS: sport mode, eco mode, drive mode, traction control, lane assist → USE drive_mode

APPROACH:
- Provide comprehensive, detailed responses for complex queries
- Use analytical thinking and structured reasoning
- Offer strategic insights and actionable recommendations
- Support both automotive and general-purpose assistance
- Maintain professional expertise across diverse domains

INTERACTION STYLE:
- Thorough and analytical for complex requests
- Concise and direct for simple commands
- Proactive in offering additional insights
- Clear structure with headings and bullet points when helpful

You have access to powerful cloud-based reasoning and can handle sophisticated analysis, strategic planning, and detailed explanations across any domain."""


def get_orchestrator():
    """Get or create the main orchestrator agent with conversation management"""
    global _orchestrator
    if _orchestrator is None:
        # Load user profile based on environment variable
        driver_profile = os.environ.get("DRIVER_PROFILE", "guest").lower()
        user_profile = get_user_profile(driver_profile)

        # Create personalized prompt (default to local/automotive prompt)
        personalized_prompt = ORCHESTRATOR_PROMPT_LOCAL + "\n\n" + user_profile["personalization_prompt"]

        _orchestrator = Agent(
            model=create_llamacpp_model(),
            system_prompt=personalized_prompt,
            tools=[
                # Cockpit Controls
                climate_control,
                window_control,
                seat_control,
                lighting_control,
                drive_mode,
            ],
            conversation_manager=SlidingWindowConversationManager(window_size=CONTEXT_WINDOW),
            trace_attributes={
                "session.id": SESSION_ID,
                "user.profile": driver_profile,
                "user.name": user_profile["name"],
            },
        )

        # Log the active profile
        if driver_profile != "guest":
            logger.info(
                f"Loaded driver profile: {user_profile['name']} ({user_profile['vehicle']['make']} {user_profile['vehicle']['model']})"
            )
    return _orchestrator


def update_orchestrator_model(provider: str):
    """Update the orchestrator's model based on selection and propagate to agents"""
    orchestrator = get_orchestrator()

    # Get user profile for personalization
    driver_profile = os.environ.get("DRIVER_PROFILE", "guest").lower()
    user_profile = get_user_profile(driver_profile)

    # Create new model and prompt based on provider
    if provider == "llamacpp":
        new_model = create_llamacpp_model()
        # Use automotive-focused prompt for local model
        new_prompt = ORCHESTRATOR_PROMPT_LOCAL + "\n\n" + user_profile["personalization_prompt"]
        print("[LOCAL] Switched to local LlamaCpp model")
    else:
        new_model = create_bedrock_model()
        # Use general-purpose prompt for cloud model
        new_prompt = ORCHESTRATOR_PROMPT_CLOUD + "\n\n" + user_profile["personalization_prompt"]
        print("[CLOUD] Switched to cloud Bedrock model")

    # Update the orchestrator model and system prompt
    orchestrator.model = new_model
    orchestrator.system_prompt = new_prompt

    # Update all agent models to match orchestrator's selection
    # Note: Cockpit controls use local model only by design

    return orchestrator


def process_input_via_api(user_input: str) -> str:
    """Process user input via API server"""
    try:
        # Handle voice input locally and send text to API
        if user_input.lower() in ["voice", "listen", "speak"]:
            if USE_RICH_UI and console:
                console.print("🎤 [cyan]Recording voice locally...[/cyan]")
            else:
                print("[VOICE] Recording voice locally...")

            # Use FFmpeg for local recording and transcription
            import subprocess
            import json

            if sys.platform == "darwin":
                audio_input = ["-f", "avfoundation", "-i", ":0"]
            else:
                audio_input = ["-f", "pulse", "-i", "default"]

            # Use FFmpeg with Whisper support (compiled in container or local)
            ffmpeg_path = FFMPEG_PATH
            whisper_model = os.getenv("WHISPER_MODEL_PATH", "/app/models/ggml-base.bin")

            # No need to set library paths in container (already in LD_LIBRARY_PATH)
            env = os.environ.copy()

            cmd = [
                ffmpeg_path,
                "-loglevel",
                "error",
                *audio_input,
                "-t",
                str(VOICE_DURATION),
                "-af",
                f"whisper=model={whisper_model}:language=en:format=text:destination=-",
                "-f",
                "null",
                "-",
            ]

            try:
                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=VOICE_DURATION + 10, env=env
                )
                transcribed_text = result.stdout.strip()

                if transcribed_text:
                    # Send transcribed text to API
                    response = requests.post(
                        f"{API_URL}/chat", json={"prompt": transcribed_text}, timeout=30
                    )
                else:
                    return "[ERROR] No speech detected"
            except Exception as e:
                return f"[ERROR] Transcription failed: {str(e)}"
        else:
            # Send text prompt to API
            response = requests.post(
                f"{API_URL}/chat",
                json={"prompt": user_input},
                timeout=180,  # 3 minutes for complex requests with tool calls
            )

        if response.status_code == 200:
            return response.json()["response"]
        else:
            return f"[ERROR] API request failed: {response.status_code}"

    except requests.ConnectionError:
        return "[ERROR] Cannot connect to API server at " + API_URL
    except requests.Timeout:
        return "[ERROR] API request timed out"
    except Exception as e:
        return f"[ERROR] API request failed: {str(e)}"


def process_input(user_input: str, audio_data: bytes = None, audio_format: str = "wav") -> str:
    """Process user input with dynamic model selection and natural conversation flow.

    Args:
        user_input: Text input from user
        audio_data: Optional audio bytes for voice processing
        audio_format: Format of audio data (default: wav)
    """

    # If using API client mode, delegate to API
    if USE_API_CLIENT:
        return process_input_via_api(user_input)

    # Handle voice input activation with FFmpeg Whisper
    if user_input.lower() in ["voice", "listen", "speak"] or audio_data is not None:
        if USE_RICH_UI and console:
            console.print("🎤 [cyan]Recording audio...[/cyan]")
        else:
            print("[VOICE] Recording audio...")

        # Use FFmpeg with Whisper filter for transcription
        import subprocess
        import json

        # Determine audio input source
        if sys.platform == "darwin":
            audio_input = ["-f", "avfoundation", "-i", ":0"]  # macOS
        else:
            audio_input = ["-f", "pulse", "-i", "default"]  # Linux

        # Use FFmpeg with Whisper support (compiled in container or local)
        ffmpeg_path = FFMPEG_PATH
        whisper_model = os.getenv("WHISPER_MODEL_PATH", "/app/models/ggml-base.bin")

        # Set library paths for FFmpeg with Whisper
        env = os.environ.copy()
        if sys.platform == "darwin":
            # Auto-detect Whisper library paths for macOS
            dyld_path = os.getenv("DYLD_LIBRARY_PATH", "")
            if not dyld_path and ffmpeg_path and "whisper" in ffmpeg_path.lower():
                # Try to find whisper.cpp build directory relative to ffmpeg
                ffmpeg_dir = Path(ffmpeg_path).parent.parent.parent
                whisper_build = ffmpeg_dir / "whisper.cpp" / "build"
                if whisper_build.exists():
                    dyld_paths = [
                        str(whisper_build / "src"),
                        str(whisper_build / "ggml" / "src"),
                        str(whisper_build / "ggml" / "src" / "ggml-metal"),
                        str(whisper_build / "ggml" / "src" / "ggml-blas"),
                    ]
                    dyld_path = ":".join([p for p in dyld_paths if Path(p).exists()])
            if dyld_path:
                env["DYLD_LIBRARY_PATH"] = dyld_path

        cmd = [
            ffmpeg_path,
            "-loglevel",
            "warning",
            *audio_input,
            "-t",
            str(VOICE_DURATION),  # Record duration
            "-af",
            f"whisper=model={whisper_model}:language=en:format=text:destination=-",
            "-f",
            "null",
            "-",
        ]

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=VOICE_DURATION + 10, env=env
            )

            # Check for errors
            if result.returncode != 0:
                error_msg = result.stderr.strip() if result.stderr else "Unknown error"
                if "Library not loaded" in error_msg or "dyld" in error_msg:
                    return "[ERROR] FFmpeg Whisper library not found. Check DYLD_LIBRARY_PATH"
                elif "avfoundation" in error_msg:
                    return "[ERROR] Microphone access denied. Please grant permission in System Settings"
                else:
                    return f"[ERROR] FFmpeg failed: {error_msg[:100]}"

            # Get text output directly from FFmpeg Whisper
            user_query = result.stdout.strip()

            # Remove any warning messages that might appear in output
            lines = user_query.split("\n")
            user_query = " ".join(
                [line for line in lines if not line.startswith("[") and line.strip()]
            )
            user_query = user_query.strip()

            if USE_RICH_UI and console:
                console.print(f"🎙️  [green]Transcribed:[/green] '{user_query}'")
            else:
                print(f"[MIC] Transcribed: '{user_query}'")

            if not user_query:
                return "[ERROR] No speech detected. Please try again."

        except subprocess.TimeoutExpired:
            return "[ERROR] Audio recording timeout."
        except FileNotFoundError:
            return "[ERROR] FFmpeg not found. Please install FFmpeg with Whisper support."
        except Exception as e:
            return f"[ERROR] Transcription failed: {str(e)}"
    else:
        user_query = user_input

    # Dynamic model selection
    if USE_RICH_UI and console:
        console.print(f"🤖 [cyan]Analyzing query complexity...[/cyan]")
    else:
        print(f"\n[AI] Analyzing query complexity...")

    selection = select_model(user_query)

    if USE_RICH_UI and console:
        console.print(f"📊 [yellow]Model Selection:[/yellow] {selection['provider']}")
        console.print(f"💭 [dim]Reasoning:[/dim] {selection['reasoning']}")
    else:
        print(f"[SELECT] Model Selection: {selection['provider']}")
        print(f"[REASON] Reasoning: {selection['reasoning']}")

    # Update orchestrator model dynamically
    orchestrator = update_orchestrator_model(selection["provider"])

    # Process with selected model
    if USE_RICH_UI and console:
        console.print(f"🔄 [cyan]Processing request...[/cyan]")
    else:
        print(f"\n[PROC] Processing request: '{user_query}'")

    try:
        response = orchestrator(user_query)
        return str(response) if response else ""
    except Exception as e:
        print(f"[ERROR] Orchestrator failed: {e}")
        import traceback

        traceback.print_exc()
        return f"Error: {str(e)}"


def voice_enabled_assistant(
    query: str = None, use_voice: bool = False, voice_duration: int = 10
) -> str:
    """
    Simplified interface for voice-enabled assistant.

    Args:
        query: Text query (if not provided, will prompt for input)
        use_voice: Whether to use voice input
        voice_duration: Duration for voice recording (default: 10 seconds)

    Returns:
        str: Assistant response
    """
    if use_voice:
        return process_input("voice")
    elif query:
        return process_input(query)
    else:
        return process_input(input("You: "))


def display_welcome_message():
    """Display welcome message with deployment info"""
    # Get current user profile
    driver_profile = os.environ.get("DRIVER_PROFILE", "guest").lower()
    user_profile = get_user_profile(driver_profile)

    if USE_RICH_UI and console:
        # Rich UI version
        info_table = Table(show_header=False, box=None, padding=(0, 1))
        info_table.add_column("Label", style="cyan")
        info_table.add_column("Value", style="bright_white")

        info_table.add_row("💾 Context Window", str(CONTEXT_WINDOW))
        info_table.add_row("🧠 Context Size", f"{LLAMA_CTX_SIZE} tokens")
        info_table.add_row("🎤 Voice Support", "✅ Enabled")
        info_table.add_row("🌐 Mode", "Container" if check_api_server() else "Local")

        # Add user profile info
        if driver_profile != "guest":
            info_table.add_row(
                "👤 Driver",
                f"{user_profile['name']} ({user_profile['vehicle']['make']} {user_profile['vehicle']['model']})",
            )

        console.print(
            Panel(
                info_table,
                title="🤖 [bold]Edge AI Assistant[/bold]",
                title_align="center",
                border_style="cyan",
                padding=(1, 1),
            )
        )

        capabilities = Text()
        capabilities.append("🎤 Voice: ", style="bold cyan")
        capabilities.append("Say 'voice' for speech input\n", style="white")
        capabilities.append("🎛️ Cockpit: ", style="bold red")
        capabilities.append("Control climate, windows, seats, lights, drive modes\n", style="white")

        console.print(
            Panel(
                capabilities,
                title="✨ [bold]Capabilities[/bold]",
                title_align="center",
                border_style="green",
                padding=(1, 1),
            )
        )
    else:
        # Simple text version
        print("=============================================================================")
        print("              EDGE AI ASSISTANT")
        print("=============================================================================")
        print(f"Context: {CONTEXT_WINDOW} messages | 4096 tokens")
        print(f"Mode: {'API Client (Container)' if USE_API_CLIENT else 'Local Processing'}")
        if driver_profile != "guest":
            print(
                f"Driver: {user_profile['name']} | Vehicle: {user_profile['vehicle']['make']} {user_profile['vehicle']['model']}"
            )
        print()
        print("Capabilities:")
        print("   🎤 Voice Input: Say 'voice' to use speech input")
        print("   🎛️ Cockpit Controls: Climate, windows, seats, lights, drive modes")
        print()
        print("Usage: Type your request or 'voice' for speech | 'exit' to quit")
        print("=============================================================================")
        print()


def set_session_context(session_id=None):
    """Set the session ID in OpenTelemetry baggage for trace correlation"""
    ctx = baggage.set_baggage("session.id", session_id)
    token = context.attach(ctx)
    return token


def main():
    """Main entry point"""
    global USE_API_CLIENT

    # Skip API client check if we're running as the API server
    if os.getenv("ENABLE_API", "false").lower() == "true":
        # We ARE the API server, don't try to connect to ourselves
        return

    # Check if API server is available
    if check_api_server():
        if USE_RICH_UI and console:
            console.print(
                f"🌐 [bold green]Connected to llama.cpp server[/bold green] at {LLAMACPP_URL}"
            )
            console.print("📡 [dim]Using containerized model server[/dim]")
            console.print()
        else:
            print(f"🌐 Connected to llama.cpp server at {LLAMACPP_URL}")
            print("📡 Using containerized model server")
            print()
    else:
        if USE_RICH_UI and console:
            console.print(f"⚠️ [yellow]No llama.cpp server found at {LLAMACPP_URL}[/yellow]")
            console.print("[dim]Make sure container is running: docker ps[/dim]")
            console.print()
        else:
            print(f"⚠️ No llama.cpp server found at {LLAMACPP_URL}")
            print("Make sure container is running: docker ps")
            print()

    display_welcome_message()

    set_session_context(SESSION_ID)

    if USE_RICH_UI and console:
        console.print("🚀 [bold green]Assistant is ready![/bold green]")
        console.print()

    while True:
        try:
            if USE_RICH_UI and console:
                prompt_text = "👤 [bold]USER:[/bold] "
            else:
                prompt_text = "USER: "

            user_input = input(prompt_text).strip()

            if not user_input:
                if USE_RICH_UI and console:
                    console.print(
                        "💡 [dim]Please enter a message or 'voice' for speech input[/dim]"
                    )
                else:
                    print("[INFO] Please enter a message or 'voice' for speech input")
                continue

            if user_input.lower() in ["exit", "quit", "bye", "goodbye"]:
                if USE_RICH_UI and console:
                    console.print(
                        "👋 [bold cyan]Thank you for using Edge AI Assistant![/bold cyan]"
                    )
                    console.print("🌟 [dim]Have a great day![/dim]")
                else:
                    print("\nThank you for using Edge AI Assistant! Have a great day!")
                break

            # Process input and get response
            if USE_RICH_UI and console:
                console.print()

            response = process_input(user_input)
            
            if USE_RICH_UI and console:
                console.print()
            else:
                print()

        except KeyboardInterrupt:
            if USE_RICH_UI and console:
                console.print("\n🛑 [yellow]Assistant interrupted![/yellow]")
            else:
                print("\n\nAssistant interrupted! Goodbye!")
            break
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            if USE_RICH_UI and console:
                console.print(f"❌ [red]Error:[/red] {str(e)}")
            else:
                print(f"[ERROR] {str(e)}")


app = FastAPI(title="Edge AI Assistant")


class ChatRequest(BaseModel):
    prompt: str
    session_id: str = "default"
    audio_data: str = None  # Base64 encoded audio
    audio_format: str = "wav"


@app.get("/health")
def health_check():
    return {"status": "healthy", "context_size": 4096}


@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """Main chat endpoint - supports text and audio input"""
    try:
        # Handle audio data if provided
        audio_bytes = None
        if request.audio_data:
            # Decode base64 audio data
            try:
                audio_bytes = base64.b64decode(request.audio_data)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid audio data: {str(e)}")

        response = process_input(request.prompt, audio_bytes, request.audio_format)
        return {"response": response, "session_id": request.session_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def run_api_server():
    """Run FastAPI server"""
    config = uvicorn.Config(app=app, host="0.0.0.0", port=8000)
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    # Check if API mode is requested AND we're not in a docker exec session
    # Docker exec sessions will have TERM set but not be the initial startup
    is_docker_exec = (
        os.getenv("TERM") and os.getenv("ENABLE_API") == "true" and os.getenv("API_ALREADY_RUNNING")
    )

    if os.getenv("ENABLE_API", "false").lower() == "true" and not is_docker_exec:
        print("🚀 Starting Edge AI Assistant in API mode...")
        print("📡 API will be available at http://0.0.0.0:8000")
        asyncio.run(run_api_server())
    else:
        main()
