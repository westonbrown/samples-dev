#!/usr/bin/env python3
"""
Audio CLI for recording and sending audio to the AI assistant.

This is a self-contained module that provides command-line tools for:
1. Recording audio to container exchange directory
2. Recording and sending audio to API endpoint

Usage:
    python -m src.utils.audio_cli record --duration 10
    python -m src.utils.audio_cli api --duration 10 --url http://localhost:8000/chat
"""

import argparse
import base64
import json
import subprocess
import sys
import tempfile
import wave
from pathlib import Path
from typing import Optional, Dict, Any

import requests

try:
    import sounddevice as sd
    import numpy as np

    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False


class AudioRecorder:
    """Simple audio recorder using sounddevice or subprocess."""

    def __init__(self, sample_rate: int = 16000, channels: int = 1):
        """
        Initialize audio recorder.

        Args:
            sample_rate: Sample rate in Hz
            channels: Number of audio channels
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.exchange_dir = Path("audio_exchange")

    def record(self, duration: int) -> Optional[bytes]:
        """
        Record audio for specified duration.

        Args:
            duration: Recording duration in seconds

        Returns:
            WAV audio data as bytes, or None if recording failed
        """
        if SOUNDDEVICE_AVAILABLE:
            return self._record_sounddevice(duration)
        else:
            return self._record_subprocess(duration)

    def _record_sounddevice(self, duration: int) -> Optional[bytes]:
        """Record using sounddevice library."""
        try:
            print(f"🎤 Recording for {duration} seconds...")

            # Record audio
            recording = sd.rec(
                int(duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="int16",
            )
            sd.wait()  # Wait for recording to finish

            # Convert to WAV format
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                # pylint: disable=no-member
                with wave.open(tmp_file.name, "wb") as wav_file:
                    wav_file.setnchannels(self.channels)
                    wav_file.setsampwidth(2)  # 16-bit
                    wav_file.setframerate(self.sample_rate)
                    wav_file.writeframes(recording.tobytes())
                # pylint: enable=no-member

                # Read back the WAV data
                with open(tmp_file.name, "rb") as f:
                    audio_data = f.read()

                Path(tmp_file.name).unlink()  # Clean up temp file
                return audio_data

        except Exception as e:
            print(f"Recording error: {e}")
            return None

    def _record_subprocess(self, duration: int) -> Optional[bytes]:
        """Record using subprocess (ffmpeg or sox)."""
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                # Try ffmpeg first
                if sys.platform == "darwin":
                    # macOS
                    cmd = [
                        "ffmpeg",
                        "-y",
                        "-f",
                        "avfoundation",
                        "-i",
                        ":0",
                        "-t",
                        str(duration),
                        "-ar",
                        str(self.sample_rate),
                        "-ac",
                        str(self.channels),
                        "-f",
                        "wav",
                        tmp_file.name,
                    ]
                else:
                    # Linux
                    cmd = [
                        "ffmpeg",
                        "-y",
                        "-f",
                        "pulse",
                        "-i",
                        "default",
                        "-t",
                        str(duration),
                        "-ar",
                        str(self.sample_rate),
                        "-ac",
                        str(self.channels),
                        "-f",
                        "wav",
                        tmp_file.name,
                    ]

                print(f"🎤 Recording for {duration} seconds...")
                result = subprocess.run(cmd, capture_output=True)

                if result.returncode == 0:
                    with open(tmp_file.name, "rb") as f:
                        audio_data = f.read()
                    Path(tmp_file.name).unlink()
                    return audio_data
                else:
                    print(f"Recording failed: {result.stderr.decode()}")
                    return None

        except Exception as e:
            print(f"Recording error: {e}")
            return None

    def save_for_container(self, audio_data: bytes, filename: str = "voice_input.wav") -> Path:
        """
        Save audio data to container exchange directory.

        Args:
            audio_data: WAV audio data
            filename: Output filename

        Returns:
            Path to saved file
        """
        # Create exchange directory if it doesn't exist
        self.exchange_dir.mkdir(exist_ok=True)

        # Save audio file
        output_path = self.exchange_dir / filename
        with open(output_path, "wb") as f:
            f.write(audio_data)

        return output_path


class AudioTransport:
    """Handle audio data transport for API communication."""

    @staticmethod
    def prepare_api_request(audio_data: bytes, session_id: str = "default") -> Dict[str, Any]:
        """
        Prepare audio data for API request.

        Args:
            audio_data: WAV audio data
            session_id: Session identifier

        Returns:
            Dictionary ready for JSON serialization
        """
        # Encode audio as base64
        audio_base64 = base64.b64encode(audio_data).decode("utf-8")

        return {
            "prompt": "voice",
            "audio_data": audio_base64,
            "audio_format": "wav",
            "session_id": session_id,
        }


def record_for_container(duration: int = 10, filename: str = "voice_input.wav") -> int:
    """
    Record audio and save to container exchange directory.

    Args:
        duration: Recording duration in seconds
        filename: Output filename

    Returns:
        Exit code (0 for success, 1 for error)
    """
    try:
        recorder = AudioRecorder()

        print(f"Recording for {duration} seconds...")
        audio_data = recorder.record(duration)

        if audio_data:
            # Save to container exchange directory
            saved_path = recorder.save_for_container(audio_data, filename)
            print(f"Audio saved to: {saved_path}")
            return 0
        else:
            print("Failed to record audio")
            return 1

    except Exception as e:
        print(f"Error recording audio: {str(e)}")
        return 1


def send_to_api(
    duration: int = 10, api_url: str = "http://localhost:8000/chat", session_id: str = "cli"
) -> int:
    """
    Record audio and send to API endpoint.

    Args:
        duration: Recording duration in seconds
        api_url: API endpoint URL
        session_id: Session identifier

    Returns:
        Exit code (0 for success, 1 for error)
    """
    try:
        recorder = AudioRecorder()

        print(f"Recording for {duration} seconds...")
        audio_data = recorder.record(duration)

        if not audio_data:
            print("Failed to record audio")
            return 1

        # Prepare API request
        request_data = AudioTransport.prepare_api_request(audio_data, session_id)

        print(f"Sending audio to {api_url}...")
        response = requests.post(api_url, json=request_data, timeout=300)

        if response.status_code == 200:
            result = response.json()
            
            # Show transcription if available
            if "transcription" in result:
                print(f"🎙️  Transcribed: '{result['transcription']}'")
            
            print("\nResponse:")
            print(result.get("response", "No response"))
            return 0
        else:
            print(f"API error: {response.status_code}")
            print(response.text)
            return 1

    except Exception as e:
        print(f"Error: {str(e)}")
        return 1


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(description="Audio CLI for AI Assistant", prog="audio_cli")

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Record command
    record_parser = subparsers.add_parser("record", help="Record audio for container mode")
    record_parser.add_argument(
        "--duration", type=int, default=10, help="Recording duration in seconds (default: 10)"
    )
    record_parser.add_argument(
        "--filename", default="voice_input.wav", help="Output filename (default: voice_input.wav)"
    )

    # API command
    api_parser = subparsers.add_parser("api", help="Record and send audio to API")
    api_parser.add_argument(
        "--duration", type=int, default=10, help="Recording duration in seconds (default: 10)"
    )
    api_parser.add_argument(
        "--url",
        default="http://localhost:8000/chat",
        help="API endpoint URL (default: http://localhost:8000/chat)",
    )
    api_parser.add_argument("--session-id", default="cli", help="Session identifier (default: cli)")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    if args.command == "record":
        return record_for_container(args.duration, args.filename)
    elif args.command == "api":
        return send_to_api(args.duration, args.url, args.session_id)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
