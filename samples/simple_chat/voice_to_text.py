"""Voice-to-text CLI using Amazon Nova with microphone input.

This script captures audio from your microphone and sends it to Amazon Nova
for transcription/analysis.

Requirements:
    pip install sounddevice soundfile numpy

Usage:
    python voice_to_text.py

Press Ctrl+C to stop recording and get the transcription.
"""

import argparse
import base64
import io
import os
import sys
import wave

try:
    import numpy as np
    import sounddevice as sd
    import soundfile as sf
except ImportError:
    print("Error: Missing required packages.")
    print("Please install: pip install sounddevice soundfile numpy")
    sys.exit(1)

from langchain_amazon_nova import ChatAmazonNova


def record_audio(duration=None, sample_rate=16000):
    """Record audio from microphone.

    Args:
        duration: Recording duration in seconds. If None, records until interrupted.
        sample_rate: Audio sample rate in Hz. Default is 16000.

    Returns:
        numpy array of audio data
    """
    print(f"🎤 Recording audio at {sample_rate}Hz...")
    if duration:
        print(f"   Duration: {duration} seconds")
        audio_data = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype="int16",
        )
        sd.wait()
    else:
        print("   Press Ctrl+C to stop recording...")
        recording = []
        try:
            with sd.InputStream(
                samplerate=sample_rate, channels=1, dtype="int16"
            ) as stream:
                while True:
                    data, overflowed = stream.read(sample_rate)
                    recording.append(data)
        except KeyboardInterrupt:
            print("\n✓ Recording stopped")

        audio_data = np.concatenate(recording, axis=0)

    print(f"✓ Recorded {len(audio_data) / sample_rate:.1f} seconds of audio")
    return audio_data, sample_rate


def audio_to_base64(audio_data, sample_rate, format="wav"):
    """Convert audio numpy array to base64-encoded string.

    Args:
        audio_data: numpy array of audio samples
        sample_rate: Sample rate of the audio
        format: Audio format (only 'wav' is currently supported)

    Returns:
        base64-encoded audio data string
    """
    buffer = io.BytesIO()

    if format == "wav":
        with wave.open(buffer, "wb") as wf:
            wf.setnchannels(1)  # Mono
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sample_rate)
            wf.writeframes(audio_data.tobytes())
    else:
        # Use soundfile for other formats
        sf.write(buffer, audio_data, sample_rate, format=format)

    buffer.seek(0)
    audio_bytes = buffer.read()
    return base64.b64encode(audio_bytes).decode("utf-8")


def transcribe_audio_with_nova(audio_base64, model="nova-pro-v1", prompt=None):
    """Send audio to Nova for transcription/analysis.

    Args:
        audio_base64: Base64-encoded audio data
        model: Nova model to use
        prompt: Optional text prompt to guide the response

    Returns:
        Transcription/analysis text from Nova
    """
    llm = ChatAmazonNova(model=model, temperature=0)

    # Create message with audio input
    default_prompt = "Please respond to this audio"
    text_prompt = prompt or default_prompt

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text_prompt},
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": audio_base64,
                        "format": "wav",
                    },
                },
            ],
        }
    ]

    print(f"\n🤖 Sending to Amazon Nova ({model})...")
    response = llm.invoke(messages)
    return response.content


def main():
    parser = argparse.ArgumentParser(description="Voice-to-text CLI using Amazon Nova")
    parser.add_argument(
        "--duration",
        "-d",
        type=float,
        default=None,
        help="Recording duration in seconds (default: record until Ctrl+C)",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="nova-pro-v1",
        help="Nova model to use (default: nova-pro-v1)",
    )
    parser.add_argument(
        "--sample-rate",
        "-s",
        type=int,
        default=16000,
        help="Audio sample rate in Hz (default: 16000)",
    )
    parser.add_argument(
        "--prompt",
        "-p",
        type=str,
        default=None,
        help="Custom prompt for Nova (default: 'Please respond to this audio')",
    )
    parser.add_argument(
        "--file",
        "-f",
        type=str,
        default=None,
        help="Path to audio file instead of recording from microphone",
    )

    args = parser.parse_args()

    # Check for API key
    if not os.getenv("NOVA_API_KEY"):
        print("Error: NOVA_API_KEY environment variable not set")
        print("Please set it with: export NOVA_API_KEY='your-api-key'")
        sys.exit(1)

    try:
        if args.file:
            # Load audio from file
            print(f"📁 Loading audio from: {args.file}")
            audio_data, sample_rate = sf.read(args.file, dtype="int16")
            if len(audio_data.shape) > 1:
                # Convert stereo to mono
                audio_data = audio_data.mean(axis=1).astype("int16")
            print(f"✓ Loaded {len(audio_data) / sample_rate:.1f} seconds of audio")
        else:
            # Record from microphone
            audio_data, sample_rate = record_audio(
                duration=args.duration, sample_rate=args.sample_rate
            )

        # Convert to base64
        print("\n🔄 Encoding audio...")
        audio_base64 = audio_to_base64(audio_data, sample_rate)
        print(f"✓ Encoded {len(audio_base64)} bytes")

        # Send to Nova
        result = transcribe_audio_with_nova(
            audio_base64, model=args.model, prompt=args.prompt
        )

        # Display result
        print("\n" + "=" * 50)
        print("Response:")
        print("=" * 50)
        print(result)
        print("=" * 50)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
