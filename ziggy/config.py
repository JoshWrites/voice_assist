"""Configuration and resource profiles for Ziggy."""

RESOURCE_PROFILES = {
    "minimal": {
        "name": "Minimal",
        "description": "Low resource usage for gaming or older systems",
        "requirements": "4-8GB VRAM",
        "context_tokens": 8000,
        "history_limit": 10,
        "response_tokens": 500,
        "recording_conversational": 120,
        "recording_command": 30,
        "tts_engine": "espeak",
    },
    "standard": {
        "name": "Standard",
        "description": "Balanced performance for everyday use",
        "requirements": "8-16GB VRAM",
        "context_tokens": 16000,
        "history_limit": 25,
        "response_tokens": 1000,
        "recording_conversational": 300,
        "recording_command": 60,
        "tts_engine": "piper",
    },
    "performance": {
        "name": "Performance",
        "description": "Maximum capabilities for research and long conversations",
        "requirements": "16GB+ VRAM",
        "context_tokens": 32000,
        "history_limit": 50,
        "response_tokens": 2000,
        "recording_conversational": 600,
        "recording_command": 60,
        "tts_engine": "voxtral",
    },
}

PROFILE_ALIASES = {
    "gaming": "minimal",
    "game": "minimal",
    "low": "minimal",
    "normal": "standard",
    "balanced": "standard",
    "high": "performance",
    "maximum": "performance",
    "research": "performance",
}

WAKE_WORD = "ziggy"
SHUTDOWN_PHRASE = "take a break"

AUDIO_SAMPLE_RATE = 16000
AUDIO_CHUNK_SIZE = 4000
AUDIO_CHANNELS = 1

SILENCE_THRESHOLD = 1.5  # seconds of silence before stopping recording

VOSK_MODEL_PATHS = [
    "vosk-model-small-en-us-0.15",
    "vosk-model-en-us-0.22",
    "vosk-model-en-us-0.22-lgraph",
]
