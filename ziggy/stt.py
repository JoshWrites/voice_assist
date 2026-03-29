"""Speech-to-text via Vosk (offline)."""

import json
import tempfile
import wave
from pathlib import Path

import vosk

from ziggy.config import VOSK_MODEL_PATHS, AUDIO_SAMPLE_RATE, AUDIO_CHUNK_SIZE


class SpeechRecognizer:
    def __init__(self):
        self.model = None
        self.sample_rate = AUDIO_SAMPLE_RATE
        self.chunk_size = AUDIO_CHUNK_SIZE

    def setup(self) -> bool:
        for path_str in VOSK_MODEL_PATHS:
            path = Path(path_str)
            if path.exists():
                self.model = vosk.Model(str(path))
                print("  Speech recognition model loaded")
                return True
        print("  No Vosk model found. Please download one:")
        print("  wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip")
        return False

    def transcribe(self, audio_data: bytes, sample_size: int) -> str:
        """Convert raw audio bytes to text."""
        if not self.model:
            return ""
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp_path = tmp.name

            with wave.open(tmp_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(sample_size)
                wf.setframerate(self.sample_rate)
                wf.writeframes(audio_data)

            recognizer = vosk.KaldiRecognizer(self.model, self.sample_rate)
            results = []
            with wave.open(tmp_path, "rb") as wf:
                while True:
                    data = wf.readframes(self.chunk_size)
                    if len(data) == 0:
                        break
                    if recognizer.AcceptWaveform(data):
                        result = json.loads(recognizer.Result())
                        if result.get("text"):
                            results.append(result["text"])

            final = json.loads(recognizer.FinalResult())
            if final.get("text"):
                results.append(final["text"])

            Path(tmp_path).unlink(missing_ok=True)
            return " ".join(results).strip()

        except Exception as e:
            print(f"Speech recognition error: {e}")
            return ""

    def create_recognizer(self):
        """Create a new KaldiRecognizer for streaming use."""
        return vosk.KaldiRecognizer(self.model, self.sample_rate)
