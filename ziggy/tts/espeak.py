"""espeak TTS engine — basic fallback, always available on most Linux systems."""

import subprocess

from ziggy.tts import TTSEngine


class EspeakTTS(TTSEngine):
    name = "espeak"

    def __init__(self):
        self._available = False

    def setup(self):
        try:
            result = subprocess.run(
                ["which", "espeak"], capture_output=True, text=True
            )
            self._available = result.returncode == 0
            if self._available:
                print("  espeak TTS ready")
        except Exception:
            pass

    def is_available(self) -> bool:
        return self._available

    def speak(self, text: str) -> bool:
        try:
            subprocess.run(
                ["espeak", "-s", "150", "-v", "en", text],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return True
        except Exception:
            return False

    def speak_sentence_async(self, sentence: str):
        try:
            process = subprocess.Popen(
                ["espeak", "-s", "150", "-v", "en", sentence],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return process, None
        except Exception:
            return None, None
