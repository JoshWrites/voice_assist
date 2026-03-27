"""Piper TTS engine — high-quality local neural voice."""

import os
import subprocess

from ziggy.tts import TTSEngine


class PiperTTS(TTSEngine):
    name = "piper"

    def __init__(self):
        self._available = False
        self.piper_path = os.path.expanduser("~/.local/share/piper/piper/piper")
        self.model_path = os.path.expanduser(
            "~/.local/share/piper/en_US-amy-medium.onnx"
        )

    def setup(self):
        if not (os.path.exists(self.piper_path) and os.path.exists(self.model_path)):
            return
        try:
            result = subprocess.run(
                [self.piper_path, "--version"], capture_output=True, text=True
            )
            if result.returncode == 0:
                self._available = True
                print("  Piper TTS ready (natural voice)")
        except Exception:
            pass

    def is_available(self) -> bool:
        return self._available

    def _escape(self, text: str) -> str:
        return text.replace("'", "'\"'\"'")

    def _piper_cmd(self, text: str):
        escaped = self._escape(text)
        return (
            f"echo '{escaped}' | {self.piper_path} "
            f"--model {self.model_path} --output-raw"
        )

    def speak(self, text: str) -> bool:
        try:
            piper = subprocess.Popen(
                self._piper_cmd(text),
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
            player = subprocess.Popen(
                ["aplay", "-r", "22050", "-f", "S16_LE", "-t", "raw", "-"],
                stdin=piper.stdout,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            player.wait()
            return True
        except Exception:
            return False

    def speak_sentence_async(self, sentence: str):
        try:
            piper = subprocess.Popen(
                self._piper_cmd(sentence),
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
            player = subprocess.Popen(
                ["aplay", "-r", "22050", "-f", "S16_LE", "-t", "raw", "-"],
                stdin=piper.stdout,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return player, None
        except Exception:
            return None, None
