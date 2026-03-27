"""Voxtral 4B TTS engine — Mistral's open-weight neural voice via vLLM."""

import os
import subprocess
import tempfile

import requests

from ziggy.tts import TTSEngine

MODEL_ID = "mistralai/Voxtral-4B-TTS-2603"
DEFAULT_VOICE = "casual_male"
DEFAULT_URL = "http://localhost:8000"


class VoxtralTTS(TTSEngine):
    name = "voxtral"

    def __init__(self, url=DEFAULT_URL, voice=DEFAULT_VOICE):
        self.url = url
        self.voice = voice
        self._available = False

    def setup(self):
        print("  Checking for Voxtral TTS server...")
        try:
            resp = requests.get(f"{self.url}/v1/models", timeout=3)
            if resp.status_code == 200:
                for model in resp.json().get("data", []):
                    mid = model.get("id", "").lower()
                    if "voxtral" in mid or "tts" in mid:
                        self._available = True
                        print(f"  Voxtral TTS ready (voice: {self.voice})")
                        return
                print("  vLLM server running but Voxtral model not loaded")
        except requests.exceptions.ConnectionError:
            pass
        except Exception as e:
            print(f"  Voxtral check failed: {e}")

        print("  To enable Voxtral TTS:")
        print("    pip install -U vllm")
        print("    pip install git+https://github.com/vllm-project/vllm-omni.git")
        print("    vllm serve mistralai/Voxtral-4B-TTS-2603 --omni")

    def is_available(self) -> bool:
        return self._available

    def _request_audio(self, text: str):
        return requests.post(
            f"{self.url}/v1/audio/speech",
            json={
                "model": MODEL_ID,
                "input": text,
                "voice": self.voice,
                "response_format": "wav",
            },
            timeout=30,
        )

    def speak(self, text: str) -> bool:
        try:
            resp = self._request_audio(text)
            if resp.status_code != 200:
                print(f"  Voxtral TTS HTTP {resp.status_code}")
                return False
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(resp.content)
                tmp_path = tmp.name
            proc = subprocess.Popen(
                ["aplay", tmp_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            proc.wait()
            os.unlink(tmp_path)
            return True
        except Exception as e:
            print(f"  Voxtral TTS failed: {e}")
            return False

    def speak_sentence_async(self, sentence: str):
        try:
            resp = self._request_audio(sentence)
            if resp.status_code != 200:
                return None, None
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(resp.content)
                tmp_path = tmp.name
            proc = subprocess.Popen(
                ["aplay", tmp_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return proc, tmp_path
        except Exception:
            return None, None
