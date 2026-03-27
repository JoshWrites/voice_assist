"""Ollama backend — local LLM server with its own API format."""

import subprocess
import time
from typing import Optional

import requests

from ziggy.backend import LLMBackend

DEFAULT_URL = "http://localhost:11434"


class OllamaBackend(LLMBackend):
    name = "Ollama"

    def __init__(self, url=DEFAULT_URL):
        self.url = url
        self._process = None

    def is_running(self) -> bool:
        try:
            resp = requests.get(f"{self.url}/api/tags", timeout=2)
            return resp.status_code == 200 and "models" in resp.json()
        except Exception:
            return False

    def start(self) -> bool:
        try:
            result = subprocess.run(
                ["which", "ollama"], capture_output=True, text=True
            )
            if result.returncode != 0:
                print("  Ollama not found")
                return False

            print("  Starting Ollama backend...")
            self._process = subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            for i in range(30):
                time.sleep(1)
                if self.is_running():
                    print("  Ollama started")
                    return True
                if i % 5 == 0:
                    print(f"  Waiting for Ollama... ({i + 1}/30)")
            print("  Ollama failed to start within 30 seconds")
            return False
        except Exception as e:
            print(f"  Error starting Ollama: {e}")
            return False

    def get_default_model(self) -> Optional[str]:
        try:
            resp = requests.get(f"{self.url}/api/tags", timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                if "models" in data and data["models"]:
                    return data["models"][0]["name"]
            return "llama2"
        except Exception:
            return "llama2"

    def query(self, messages, model, temperature=0.7, max_tokens=500):
        try:
            prompt = ""
            for msg in messages:
                role = msg["role"].capitalize()
                prompt += f"{role}: {msg['content']}\n"
            prompt += "Assistant: "

            resp = requests.post(
                f"{self.url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens,
                    },
                },
                timeout=30,
            )
            if resp.status_code == 200:
                return resp.json()["response"].strip()
            return None
        except Exception as e:
            print(f"Ollama query error: {e}")
            return None

    def stop(self):
        if self._process:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()

    @property
    def was_started_by_us(self):
        return self._process is not None
