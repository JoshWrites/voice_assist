"""Msty backend — OpenAI-compatible local LLM server."""

import subprocess
import time
from typing import Optional

import requests

from ziggy.backend import LLMBackend

DEFAULT_URL = "http://localhost:10000"


class MstyBackend(LLMBackend):
    name = "Msty"

    def __init__(self, url=DEFAULT_URL):
        self.url = url
        self._process = None

    def is_running(self) -> bool:
        try:
            # Make sure it's not actually Ollama on this port
            try:
                r = requests.get(f"{self.url}/api/tags", timeout=1)
                if r.status_code == 200:
                    return False  # That's Ollama
            except Exception:
                pass

            resp = requests.get(f"{self.url}/v1/models", timeout=2)
            if resp.status_code == 200:
                data = resp.json()
                if "data" in data and data["data"]:
                    first = data["data"][0]
                    if first.get("owned_by") == "library":
                        return False  # Ollama masquerading
                    return True
            return False
        except Exception:
            return False

    def start(self) -> bool:
        try:
            result = subprocess.run(
                ["which", "msty"], capture_output=True, text=True
            )
            if result.returncode != 0:
                print("  Msty not found")
                return False

            print("  Starting Msty backend...")
            self._process = subprocess.Popen(
                ["msty", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            for i in range(30):
                time.sleep(1)
                if self.is_running():
                    print("  Msty started")
                    return True
                if i % 5 == 0:
                    print(f"  Waiting for Msty... ({i + 1}/30)")
            print("  Msty failed to start within 30 seconds")
            return False
        except Exception as e:
            print(f"  Error starting Msty: {e}")
            return False

    def get_default_model(self) -> Optional[str]:
        try:
            resp = requests.get(f"{self.url}/v1/models", timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                if "data" in data and data["data"]:
                    return data["data"][0]["id"]
            return "llama3.2:latest"
        except Exception:
            return "llama3.2:latest"

    def query(self, messages, model, temperature=0.7, max_tokens=500):
        try:
            resp = requests.post(
                f"{self.url}/v1/chat/completions",
                json={
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
                timeout=30,
            )
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"].strip()
            return None
        except Exception as e:
            print(f"Msty query error: {e}")
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
