"""Ringmaster backend — manages GPU and model selection via Ringmaster sessions.

Ringmaster is a workstation-resident daemon that manages GPU access.
Instead of talking to Ollama directly, Ziggy opens a session through
Ringmaster, which handles model loading, GPU selection, and resource
management.

If Ringmaster is not running, Ziggy falls back to direct Ollama access.
"""

import os
from typing import Optional

import requests

from ziggy.backend import LLMBackend

DEFAULT_URL = "http://localhost:8420"
CLIENT_ID = "ziggy-voice-assistant"
TOKEN_ENV_VAR = "RINGMASTER_TOKEN"

# Model recommendations per profile (VRAM-aware)
PROFILE_MODELS = {
    "minimal": "qwen3:4b",
    "standard": "qwen3:8b",
    "performance": "qwen3:32b",
}


class RingmasterBackend(LLMBackend):
    """LLM backend that routes through Ringmaster for GPU/model management."""

    name = "Ringmaster"

    def __init__(self, url=DEFAULT_URL, token=None):
        self.url = url
        self._token = token or os.environ.get("RINGMASTER_TOKEN", "")
        self._session_id = None
        self._session_model = None

    def _headers(self):
        h = {"Content-Type": "application/json"}
        if self._token:
            h["Authorization"] = f"Bearer {self._token}"
        return h

    def is_running(self) -> bool:
        try:
            resp = requests.get(f"{self.url}/health", timeout=2)
            return resp.status_code == 200
        except Exception:
            return False

    def start(self) -> bool:
        # Ringmaster is a system daemon — we don't start it
        return self.is_running()

    def get_status(self) -> Optional[dict]:
        """Get Ringmaster system status."""
        try:
            resp = requests.get(f"{self.url}/status", headers=self._headers(), timeout=5)
            if resp.status_code == 200:
                return resp.json()
        except Exception:
            pass
        return None

    def get_gpus(self) -> list:
        """Get available GPU info from Ringmaster."""
        try:
            resp = requests.get(f"{self.url}/gpus", headers=self._headers(), timeout=5)
            if resp.status_code == 200:
                return resp.json()
        except Exception:
            pass
        return []

    def get_available_models(self) -> list:
        """Get models available on the Ollama instance via Ringmaster."""
        try:
            resp = requests.get(f"{self.url}/models", headers=self._headers(), timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                return data.get("models", []) if isinstance(data, dict) else data
        except Exception:
            pass
        return []

    def select_model_for_profile(self, profile_name) -> str:
        """Pick the best model for the given profile, falling back to what's available."""
        preferred = PROFILE_MODELS.get(profile_name, "qwen3:8b")
        available = self.get_available_models()
        available_names = [m.get("name", "") for m in available] if available else []

        if preferred in available_names:
            return preferred

        # Try to find any qwen3 variant
        for name in available_names:
            if "qwen3" in name:
                return name

        # Fall back to first available model
        if available_names:
            return available_names[0]

        return preferred  # hope for the best

    def open_session(self, model: str) -> bool:
        """Open a Ringmaster session for interactive inference."""
        try:
            resp = requests.post(
                f"{self.url}/sessions",
                headers=self._headers(),
                json={
                    "model": model,
                    "client_id": CLIENT_ID,
                    "session_idle_timeout_seconds": 600,
                    "unattended_policy": "run",
                },
                timeout=30,
            )
            if resp.status_code in (200, 201):
                data = resp.json()
                self._session_id = data["id"]
                self._session_model = model
                print(f"  Ringmaster session opened: {self._session_id} ({model})")
                return True
            print(f"  Ringmaster session failed: HTTP {resp.status_code}")
        except Exception as e:
            print(f"  Ringmaster session error: {e}")
        return False

    def close_session(self):
        """Close the current Ringmaster session."""
        if not self._session_id:
            return
        try:
            requests.delete(
                f"{self.url}/sessions/{self._session_id}",
                headers=self._headers(),
                timeout=10,
            )
            print(f"  Ringmaster session closed: {self._session_id}")
        except Exception as e:
            print(f"  Ringmaster session close error: {e}")
        self._session_id = None
        self._session_model = None

    def get_default_model(self) -> Optional[str]:
        return self._session_model

    def query(self, messages, model, temperature=0.7, max_tokens=500) -> Optional[str]:
        """Send a query through the Ringmaster session."""
        if not self._session_id:
            return None

        # Build prompt from messages (Ringmaster session uses raw prompt, not chat format)
        prompt = ""
        for msg in messages:
            role = msg["role"].capitalize()
            prompt += f"{role}: {msg['content']}\n"
        prompt += "Assistant: "

        try:
            resp = requests.post(
                f"{self.url}/sessions/{self._session_id}/generate",
                headers=self._headers(),
                json={"prompt": prompt, "stream": False},
                timeout=60,
            )
            if resp.status_code == 200:
                data = resp.json()
                # Session generate returns the result directly
                return data.get("result", "").strip()
            print(f"  Ringmaster query failed: HTTP {resp.status_code}")
        except Exception as e:
            print(f"  Ringmaster query error: {e}")
        return None

    def stop(self):
        self.close_session()

    @property
    def was_started_by_us(self):
        return False  # Ringmaster is a system daemon
