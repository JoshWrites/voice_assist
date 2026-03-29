"""espeak TTS engine — basic fallback, always available on most Linux systems.

Generates WAV via espeak --stdout and plays through PyAudio so audio goes
through the PipeWire/PulseAudio session like everything else.
"""

import io
import subprocess
import threading
import wave

import pyaudio

from ziggy.tts import TTSEngine


class _Player:
    """Wraps a background audio playback thread with a subprocess-like interface."""

    def __init__(self, target, args):
        self._done = threading.Event()
        self._thread = threading.Thread(target=self._run, args=(target, args), daemon=True)
        self._thread.start()

    def _run(self, target, args):
        try:
            target(*args)
        finally:
            self._done.set()

    def poll(self):
        return 0 if self._done.is_set() else None

    def terminate(self):
        self._done.set()


class EspeakTTS(TTSEngine):
    name = "espeak"

    def __init__(self):
        self._available = False
        self._pyaudio = None

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

    def _get_pyaudio(self):
        if self._pyaudio is None:
            self._pyaudio = pyaudio.PyAudio()
        return self._pyaudio

    @staticmethod
    def _generate_wav(text: str) -> bytes | None:
        """Run espeak --stdout and return raw WAV bytes."""
        try:
            result = subprocess.run(
                ["espeak", "-s", "150", "-v", "en", "--stdout", text],
                capture_output=True,
            )
            if result.returncode == 0 and result.stdout:
                return result.stdout
        except Exception:
            pass
        return None

    def _play_wav_bytes(self, wav_data: bytes) -> None:
        """Play WAV bytes through PyAudio (shares PipeWire session)."""
        p = self._get_pyaudio()
        with wave.open(io.BytesIO(wav_data)) as wf:
            stream = p.open(
                format=p.get_format_from_width(wf.getsampwidth()),
                channels=wf.getnchannels(),
                rate=wf.getframerate(),
                output=True,
            )
            chunk = 1024
            data = wf.readframes(chunk)
            while data:
                stream.write(data)
                data = wf.readframes(chunk)
            stream.stop_stream()
            stream.close()

    def speak(self, text: str) -> bool:
        wav = self._generate_wav(text)
        if not wav:
            return False
        try:
            self._play_wav_bytes(wav)
            return True
        except Exception:
            return False

    def speak_sentence_async(self, sentence: str):
        """Start generating and playing in a background thread.

        Returns (_Player, None). The caller polls player.poll() and can
        call player.terminate() for interruption support.
        """
        return _Player(self._speak_async, (sentence,)), None

    def _speak_async(self, sentence: str):
        wav = self._generate_wav(sentence)
        if wav:
            self._play_wav_bytes(wav)
