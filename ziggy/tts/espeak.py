"""espeak TTS engine — basic fallback, always available on most Linux systems.

Generates WAV via espeak --stdout and plays through PyAudio so audio goes
through the PipeWire/PulseAudio session like everything else.
"""

import io
import subprocess
import threading
import wave

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

    def _generate_wav(self, text: str) -> bytes | None:
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

    @staticmethod
    def _play_wav_bytes(wav_data: bytes) -> None:
        """Play WAV bytes through PyAudio (shares PipeWire session)."""
        import pyaudio

        with wave.open(io.BytesIO(wav_data)) as wf:
            p = pyaudio.PyAudio()
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
            p.terminate()

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
        """Start speaking in a background thread; return (thread, None).

        The caller can check thread.is_alive() and join() — the interface
        expects a process-like object with .poll() and .terminate(), so we
        wrap the thread to match.
        """
        wav = self._generate_wav(sentence)
        if not wav:
            return None, None

        class _Player:
            """Thin wrapper so the caller can treat this like a subprocess."""
            def __init__(self, wav_data, play_fn):
                self._done = threading.Event()
                self._thread = threading.Thread(target=self._run, args=(wav_data, play_fn), daemon=True)
                self._thread.start()

            def _run(self, wav_data, play_fn):
                try:
                    play_fn(wav_data)
                finally:
                    self._done.set()

            def poll(self):
                return 0 if self._done.is_set() else None

            def terminate(self):
                self._done.set()

        return _Player(wav, self._play_wav_bytes), None
