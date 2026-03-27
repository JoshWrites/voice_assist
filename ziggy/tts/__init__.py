"""Text-to-speech engine abstraction.

Each TTS engine implements the same interface so they can be swapped
transparently. The factory picks the best available engine based on
the active profile.
"""

import abc


class TTSEngine(abc.ABC):
    """Base class for all TTS engines."""

    name: str = "base"

    @abc.abstractmethod
    def is_available(self) -> bool:
        """Check whether this engine is ready to use."""

    @abc.abstractmethod
    def speak(self, text: str) -> bool:
        """Speak *text* synchronously. Return True on success."""

    @abc.abstractmethod
    def speak_sentence_async(self, sentence: str):
        """Start speaking *sentence*, return (process, cleanup_path | None).

        The caller polls ``process.poll()`` and can ``process.terminate()``
        for interruption support.
        """

    def setup(self) -> None:
        """Optional one-time setup. Called during init."""


def create_tts_engine(profile_settings):
    """Build the best available TTS engine for the given profile."""
    from ziggy.tts.voxtral import VoxtralTTS
    from ziggy.tts.piper import PiperTTS
    from ziggy.tts.espeak import EspeakTTS

    preferred = profile_settings.get("tts_engine", "espeak")

    # Build in priority order based on profile preference
    if preferred == "voxtral":
        order = [VoxtralTTS, PiperTTS, EspeakTTS]
    elif preferred == "piper":
        order = [PiperTTS, VoxtralTTS, EspeakTTS]
    else:
        order = [EspeakTTS, PiperTTS, VoxtralTTS]

    engines_tried = []
    for cls in order:
        engine = cls()
        engine.setup()
        if engine.is_available():
            return engine
        engines_tried.append(cls.name)

    # Nothing worked — return espeak as last resort (it'll fail gracefully)
    print(f"  No TTS engine available (tried: {', '.join(engines_tried)})")
    return EspeakTTS()
