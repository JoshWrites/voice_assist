"""Tests for ziggy.tts — TTS engine abstraction and espeak implementation."""

from unittest.mock import patch, MagicMock

from ziggy.tts import create_tts_engine
from ziggy.tts.espeak import EspeakTTS


class TestEspeakTTS:
    @patch("ziggy.tts.espeak.subprocess.run")
    def test_setup_available(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        engine = EspeakTTS()
        engine.setup()
        assert engine.is_available() is True

    @patch("ziggy.tts.espeak.subprocess.run")
    def test_setup_not_available(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1)
        engine = EspeakTTS()
        engine.setup()
        assert engine.is_available() is False

    @patch.object(EspeakTTS, "_play_wav_bytes")
    @patch("ziggy.tts.espeak.subprocess.run")
    def test_speak_success(self, mock_run, mock_play):
        mock_run.return_value = MagicMock(returncode=0, stdout=b"RIFF_wav_data")
        engine = EspeakTTS()
        assert engine.speak("hello") is True
        mock_play.assert_called_once()

    @patch("ziggy.tts.espeak.subprocess.run")
    def test_speak_failure_no_wav(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stdout=b"")
        engine = EspeakTTS()
        assert engine.speak("hello") is False

    @patch("ziggy.tts.espeak.subprocess.run")
    def test_generate_wav_returns_none_on_error(self, mock_run):
        mock_run.side_effect = FileNotFoundError
        engine = EspeakTTS()
        assert engine._generate_wav("hello") is None

    @patch.object(EspeakTTS, "_play_wav_bytes")
    @patch("ziggy.tts.espeak.subprocess.run")
    def test_speak_sentence_async(self, mock_run, mock_play):
        mock_run.return_value = MagicMock(returncode=0, stdout=b"RIFF_wav_data")
        engine = EspeakTTS()
        player, cleanup = engine.speak_sentence_async("hello")
        assert player is not None
        assert cleanup is None
        # Player should have poll/terminate interface
        assert hasattr(player, "poll")
        assert hasattr(player, "terminate")

    @patch("ziggy.tts.espeak.subprocess.run")
    def test_speak_sentence_async_failure(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stdout=b"")
        engine = EspeakTTS()
        process, cleanup = engine.speak_sentence_async("hello")
        assert process is None


class TestCreateTTSEngine:
    @patch("ziggy.tts.espeak.subprocess.run")
    def test_minimal_profile_gets_espeak(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        engine = create_tts_engine({"tts_engine": "espeak"})
        assert isinstance(engine, EspeakTTS)

    @patch("ziggy.tts.espeak.subprocess.run")
    def test_fallback_to_espeak_when_piper_unavailable(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        # Piper won't be available (no binary), should fall back to espeak
        engine = create_tts_engine({"tts_engine": "piper"})
        assert engine.is_available() or isinstance(engine, EspeakTTS)
