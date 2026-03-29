"""Tests for ziggy.speaker — speaker diarization."""

from unittest.mock import MagicMock
import numpy as np

from ziggy.speaker import SpeakerTracker


class TestSpeakerTrackerSetup:
    def test_setup_success(self):
        tracker = SpeakerTracker()
        tracker._encoder = MagicMock()  # simulate successful load
        assert tracker.is_active() is True

    def test_not_active_before_setup(self):
        tracker = SpeakerTracker()
        assert tracker.is_active() is False


class TestSpeakerIdentification:
    def _make_tracker(self):
        tracker = SpeakerTracker()
        tracker._encoder = MagicMock()
        return tracker

    def test_first_speaker_gets_speaker_1(self):
        tracker = self._make_tracker()
        tracker._encoder.embed_utterance.return_value = np.array([1.0, 0.0, 0.0])
        label = tracker.identify(b"\x00" * 32000, 16000)
        assert label == "Speaker 1"

    def test_same_voice_gets_same_label(self):
        tracker = self._make_tracker()
        embedding = np.array([1.0, 0.0, 0.0])
        tracker._encoder.embed_utterance.return_value = embedding
        label1 = tracker.identify(b"\x00" * 32000, 16000)
        label2 = tracker.identify(b"\x00" * 32000, 16000)
        assert label1 == label2 == "Speaker 1"

    def test_different_voice_gets_different_label(self):
        tracker = self._make_tracker()
        tracker._encoder.embed_utterance.side_effect = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
        ]
        label1 = tracker.identify(b"\x00" * 32000, 16000)
        label2 = tracker.identify(b"\x00" * 32000, 16000)
        assert label1 == "Speaker 1"
        assert label2 == "Speaker 2"

    def test_returns_name_if_set(self):
        tracker = self._make_tracker()
        tracker._encoder.embed_utterance.return_value = np.array([1.0, 0.0, 0.0])
        tracker.identify(b"\x00" * 32000, 16000)
        tracker.set_name("Speaker 1", "Josh")
        label = tracker.identify(b"\x00" * 32000, 16000)
        assert label == "Josh"

    def test_identify_returns_none_when_inactive(self):
        tracker = SpeakerTracker()
        label = tracker.identify(b"\x00" * 32000, 16000)
        assert label is None

    def test_identify_returns_none_on_error(self):
        tracker = self._make_tracker()
        tracker._encoder.embed_utterance.side_effect = Exception("bad audio")
        label = tracker.identify(b"\x00" * 32000, 16000)
        assert label is None


class TestNameMapping:
    def test_set_and_get_name(self):
        tracker = SpeakerTracker()
        tracker.set_name("Speaker 1", "Maya")
        assert tracker.get_display_label("Speaker 1") == "Maya"

    def test_unknown_speaker_returns_id(self):
        tracker = SpeakerTracker()
        assert tracker.get_display_label("Speaker 5") == "Speaker 5"

    def test_reset_clears_everything(self):
        tracker = SpeakerTracker()
        tracker._encoder = MagicMock()
        tracker._encoder.embed_utterance.return_value = np.array([1.0, 0.0])
        tracker.identify(b"\x00" * 32000, 16000)
        tracker.set_name("Speaker 1", "Josh")
        tracker.reset()
        assert tracker._embeddings == []
        assert tracker._names == {}
        assert tracker._next_id == 1
