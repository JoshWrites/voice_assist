"""Speaker diarization — session-scoped speaker identification via voice embeddings."""

import numpy as np


class SpeakerTracker:
    """Identifies speakers by comparing voice embeddings within a session."""

    def __init__(self, threshold=0.75):
        self._encoder = None
        self._embeddings = []  # list of (speaker_id, embedding) tuples
        self._names = {}  # speaker_id -> name
        self._next_id = 1
        self._threshold = threshold

    def setup(self) -> bool:
        try:
            from resemblyzer import VoiceEncoder
            self._encoder = VoiceEncoder()
            print("  Speaker tracking ready")
            return True
        except Exception as e:
            print(f"  Speaker tracking unavailable: {e}")
            self._encoder = None
            return False

    def is_active(self) -> bool:
        return self._encoder is not None

    def reset(self):
        self._embeddings = []
        self._names = {}
        self._next_id = 1

    def identify(self, audio_bytes, sample_rate=16000):
        """Identify the speaker from raw audio bytes. Returns display label or None."""
        if not self.is_active():
            return None
        try:
            audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            if len(audio) < sample_rate * 0.5:
                return None
            embedding = self._encoder.embed_utterance(audio)
            return self._match_or_register(embedding)
        except Exception as e:
            print(f"  Speaker identification error: {e}")
            return None

    def _match_or_register(self, embedding):
        """Compare embedding against known speakers. Register if new."""
        best_sim = -1
        best_id = None
        for speaker_id, known_emb in self._embeddings:
            sim = self._cosine_similarity(embedding, known_emb)
            if sim > best_sim:
                best_sim = sim
                best_id = speaker_id

        if best_sim >= self._threshold and best_id is not None:
            return self.get_display_label(best_id)

        speaker_id = f"Speaker {self._next_id}"
        self._next_id += 1
        self._embeddings.append((speaker_id, embedding))
        return self.get_display_label(speaker_id)

    def set_name(self, speaker_id, name):
        """Map a speaker ID to a human name."""
        self._names[speaker_id] = name

    def get_display_label(self, speaker_id):
        """Return the name if known, else the raw speaker ID."""
        return self._names.get(speaker_id, speaker_id)

    @staticmethod
    def _cosine_similarity(a, b):
        dot = np.dot(a, b)
        norm = np.linalg.norm(a) * np.linalg.norm(b)
        if norm == 0:
            return 0.0
        return dot / norm
