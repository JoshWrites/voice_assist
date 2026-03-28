"""Tests for ziggy.config — verify configuration constants are consistent."""

from ziggy.config import (
    RESOURCE_PROFILES, PROFILE_ALIASES,
    WAKE_WORD, SHUTDOWN_PHRASE,
    AUDIO_SAMPLE_RATE, AUDIO_CHUNK_SIZE, AUDIO_CHANNELS,
)


class TestResourceProfiles:
    def test_all_profiles_exist(self):
        assert "minimal" in RESOURCE_PROFILES
        assert "standard" in RESOURCE_PROFILES
        assert "performance" in RESOURCE_PROFILES

    def test_profiles_have_required_keys(self):
        required = [
            "name", "description", "context_tokens", "history_limit",
            "response_tokens", "recording_conversational", "recording_command",
            "tts_engine",
        ]
        for profile_name, profile in RESOURCE_PROFILES.items():
            for key in required:
                assert key in profile, f"{profile_name} missing '{key}'"

    def test_context_tokens_ordering(self):
        assert RESOURCE_PROFILES["minimal"]["context_tokens"] < \
               RESOURCE_PROFILES["standard"]["context_tokens"] < \
               RESOURCE_PROFILES["performance"]["context_tokens"]

    def test_history_limit_ordering(self):
        assert RESOURCE_PROFILES["minimal"]["history_limit"] < \
               RESOURCE_PROFILES["standard"]["history_limit"] < \
               RESOURCE_PROFILES["performance"]["history_limit"]


class TestProfileAliases:
    def test_all_aliases_point_to_valid_profiles(self):
        for alias, target in PROFILE_ALIASES.items():
            assert target in RESOURCE_PROFILES, f"Alias '{alias}' points to unknown profile '{target}'"

    def test_gaming_is_minimal(self):
        assert PROFILE_ALIASES["gaming"] == "minimal"

    def test_research_is_performance(self):
        assert PROFILE_ALIASES["research"] == "performance"


class TestConstants:
    def test_wake_word_is_lowercase(self):
        assert WAKE_WORD == WAKE_WORD.lower()

    def test_shutdown_phrase_is_lowercase(self):
        assert SHUTDOWN_PHRASE == SHUTDOWN_PHRASE.lower()

    def test_audio_settings_reasonable(self):
        assert 8000 <= AUDIO_SAMPLE_RATE <= 48000
        assert AUDIO_CHUNK_SIZE > 0
        assert AUDIO_CHANNELS >= 1
