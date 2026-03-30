"""Tests for ziggy.backend.ringmaster — Ringmaster backend integration."""

from unittest.mock import patch, MagicMock

from ziggy.backend.ringmaster import RingmasterBackend, PROFILE_MODELS


class TestRingmasterBackend:
    def test_init_defaults(self):
        b = RingmasterBackend()
        assert b.name == "Ringmaster"
        assert "8420" in b.url
        assert not b.was_started_by_us

    @patch("ziggy.backend.ringmaster.requests.get")
    def test_is_running_true(self, mock_get):
        mock_get.return_value = MagicMock(status_code=200)
        assert RingmasterBackend().is_running() is True

    @patch("ziggy.backend.ringmaster.requests.get")
    def test_is_running_false(self, mock_get):
        mock_get.side_effect = ConnectionError
        assert RingmasterBackend().is_running() is False

    @patch("ziggy.backend.ringmaster.requests.get")
    def test_get_available_models(self, mock_get):
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: [{"name": "qwen3:8b"}, {"name": "qwen3:32b"}],
        )
        b = RingmasterBackend()
        models = b.get_available_models()
        assert len(models) == 2

    @patch("ziggy.backend.ringmaster.requests.post")
    def test_open_session(self, mock_post):
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"id": "session-123", "model": "qwen3:8b", "status": "open"},
        )
        b = RingmasterBackend()
        assert b.open_session("qwen3:8b") is True
        assert b._session_id == "session-123"
        assert b._session_model == "qwen3:8b"

    @patch("ziggy.backend.ringmaster.requests.post")
    def test_open_session_failure(self, mock_post):
        mock_post.return_value = MagicMock(status_code=503)
        b = RingmasterBackend()
        assert b.open_session("qwen3:8b") is False
        assert b._session_id is None

    @patch("ziggy.backend.ringmaster.requests.delete")
    def test_close_session(self, mock_delete):
        mock_delete.return_value = MagicMock(status_code=200)
        b = RingmasterBackend()
        b._session_id = "session-123"
        b._session_model = "qwen3:8b"
        b.close_session()
        assert b._session_id is None
        assert b._session_model is None

    @patch("ziggy.backend.ringmaster.requests.post")
    def test_query_success(self, mock_post):
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"result": "  Hello there  "},
        )
        b = RingmasterBackend()
        b._session_id = "session-123"
        result = b.query([{"role": "user", "content": "hi"}], "qwen3:8b")
        assert result == "Hello there"

    def test_query_no_session(self):
        b = RingmasterBackend()
        assert b.query([{"role": "user", "content": "hi"}], "m") is None

    def test_get_default_model_returns_session_model(self):
        b = RingmasterBackend()
        b._session_model = "qwen3:32b"
        assert b.get_default_model() == "qwen3:32b"


class TestModelSelection:
    @patch("ziggy.backend.ringmaster.requests.get")
    def test_selects_preferred_model(self, mock_get):
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: [{"name": "qwen3:4b"}, {"name": "qwen3:8b"}, {"name": "qwen3:32b"}],
        )
        b = RingmasterBackend()
        assert b.select_model_for_profile("minimal") == "qwen3:4b"
        assert b.select_model_for_profile("standard") == "qwen3:8b"
        assert b.select_model_for_profile("performance") == "qwen3:32b"

    @patch("ziggy.backend.ringmaster.requests.get")
    def test_falls_back_to_available_qwen3(self, mock_get):
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: [{"name": "qwen3:14b"}],
        )
        b = RingmasterBackend()
        assert b.select_model_for_profile("standard") == "qwen3:14b"

    @patch("ziggy.backend.ringmaster.requests.get")
    def test_falls_back_to_first_available(self, mock_get):
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: [{"name": "llama3:8b"}],
        )
        b = RingmasterBackend()
        assert b.select_model_for_profile("standard") == "llama3:8b"

    def test_profile_models_mapping(self):
        assert "minimal" in PROFILE_MODELS
        assert "standard" in PROFILE_MODELS
        assert "performance" in PROFILE_MODELS


class TestDetectBackendWithRingmaster:
    @patch("ziggy.backend.ringmaster.RingmasterBackend.is_running", return_value=True)
    def test_ringmaster_detected_first(self, *_):
        from ziggy.backend import detect_backend
        backend = detect_backend()
        assert backend is not None
        assert backend.name == "Ringmaster"

    @patch("ziggy.backend.ringmaster.RingmasterBackend.is_running", return_value=False)
    @patch("ziggy.backend.msty.MstyBackend.is_running", return_value=False)
    @patch("ziggy.backend.ollama.OllamaBackend.is_running", return_value=True)
    def test_falls_back_to_ollama(self, *_):
        from ziggy.backend import detect_backend
        backend = detect_backend()
        assert backend is not None
        assert backend.name == "Ollama"
