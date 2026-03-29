"""Tests for ziggy.backend — Ollama and Msty backend clients."""

from unittest.mock import patch, MagicMock

from ziggy.backend import detect_backend
from ziggy.backend.ollama import OllamaBackend
from ziggy.backend.msty import MstyBackend


class TestOllamaBackend:
    def test_init_defaults(self):
        b = OllamaBackend()
        assert b.name == "Ollama"
        assert "11434" in b.url
        assert not b.was_started_by_us

    @patch("ziggy.backend.ollama.requests.get")
    def test_is_running_true(self, mock_get):
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"models": [{"name": "llama2"}]}
        )
        assert OllamaBackend().is_running() is True

    @patch("ziggy.backend.ollama.requests.get")
    def test_is_running_false_bad_status(self, mock_get):
        mock_get.return_value = MagicMock(status_code=500)
        assert OllamaBackend().is_running() is False

    @patch("ziggy.backend.ollama.requests.get")
    def test_is_running_false_no_models_key(self, mock_get):
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"error": "something"}
        )
        assert OllamaBackend().is_running() is False

    @patch("ziggy.backend.ollama.requests.get")
    def test_is_running_connection_error(self, mock_get):
        mock_get.side_effect = ConnectionError
        assert OllamaBackend().is_running() is False

    @patch("ziggy.backend.ollama.requests.get")
    def test_get_default_model(self, mock_get):
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"models": [{"name": "mistral:latest"}]}
        )
        assert OllamaBackend().get_default_model() == "mistral:latest"

    @patch("ziggy.backend.ollama.requests.get")
    def test_get_default_model_fallback(self, mock_get):
        mock_get.side_effect = ConnectionError
        assert OllamaBackend().get_default_model() == "llama2"

    @patch("ziggy.backend.ollama.requests.post")
    def test_query_success(self, mock_post):
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"response": "  Hello there  "}
        )
        b = OllamaBackend()
        result = b.query(
            [{"role": "user", "content": "hi"}],
            "llama2"
        )
        assert result == "Hello there"

    @patch("ziggy.backend.ollama.requests.post")
    def test_query_failure(self, mock_post):
        mock_post.return_value = MagicMock(status_code=500)
        assert OllamaBackend().query([{"role": "user", "content": "hi"}], "m") is None

    @patch("ziggy.backend.ollama.requests.post")
    def test_query_exception(self, mock_post):
        mock_post.side_effect = ConnectionError
        assert OllamaBackend().query([{"role": "user", "content": "hi"}], "m") is None

    def test_stop_no_process(self):
        b = OllamaBackend()
        b.stop()  # should not raise

    def test_stop_with_process(self):
        b = OllamaBackend()
        b._process = MagicMock()
        with patch("ziggy.backend.ollama.requests.get") as mock_get:
            mock_get.return_value = MagicMock(
                status_code=200, json=lambda: {"models": []}
            )
            b.stop()
        b._process.terminate.assert_called_once()

    @patch("ziggy.backend.ollama.requests.post")
    @patch("ziggy.backend.ollama.requests.get")
    def test_unload_models(self, mock_get, mock_post):
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"models": [{"name": "llama2"}, {"name": "mistral"}]}
        )
        mock_post.return_value = MagicMock(status_code=200)
        b = OllamaBackend()
        b.unload_models()
        assert mock_post.call_count == 2
        # Verify keep_alive=0 is sent
        for call in mock_post.call_args_list:
            assert call.kwargs["json"]["keep_alive"] == 0

    @patch("ziggy.backend.ollama.requests.get")
    def test_unload_models_handles_failure(self, mock_get):
        mock_get.side_effect = ConnectionError
        b = OllamaBackend()
        b.unload_models()  # should not raise


class TestMstyBackend:
    def test_init_defaults(self):
        b = MstyBackend()
        assert b.name == "Msty"
        assert "10000" in b.url

    @patch("ziggy.backend.msty.requests.get")
    def test_is_running_true(self, mock_get):
        def side_effect(url, **kwargs):
            if "/api/tags" in url:
                raise ConnectionError  # not Ollama
            return MagicMock(
                status_code=200,
                json=lambda: {"data": [{"id": "model1", "owned_by": "msty"}]}
            )
        mock_get.side_effect = side_effect
        assert MstyBackend().is_running() is True

    @patch("ziggy.backend.msty.requests.get")
    def test_is_running_false_when_ollama(self, mock_get):
        # First call (Ollama check) succeeds — means it's actually Ollama
        def side_effect(url, **kwargs):
            if "/api/tags" in url:
                return MagicMock(status_code=200)
            return MagicMock(status_code=200, json=lambda: {"data": []})
        mock_get.side_effect = side_effect
        assert MstyBackend().is_running() is False

    @patch("ziggy.backend.msty.requests.get")
    def test_get_default_model(self, mock_get):
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"data": [{"id": "llama3.2:latest"}]}
        )
        assert MstyBackend().get_default_model() == "llama3.2:latest"

    @patch("ziggy.backend.msty.requests.post")
    def test_query_success(self, mock_post):
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"choices": [{"message": {"content": "  Hi!  "}}]}
        )
        b = MstyBackend()
        result = b.query([{"role": "user", "content": "hello"}], "model")
        assert result == "Hi!"

    @patch("ziggy.backend.msty.requests.post")
    def test_query_failure(self, mock_post):
        mock_post.return_value = MagicMock(status_code=500)
        assert MstyBackend().query([{"role": "user", "content": "hi"}], "m") is None


class TestDetectBackend:
    @patch("ziggy.backend.msty.MstyBackend.is_running", return_value=False)
    @patch("ziggy.backend.ollama.OllamaBackend.is_running", return_value=True)
    def test_detects_ollama(self, *_):
        backend = detect_backend()
        assert backend is not None
        assert backend.name == "Ollama"

    @patch("ziggy.backend.msty.MstyBackend.is_running", return_value=True)
    def test_detects_msty_first(self, *_):
        backend = detect_backend()
        assert backend is not None
        assert backend.name == "Msty"

    @patch("ziggy.backend.msty.MstyBackend.is_running", return_value=False)
    @patch("ziggy.backend.ollama.OllamaBackend.is_running", return_value=False)
    def test_returns_none_when_nothing_running(self, *_):
        assert detect_backend() is None
