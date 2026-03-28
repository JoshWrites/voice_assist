"""Tests for ziggy.router — query routing logic."""

from unittest.mock import MagicMock, patch

from ziggy.router import QueryRouter
from ziggy.tools import ToolRegistry
from ziggy.config import RESOURCE_PROFILES


def _make_router(backend_response="AI says hello"):
    tools = ToolRegistry()
    tools.register("time", "Get time", ["time", "clock"], lambda t: "It is 3pm")

    backend = MagicMock()
    backend.query.return_value = backend_response

    profile = MagicMock()
    profile.switch_profile.return_value = "Switched to Minimal profile."
    profile.describe_current.return_value = "Running Standard mode."
    profile.list_profiles.return_value = "Minimal, Standard, Performance."
    profile.describe_memory.return_value = "Using 4GB of 16GB."

    conversation = MagicMock()
    conversation.profile_settings = RESOURCE_PROFILES["standard"].copy()

    return QueryRouter(tools, backend, "test-model", profile, conversation)


class TestRouterShutdown:
    def test_shutdown_phrase(self):
        router = _make_router()
        route_type, response = router.route("please take a break")
        assert route_type == "shutdown"
        assert "bye" in response.lower()

    def test_shutdown_case_insensitive(self):
        router = _make_router()
        route_type, _ = router.route("Take A Break now")
        assert route_type == "shutdown"


class TestRouterConversationReset:
    def test_new_conversation(self):
        router = _make_router()
        route_type, response = router.route("new conversation")
        assert route_type == "local"
        assert "fresh" in response.lower()

    def test_clear_history(self):
        router = _make_router()
        route_type, _ = router.route("clear history please")
        assert route_type == "local"

    def test_start_over(self):
        router = _make_router()
        route_type, _ = router.route("let's start over")
        assert route_type == "local"


class TestRouterProfileCommands:
    def test_switch_to_mode(self):
        router = _make_router()
        route_type, response = router.route("switch to minimal mode")
        assert route_type == "local"
        assert "Switched" in response

    def test_what_profile(self):
        router = _make_router()
        route_type, response = router.route("what profile am I using")
        assert route_type == "local"
        assert "Standard" in response

    def test_list_profiles(self):
        router = _make_router()
        route_type, response = router.route("what profiles are available")
        assert route_type == "local"

    def test_memory_usage(self):
        router = _make_router()
        route_type, response = router.route("how much memory are you using")
        assert route_type == "local"
        assert "4GB" in response


class TestRouterToolRouting:
    def test_routes_to_tool(self):
        router = _make_router()
        route_type, response = router.route("what time is it")
        assert route_type == "local"
        assert "3pm" in response

    def test_tool_returning_none_falls_through_to_ai(self):
        tools = ToolRegistry()
        tools.register("broken", "Broken tool", ["magic"], lambda t: None)
        backend = MagicMock()
        backend.query.return_value = "AI fallback"
        conversation = MagicMock()
        conversation.profile_settings = RESOURCE_PROFILES["standard"].copy()
        router = QueryRouter(tools, backend, "m", MagicMock(), conversation)
        route_type, response = router.route("do some magic")
        assert route_type == "ai"
        assert response == "AI fallback"


class TestRouterAIFallback:
    def test_unknown_query_goes_to_ai(self):
        router = _make_router()
        route_type, response = router.route("tell me about black holes")
        assert route_type == "ai"
        assert response == "AI says hello"

    def test_ai_returns_none(self):
        router = _make_router(backend_response=None)
        route_type, response = router.route("tell me a joke")
        assert route_type == "ai"
        assert "sorry" in response.lower()
