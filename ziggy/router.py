"""Query routing — directs user input to the right handler.

The router checks tool registry first (local tools), then falls back
to the LLM backend for complex queries.
"""

from ziggy.config import (
    SHUTDOWN_PHRASE, RESOURCE_PROFILES, PROFILE_ALIASES,
    SYSTEM_PROMPT, THINK_TRIGGERS,
)

_ONLINE_SENTINEL = "i need online resources"


class QueryRouter:
    def __init__(self, tool_registry, backend, model, profile_mgr, conversation):
        self.tools = tool_registry
        self.backend = backend
        self.model = model
        self.profile_mgr = profile_mgr
        self.conversation = conversation

    def get_system_prompt(self):
        """Build the system prompt with current profile and model info."""
        return SYSTEM_PROMPT.format(
            profile_name=self.profile_mgr.settings["name"],
            model_name=self.model,
        )

    def route(self, text, speaker_label=None):
        """Route a query and return (route_type, response).

        route_type is one of: "shutdown", "local", "ai"
        """
        text_lower = text.lower().strip()

        # Shutdown
        if SHUTDOWN_PHRASE in text_lower:
            return "shutdown", "Okay, bye!"

        # Conversation reset
        reset_phrases = ["new conversation", "start over", "clear history", "fresh start"]
        if any(p in text_lower for p in reset_phrases):
            self.conversation.clear()
            return "local", "Starting fresh. What would you like to talk about?"

        # Profile management
        result = self._handle_profile_command(text_lower)
        if result:
            return "local", result

        # Check tool registry
        tool = self.tools.find_tool(text_lower)
        if tool:
            response = tool.handler(text)
            if response is not None:
                return "local", response

        # Default: send to LLM
        response = self._query_ai(text, speaker_label=speaker_label)

        # If the LLM says it needs online resources, try web search
        if _ONLINE_SENTINEL in response.lower():
            web_tool = self.tools.find_tool("search")
            if web_tool:
                web_response = web_tool.handler(text)
                if web_response is not None:
                    return "local", web_response

        return "ai", response

    def _handle_profile_command(self, text_lower):
        triggers = [
            ("switch to", "mode"), ("switched to", "mode"), ("change to", "mode"),
            ("use", "mode"), ("set", "mode"), ("enable", "mode"),
            ("switch to", "profile"), ("switched to", "profile"), ("change to", "profile"),
            ("use", "profile"), ("set", "profile"), ("enable", "profile"),
        ]
        for trigger, keyword in triggers:
            if trigger in text_lower and keyword in text_lower:
                name = text_lower.split(trigger)[-1].strip()
                name = name.replace("mode", "").replace("profile", "").strip()
                return self.profile_mgr.switch_profile(name)

        if any(p in text_lower for p in ["what profile", "which profile", "current profile"]):
            return self.profile_mgr.describe_current()

        if any(p in text_lower for p in ["what profiles", "available profiles", "list profiles"]):
            return self.profile_mgr.list_profiles()

        if "memory" in text_lower and any(w in text_lower for w in ["using", "usage", "status"]):
            return self.profile_mgr.describe_memory()

        return None

    def _should_think(self, text):
        """Check if the user wants the model to reason (standard/performance only)."""
        if self.profile_mgr.current_profile == "minimal":
            return False
        text_lower = text.lower()
        return any(trigger in text_lower for trigger in THINK_TRIGGERS)

    def _query_ai(self, text, speaker_label=None):
        """Send query to the LLM with the unified system prompt."""
        use_thinking = self._should_think(text)

        prompt = self.get_system_prompt()
        labeled_text = f"{speaker_label}: {text}" if speaker_label else text
        messages = self.conversation.build_messages(labeled_text, system_prompt=prompt)

        # Prepend /no_think unless the user asked for reasoning
        if not use_thinking:
            messages[-1]["content"] = "/no_think " + messages[-1]["content"]

        response = self.backend.query(
            messages, self.model,
            max_tokens=self.conversation.profile_settings.get("response_tokens", 1000),
        )
        if not response:
            return "Sorry, I couldn't process that request"
        return response
