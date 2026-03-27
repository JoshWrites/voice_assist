"""Query routing — directs user input to the right handler.

The router checks tool registry first (local tools), then falls back
to the LLM backend for complex queries.
"""

from ziggy.config import SHUTDOWN_PHRASE, RESOURCE_PROFILES, PROFILE_ALIASES


class QueryRouter:
    def __init__(self, tool_registry, backend, model, profile_mgr, conversation):
        self.tools = tool_registry
        self.backend = backend
        self.model = model
        self.profile_mgr = profile_mgr
        self.conversation = conversation

    def route(self, text):
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
        return "ai", self._query_ai_local_only(text)

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

    def _query_ai_local_only(self, text):
        messages = [
            {"role": "system",
             "content": ("You are a local AI assistant. Answer questions using only your "
                         "training data. If a question requires current information, real-time "
                         "data, or internet searches, respond with exactly: "
                         "'I need online resources to answer that properly.'")},
            {"role": "user",
             "content": f"Please provide a brief, spoken response to: {text}"},
        ]
        response = self.backend.query(
            messages, self.model,
            max_tokens=self.conversation.profile_settings.get("response_tokens", 1000),
        )
        if not response:
            return "Sorry, I couldn't process that request"
        # The "I need online resources" path is preserved but simplified
        # Full permission flow will be re-added via tools
        return response
