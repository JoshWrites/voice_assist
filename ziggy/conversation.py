"""Conversation history and context management.

This module manages the conversation state, history tracking, and context
windowing for LLM queries. Future home for system prompt configuration
and tool-call message formatting.
"""


class ConversationManager:
    def __init__(self, profile_settings):
        self.history = []
        self.profile_settings = profile_settings

    def update_profile(self, profile_settings):
        self.profile_settings = profile_settings
        limit = self.profile_settings.get("history_limit", 25) * 2
        if len(self.history) > limit:
            self.history = self.history[-limit:]

    def add_exchange(self, user_text, assistant_text):
        self.history.append({"role": "user", "content": user_text})
        self.history.append({"role": "assistant", "content": assistant_text})

    def clear(self):
        self.history = []
        print("  Cleared conversation history")

    def build_messages(self, user_text, system_prompt=None):
        """Build a messages list with history, respecting token budget.

        Returns a list of message dicts ready to send to the backend.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        max_tokens = self.profile_settings.get("context_tokens", 16000)
        current_tokens = _estimate_tokens(system_prompt or "") + _estimate_tokens(user_text)

        # Walk history newest-first, prepend until budget is near
        included = []
        for msg in reversed(self.history):
            msg_tokens = _estimate_tokens(msg["content"])
            if current_tokens + msg_tokens > max_tokens - 2000:
                break
            included.insert(0, msg)
            current_tokens += msg_tokens

        messages.extend(included)
        messages.append({"role": "user", "content": user_text})

        print(f"  Context: ~{current_tokens} tokens, {len(included)} history messages")
        return messages


def _estimate_tokens(text):
    """Rough token estimate: average of char-based and word-based counts."""
    if not text:
        return 0
    char_estimate = len(text) / 4
    word_estimate = len(text.split()) / 0.75
    return int((char_estimate + word_estimate) / 2)
