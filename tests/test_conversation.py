"""Tests for ziggy.conversation — history management and token budgeting."""

from ziggy.conversation import ConversationManager, _estimate_tokens


class TestEstimateTokens:
    def test_empty_string(self):
        assert _estimate_tokens("") == 0

    def test_none_input(self):
        assert _estimate_tokens(None) == 0

    def test_short_text(self):
        result = _estimate_tokens("hello world")
        assert result > 0

    def test_longer_text_gives_more_tokens(self):
        short = _estimate_tokens("hi")
        long = _estimate_tokens("this is a much longer sentence with many words in it")
        assert long > short


class TestConversationManager:
    def _make(self, **overrides):
        settings = {"history_limit": 25, "context_tokens": 16000, "response_tokens": 1000}
        settings.update(overrides)
        return ConversationManager(settings)

    def test_init_empty_history(self):
        cm = self._make()
        assert cm.history == []

    def test_add_exchange(self):
        cm = self._make()
        cm.add_exchange("hello", "hi there")
        assert len(cm.history) == 2
        assert cm.history[0] == {"role": "user", "content": "hello"}
        assert cm.history[1] == {"role": "assistant", "content": "hi there"}

    def test_add_multiple_exchanges(self):
        cm = self._make()
        cm.add_exchange("one", "two")
        cm.add_exchange("three", "four")
        assert len(cm.history) == 4

    def test_clear(self):
        cm = self._make()
        cm.add_exchange("hello", "world")
        cm.clear()
        assert cm.history == []

    def test_build_messages_no_history(self):
        cm = self._make()
        msgs = cm.build_messages("what time is it")
        assert msgs[-1] == {"role": "user", "content": "what time is it"}

    def test_build_messages_with_system_prompt(self):
        cm = self._make()
        msgs = cm.build_messages("hello", system_prompt="you are helpful")
        assert msgs[0] == {"role": "system", "content": "you are helpful"}
        assert msgs[-1] == {"role": "user", "content": "hello"}

    def test_build_messages_includes_history(self):
        cm = self._make()
        cm.add_exchange("first question", "first answer")
        msgs = cm.build_messages("second question")
        # Should have: history (2 msgs) + current user msg
        assert len(msgs) == 3
        assert msgs[0]["content"] == "first question"
        assert msgs[1]["content"] == "first answer"
        assert msgs[2]["content"] == "second question"

    def test_build_messages_respects_token_budget(self):
        cm = self._make(context_tokens=100)
        # Add enough history to exceed budget
        for i in range(50):
            cm.add_exchange(f"long question number {i} " * 20, f"long answer number {i} " * 20)
        msgs = cm.build_messages("final question")
        # Should have trimmed old history
        assert len(msgs) < 102  # way less than 100 exchanges
        assert msgs[-1]["content"] == "final question"

    def test_update_profile_trims_history(self):
        cm = self._make(history_limit=50)
        for i in range(60):
            cm.add_exchange(f"q{i}", f"a{i}")
        assert len(cm.history) == 120
        cm.update_profile({"history_limit": 5})
        assert len(cm.history) == 10  # 5 * 2

    def test_update_profile_no_trim_if_within_limit(self):
        cm = self._make()
        cm.add_exchange("q", "a")
        cm.update_profile({"history_limit": 25})
        assert len(cm.history) == 2
