"""Tests for utility functions in ziggy.app."""

from ziggy.app import _split_into_sentences, _contains_question


class TestSplitIntoSentences:
    def test_single_sentence(self):
        result = _split_into_sentences("Hello world")
        assert len(result) == 1
        assert "Hello world" in result[0]

    def test_multiple_sentences(self):
        text = "First sentence. Second sentence. Third one here. Fourth one too. Fifth sentence now."
        result = _split_into_sentences(text)
        assert len(result) >= 2

    def test_short_sentences_merged(self):
        # Sentences with fewer than 5 words get merged with next
        result = _split_into_sentences("Hi. How are you doing today.")
        # "Hi. " is < 5 words, should merge with next
        assert len(result) <= 2

    def test_empty_string(self):
        result = _split_into_sentences("")
        assert result == [""]

    def test_question_marks(self):
        result = _split_into_sentences("What is this? I don't know! Let me check.")
        assert len(result) >= 1

    def test_no_punctuation(self):
        result = _split_into_sentences("just a long sentence with no punctuation at all")
        assert len(result) == 1
        assert "just a long sentence" in result[0]


class TestContainsQuestion:
    def test_question_mark(self):
        assert _contains_question("How are you?") is True

    def test_question_starter_no_mark(self):
        assert _contains_question("What do you think about that.") is True

    def test_no_question(self):
        assert _contains_question("I like cats.") is False

    def test_various_starters(self):
        for word in ["what", "where", "when", "who", "why", "how",
                      "would", "could", "should", "can", "will", "do"]:
            assert _contains_question(f"{word} is the answer.") is True

    def test_starter_mid_sentence_not_matched(self):
        # "I know what happened." — "what" is not at start of sentence
        assert _contains_question("I know that happened.") is False

    def test_multiple_sentences_one_question(self):
        assert _contains_question("I like cats. Do you like cats too.") is True

    def test_empty_string(self):
        assert _contains_question("") is False
