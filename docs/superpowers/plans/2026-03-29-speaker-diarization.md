# Speaker Diarization (Phase 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add in-session speaker identification so the LLM receives speaker-attributed transcripts ("Josh: what's the weather").

**Architecture:** A new `SpeakerTracker` module uses resemblyzer (CPU) to generate voice embeddings per utterance and compare against known session speakers. It sits between audio capture and STT in the pipeline. Disabled in minimal profile.

**Tech Stack:** resemblyzer, numpy, Python 3.12

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `ziggy/speaker.py` | Create | SpeakerTracker class — embedding, comparison, name mapping |
| `ziggy/config.py` | Modify | Add `speaker_tracking` to profiles, speaker prompt text |
| `ziggy/app.py` | Modify | Initialize SpeakerTracker, wire into command flow, add introduction |
| `ziggy/router.py` | Modify | Accept and pass through speaker labels in transcripts |
| `tests/test_speaker.py` | Create | All SpeakerTracker tests |
| `tests/test_router.py` | Modify | Test speaker-labeled transcripts |

---

### Task 1: Install resemblyzer and create SpeakerTracker skeleton

**Files:**
- Create: `ziggy/speaker.py`
- Create: `tests/test_speaker.py`

- [ ] **Step 1: Install resemblyzer**

```bash
.venv/bin/pip install resemblyzer
```

Expected: Successfully installed resemblyzer and dependencies (librosa, etc.)

- [ ] **Step 2: Write test for SpeakerTracker init and setup**

Write to `tests/test_speaker.py`:

```python
"""Tests for ziggy.speaker — speaker diarization."""

from unittest.mock import patch, MagicMock
import numpy as np

from ziggy.speaker import SpeakerTracker


class TestSpeakerTrackerSetup:
    @patch("ziggy.speaker.VoiceEncoder")
    def test_setup_success(self, mock_encoder_cls):
        tracker = SpeakerTracker()
        assert tracker.setup() is True
        assert tracker.is_active() is True

    @patch("ziggy.speaker.VoiceEncoder", side_effect=ImportError)
    def test_setup_failure(self, mock_encoder_cls):
        tracker = SpeakerTracker()
        assert tracker.setup() is False
        assert tracker.is_active() is False

    def test_not_active_before_setup(self):
        tracker = SpeakerTracker()
        assert tracker.is_active() is False
```

- [ ] **Step 3: Run test to verify it fails**

```bash
.venv/bin/python -m pytest tests/test_speaker.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'ziggy.speaker'`

- [ ] **Step 4: Write minimal SpeakerTracker**

Write to `ziggy/speaker.py`:

```python
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
```

- [ ] **Step 5: Run test to verify it passes**

```bash
.venv/bin/python -m pytest tests/test_speaker.py -v
```

Expected: 3 passed

- [ ] **Step 6: Commit**

```bash
git add ziggy/speaker.py tests/test_speaker.py
git commit -m "feat: add SpeakerTracker skeleton with setup and activation"
```

---

### Task 2: Implement speaker identification

**Files:**
- Modify: `ziggy/speaker.py`
- Modify: `tests/test_speaker.py`

- [ ] **Step 1: Write tests for identify()**

Append to `tests/test_speaker.py`:

```python
class TestSpeakerIdentification:
    def _make_tracker(self):
        """Create a tracker with a mocked encoder."""
        tracker = SpeakerTracker()
        tracker._encoder = MagicMock()
        return tracker

    def test_first_speaker_gets_speaker_1(self):
        tracker = self._make_tracker()
        tracker._encoder.embed_utterance.return_value = np.array([1.0, 0.0, 0.0])
        label = tracker.identify(b"\x00" * 32000, 16000)
        assert label == "Speaker 1"

    def test_same_voice_gets_same_label(self):
        tracker = self._make_tracker()
        embedding = np.array([1.0, 0.0, 0.0])
        tracker._encoder.embed_utterance.return_value = embedding
        label1 = tracker.identify(b"\x00" * 32000, 16000)
        label2 = tracker.identify(b"\x00" * 32000, 16000)
        assert label1 == label2 == "Speaker 1"

    def test_different_voice_gets_different_label(self):
        tracker = self._make_tracker()
        tracker._encoder.embed_utterance.side_effect = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
        ]
        label1 = tracker.identify(b"\x00" * 32000, 16000)
        label2 = tracker.identify(b"\x00" * 32000, 16000)
        assert label1 == "Speaker 1"
        assert label2 == "Speaker 2"

    def test_returns_name_if_set(self):
        tracker = self._make_tracker()
        tracker._encoder.embed_utterance.return_value = np.array([1.0, 0.0, 0.0])
        tracker.identify(b"\x00" * 32000, 16000)
        tracker.set_name("Speaker 1", "Josh")
        label = tracker.identify(b"\x00" * 32000, 16000)
        assert label == "Josh"

    def test_identify_returns_none_when_inactive(self):
        tracker = SpeakerTracker()
        label = tracker.identify(b"\x00" * 32000, 16000)
        assert label is None

    def test_identify_returns_none_on_error(self):
        tracker = self._make_tracker()
        tracker._encoder.embed_utterance.side_effect = Exception("bad audio")
        label = tracker.identify(b"\x00" * 32000, 16000)
        assert label is None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/test_speaker.py::TestSpeakerIdentification -v
```

Expected: FAIL — `identify` and `set_name` not implemented

- [ ] **Step 3: Implement identify() and helpers**

Add to `ziggy/speaker.py` inside the `SpeakerTracker` class, after `reset()`:

```python
    def identify(self, audio_bytes, sample_rate=16000):
        """Identify the speaker from raw audio bytes. Returns display label or None."""
        if not self.is_active():
            return None
        try:
            audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            if len(audio) < sample_rate * 0.5:
                return None  # too short to embed reliably
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/test_speaker.py -v
```

Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add ziggy/speaker.py tests/test_speaker.py
git commit -m "feat: implement speaker identification with cosine similarity matching"
```

---

### Task 3: Add name mapping and reset tests

**Files:**
- Modify: `tests/test_speaker.py`

- [ ] **Step 1: Write tests for name mapping and reset**

Append to `tests/test_speaker.py`:

```python
class TestNameMapping:
    def test_set_and_get_name(self):
        tracker = SpeakerTracker()
        tracker.set_name("Speaker 1", "Maya")
        assert tracker.get_display_label("Speaker 1") == "Maya"

    def test_unknown_speaker_returns_id(self):
        tracker = SpeakerTracker()
        assert tracker.get_display_label("Speaker 5") == "Speaker 5"

    def test_reset_clears_everything(self):
        tracker = SpeakerTracker()
        tracker._encoder = MagicMock()
        tracker._encoder.embed_utterance.return_value = np.array([1.0, 0.0])
        tracker.identify(b"\x00" * 32000, 16000)
        tracker.set_name("Speaker 1", "Josh")
        tracker.reset()
        assert tracker._embeddings == []
        assert tracker._names == {}
        assert tracker._next_id == 1
```

- [ ] **Step 2: Run tests**

```bash
.venv/bin/python -m pytest tests/test_speaker.py -v
```

Expected: 12 passed

- [ ] **Step 3: Commit**

```bash
git add tests/test_speaker.py
git commit -m "test: add name mapping and reset tests for SpeakerTracker"
```

---

### Task 4: Add speaker_tracking to profile config

**Files:**
- Modify: `ziggy/config.py`
- Modify: `tests/test_config.py`

- [ ] **Step 1: Write test for speaker_tracking in profiles**

Append to `tests/test_config.py` inside `TestResourceProfiles`:

```python
    def test_minimal_has_no_speaker_tracking(self):
        assert RESOURCE_PROFILES["minimal"]["speaker_tracking"] is False

    def test_standard_has_speaker_tracking(self):
        assert RESOURCE_PROFILES["standard"]["speaker_tracking"] is True

    def test_performance_has_speaker_tracking(self):
        assert RESOURCE_PROFILES["performance"]["speaker_tracking"] is True
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/test_config.py::TestResourceProfiles::test_minimal_has_no_speaker_tracking -v
```

Expected: FAIL — KeyError: 'speaker_tracking'

- [ ] **Step 3: Add speaker_tracking to profiles**

In `ziggy/config.py`, add `"speaker_tracking"` to each profile dict:

In `"minimal"` add:
```python
        "speaker_tracking": False,
```

In `"standard"` add:
```python
        "speaker_tracking": True,
```

In `"performance"` add:
```python
        "speaker_tracking": True,
```

Also add to `test_profiles_have_required_keys` in `tests/test_config.py` — add `"speaker_tracking"` to the `required` list.

- [ ] **Step 4: Append speaker awareness to SYSTEM_PROMPT**

In `ziggy/config.py`, append to the end of `SYSTEM_PROMPT` (before the closing `\`):

```python
When transcripts include speaker labels (like "Josh:" or "Speaker 2:"), \
you're hearing from different people. Use their names when known. \
If someone introduces themselves, remember their name for this conversation.\
```

- [ ] **Step 5: Run tests**

```bash
.venv/bin/python -m pytest tests/test_config.py -v
```

Expected: All passed (including 3 new ones)

- [ ] **Step 6: Commit**

```bash
git add ziggy/config.py tests/test_config.py
git commit -m "feat: add speaker_tracking flag to profiles and speaker awareness to system prompt"
```

---

### Task 5: Wire SpeakerTracker into VoiceAssistant setup

**Files:**
- Modify: `ziggy/app.py`

- [ ] **Step 1: Add import and initialization**

In `ziggy/app.py`, add import at the top with the other imports:

```python
from ziggy.speaker import SpeakerTracker
```

In `VoiceAssistant.__init__()`, after `self.tools = ToolRegistry()`, add:

```python
        self.speaker = None
```

In `VoiceAssistant._setup()`, after `self.conversation = ConversationManager(self.profile.settings)` and before `self._register_tools()`, add:

```python
            if self.profile.settings.get("speaker_tracking"):
                self.speaker = SpeakerTracker()
                if not self.speaker.setup():
                    self.speaker = None
```

- [ ] **Step 2: Run existing tests to verify nothing broke**

```bash
.venv/bin/python -m pytest tests/ -v 2>&1 | tail -5
```

Expected: All 124+ tests pass

- [ ] **Step 3: Commit**

```bash
git add ziggy/app.py
git commit -m "feat: initialize SpeakerTracker in VoiceAssistant setup"
```

---

### Task 6: Add session introduction flow

**Files:**
- Modify: `ziggy/app.py`

- [ ] **Step 1: Add introduction to _startup_message**

In `ziggy/app.py`, modify `_startup_message()`. After the existing `self.speak(msg, allow_interruption=False)`, add the introduction flow:

```python
        if self.speaker and self.speaker.is_active():
            self.speak("Hi, I'm Ziggy. What's your name?", allow_interruption=False)
            audio = self.audio.record_command(profile_settings=self.profile.settings)
            if audio:
                speaker_label = self.speaker.identify(audio, self.audio.sample_rate)
                name_text = self.stt.transcribe(audio, self.audio.get_sample_size())
                if name_text:
                    name = self._extract_name(name_text)
                    if name and speaker_label:
                        self.speaker.set_name(speaker_label, name)
                        self.speak(f"Nice to meet you, {name}!", allow_interruption=False)
                    else:
                        self.speak("Nice to meet you!", allow_interruption=False)
                else:
                    self.speak("No worries. Let's get started!", allow_interruption=False)
```

- [ ] **Step 2: Add _extract_name helper**

Add this method to `VoiceAssistant`, after `_startup_message`:

```python
    @staticmethod
    def _extract_name(text):
        """Try to extract a name from a response like 'I'm Josh' or 'my name is Josh'."""
        text = text.strip()
        for prefix in ["i'm ", "i am ", "my name is ", "it's ", "they call me ", "name's "]:
            if text.lower().startswith(prefix):
                name = text[len(prefix):].strip().split()[0] if text[len(prefix):].strip() else None
                if name:
                    return name.capitalize()
        # If it's just one or two words, treat it as a name
        words = text.split()
        if 1 <= len(words) <= 2:
            return " ".join(w.capitalize() for w in words)
        return None
```

- [ ] **Step 3: Run existing tests**

```bash
.venv/bin/python -m pytest tests/ -v 2>&1 | tail -5
```

Expected: All pass (this is runtime code, not breaking existing logic)

- [ ] **Step 4: Commit**

```bash
git add ziggy/app.py
git commit -m "feat: add session introduction flow — Ziggy asks user's name at startup"
```

---

### Task 7: Wire speaker identification into command handling

**Files:**
- Modify: `ziggy/app.py`

- [ ] **Step 1: Add speaker label to _handle_voice_command**

In `ziggy/app.py`, in `_handle_voice_command()`, find the section after audio recording and before STT:

```python
            audio_data = self.audio.record_command(profile_settings=self.profile.settings)
            if not audio_data:
                self.speak("I didn't hear anything", allow_interruption=False)
                return

            command_text = self.stt.transcribe(audio_data, self.audio.get_sample_size())
```

Insert speaker identification between recording and transcription:

```python
            audio_data = self.audio.record_command(profile_settings=self.profile.settings)
            if not audio_data:
                self.speak("I didn't hear anything", allow_interruption=False)
                return

            speaker_label = None
            if self.speaker and self.speaker.is_active():
                speaker_label = self.speaker.identify(audio_data, self.audio.sample_rate)

            command_text = self.stt.transcribe(audio_data, self.audio.get_sample_size())
```

Then where `command_text` is passed to the router, prepend the speaker label:

Find:
```python
            print(f"Command: '{command_text}'")
            route_type, response = self.router.route(command_text)
```

Replace with:
```python
            if speaker_label:
                labeled_text = f"{speaker_label}: {command_text}"
            else:
                labeled_text = command_text
            print(f"Command: '{labeled_text}'")
            route_type, response = self.router.route(command_text, speaker_label=speaker_label)
```

- [ ] **Step 2: Do the same in _conversational_loop**

In `_conversational_loop()`, find:

```python
            text = self.stt.transcribe(audio, self.audio.get_sample_size())
            if not text:
                break

            print(f"User answered: '{text}'")
```

Replace with:

```python
            speaker_label = None
            if self.speaker and self.speaker.is_active():
                speaker_label = self.speaker.identify(audio, self.audio.sample_rate)

            text = self.stt.transcribe(audio, self.audio.get_sample_size())
            if not text:
                break

            labeled_text = f"{speaker_label}: {text}" if speaker_label else text
            print(f"User answered: '{labeled_text}'")
```

- [ ] **Step 3: Add new-speaker introduction in command handling**

After `route_type, response = self.router.route(...)` in `_handle_voice_command`, add logic to ask a new speaker's name. Find the block that starts with `if route_type == "shutdown":` and add before it:

```python
            # If this is a new unnamed speaker, ask their name after responding
            new_speaker = (
                speaker_label
                and speaker_label.startswith("Speaker ")
                and self.speaker
                and self.speaker.is_active()
            )
```

Then after the response is spoken (after `was_interrupted = self.speak(response, allow_interruption=True)`), add:

```python
            if new_speaker and not was_interrupted:
                self.speak("By the way, I don't think we've met. I'm Ziggy — what's your name?",
                           allow_interruption=False)
                name_audio = self.audio.record_command(profile_settings=self.profile.settings)
                if name_audio:
                    name_text = self.stt.transcribe(name_audio, self.audio.get_sample_size())
                    if name_text:
                        name = self._extract_name(name_text)
                        if name:
                            self.speaker.set_name(speaker_label, name)
                            self.speak(f"Nice to meet you, {name}!", allow_interruption=False)
```

- [ ] **Step 4: Run existing tests**

```bash
.venv/bin/python -m pytest tests/ -v 2>&1 | tail -5
```

Expected: All pass

- [ ] **Step 5: Commit**

```bash
git add ziggy/app.py
git commit -m "feat: wire speaker identification into command and conversation flows"
```

---

### Task 8: Update router to handle speaker labels

**Files:**
- Modify: `ziggy/router.py`
- Modify: `tests/test_router.py`

- [ ] **Step 1: Write test for speaker-labeled routing**

Append to `tests/test_router.py`:

```python
class TestSpeakerLabels:
    def test_route_with_speaker_label(self):
        router = _make_router()
        router.conversation.build_messages.return_value = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "Josh: tell me about black holes"},
        ]
        route_type, response = router.route("tell me about black holes", speaker_label="Josh")
        # The LLM message should include the speaker label
        call_args = router.backend.query.call_args
        messages = call_args[0][0]
        user_msg = messages[-1]["content"]
        assert "Josh:" in user_msg

    def test_route_without_speaker_label(self):
        router = _make_router()
        router.conversation.build_messages.return_value = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "tell me about black holes"},
        ]
        route_type, response = router.route("tell me about black holes")
        call_args = router.backend.query.call_args
        messages = call_args[0][0]
        user_msg = messages[-1]["content"]
        assert ":" not in user_msg.split("/no_think ")[-1].split(" ")[0]

    def test_tool_routing_strips_speaker_label(self):
        router = _make_router()
        route_type, response = router.route("what time is it", speaker_label="Josh")
        assert route_type == "local"
        assert "3pm" in response
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/test_router.py::TestSpeakerLabels -v
```

Expected: FAIL — `route()` doesn't accept `speaker_label`

- [ ] **Step 3: Update router.route() to accept speaker_label**

In `ziggy/router.py`, update the `route()` method signature:

```python
    def route(self, text, speaker_label=None):
```

In `_query_ai()`, update the signature and pass the label through:

```python
    def _query_ai(self, text, speaker_label=None):
```

In `_query_ai`, change how the user message is built. Replace:

```python
        messages = self.conversation.build_messages(text, system_prompt=prompt)
```

With:

```python
        labeled_text = f"{speaker_label}: {text}" if speaker_label else text
        messages = self.conversation.build_messages(labeled_text, system_prompt=prompt)
```

Update the call from `route()` to `_query_ai()`:

```python
        response = self._query_ai(text, speaker_label=speaker_label)
```

Also update the conversation history in `app.py` — in `_handle_voice_command`, where `add_exchange` is called, use the labeled text:

In `app.py`, change:
```python
            if route_type == "ai":
                self.conversation.add_exchange(command_text, response)
```
To:
```python
            if route_type == "ai":
                self.conversation.add_exchange(labeled_text, response)
```

And in `_conversational_loop`, change:
```python
            self.conversation.add_exchange(text, ai_response)
```
To:
```python
            self.conversation.add_exchange(labeled_text, ai_response)
```

- [ ] **Step 4: Run all tests**

```bash
.venv/bin/python -m pytest tests/ -v 2>&1 | tail -5
```

Expected: All pass

- [ ] **Step 5: Commit**

```bash
git add ziggy/router.py ziggy/app.py tests/test_router.py
git commit -m "feat: pass speaker labels through router to LLM and conversation history"
```

---

### Task 9: Add _extract_name tests

**Files:**
- Modify: `tests/test_app_utils.py`

- [ ] **Step 1: Write tests for _extract_name**

Append to `tests/test_app_utils.py`:

```python
from ziggy.app import VoiceAssistant


class TestExtractName:
    def test_im_prefix(self):
        assert VoiceAssistant._extract_name("I'm Josh") == "Josh"

    def test_my_name_is_prefix(self):
        assert VoiceAssistant._extract_name("my name is Maya") == "Maya"

    def test_i_am_prefix(self):
        assert VoiceAssistant._extract_name("I am David") == "David"

    def test_single_word(self):
        assert VoiceAssistant._extract_name("Josh") == "Josh"

    def test_two_words(self):
        assert VoiceAssistant._extract_name("josh levine") == "Josh Levine"

    def test_long_sentence_returns_none(self):
        assert VoiceAssistant._extract_name("I don't want to tell you my name") is None

    def test_empty_string(self):
        assert VoiceAssistant._extract_name("") is None

    def test_capitalizes(self):
        assert VoiceAssistant._extract_name("i'm josh") == "Josh"
```

- [ ] **Step 2: Run tests**

```bash
.venv/bin/python -m pytest tests/test_app_utils.py::TestExtractName -v
```

Expected: All pass (implementation was added in Task 6)

- [ ] **Step 3: Commit**

```bash
git add tests/test_app_utils.py
git commit -m "test: add _extract_name tests"
```

---

### Task 10: Final integration test and cleanup

**Files:**
- All modified files

- [ ] **Step 1: Run full test suite**

```bash
.venv/bin/python -m pytest tests/ -v
```

Expected: All tests pass (should be ~135+)

- [ ] **Step 2: Quick smoke test — verify import works**

```bash
.venv/bin/python -c "
from ziggy.speaker import SpeakerTracker
t = SpeakerTracker()
print(f'Setup: {t.setup()}')
print(f'Active: {t.is_active()}')
"
```

Expected: Setup: True, Active: True (resemblyzer loads)

- [ ] **Step 3: Verify minimal profile skips speaker tracking**

```bash
.venv/bin/python -c "
from ziggy.config import RESOURCE_PROFILES
print(f'Minimal speaker_tracking: {RESOURCE_PROFILES[\"minimal\"][\"speaker_tracking\"]}')
print(f'Standard speaker_tracking: {RESOURCE_PROFILES[\"standard\"][\"speaker_tracking\"]}')
print(f'Performance speaker_tracking: {RESOURCE_PROFILES[\"performance\"][\"speaker_tracking\"]}')
"
```

Expected: False, True, True

- [ ] **Step 4: Commit all remaining changes**

```bash
git add -A
git status
git commit -m "feat: speaker diarization phase 1 — session-scoped speaker identification

- Add SpeakerTracker using resemblyzer for CPU-based voice embeddings
- Each utterance independently identified by voice, no turn-taking assumptions
- Ziggy introduces itself and asks name at session start (standard/performance)
- New speakers asked for name after their query is answered
- Speaker labels prepended to transcripts sent to LLM
- System prompt updated with speaker awareness
- Disabled in minimal profile
- 135+ tests passing"
```
