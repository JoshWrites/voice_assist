# Speaker Diarization — Phase 1 Design

## Goal

Distinguish which person is speaking during a Ziggy session so the LLM receives speaker-attributed transcripts. Session-scoped only — no persistence across restarts.

## Scope

- In-session speaker tracking using voice embeddings (resemblyzer, CPU)
- Speaker labels assigned by voice matching, not by turn-taking assumptions
- Each utterance independently compared against known embeddings
- Ziggy introduces itself and asks the user's name at session start (standard/performance)
- Names stored in a session-scoped dict, used in transcripts sent to LLM
- Enabled in standard and performance profiles only. Minimal profile skips entirely.

## Out of Scope

- Persist speaker identities across sessions (phase 3)
- Distinguish adults from children (phase 2)
- Separate overlapping speech into individual streams
- Run on GPU (future — can move to pyannote on RX 5700 XT)

## Architecture

### Pipeline Integration

```
Mic → AudioManager.record_command() → raw audio bytes
                                          ↓
                                    SpeakerTracker.identify(audio) → speaker label
                                          ↓
                                    Vosk STT → text
                                          ↓
                                    Router receives: "Josh: what's the weather"
```

SpeakerTracker processes the same audio bytes that go to Vosk. It adds a label and passes through. Each utterance is identified independently — no assumptions about who spoke last or turn order. The same speaker can be identified ten times in a row.

### New Module: `ziggy/speaker.py`

**Class: `SpeakerTracker`**

State:
- `_encoder`: resemblyzer `VoiceEncoder` instance (loaded once at setup)
- `_embeddings`: list of `(speaker_id, embedding_vector)` tuples seen this session
- `_names`: dict mapping speaker_id to name, e.g. `{"Speaker 1": "Josh"}`
- `_next_id`: int counter for assigning new speaker labels
- `_threshold`: cosine similarity cutoff for matching (default 0.75)

Methods:
- `setup() -> bool`: Load the voice encoder model. Returns False if resemblyzer unavailable.
- `identify(audio_bytes, sample_rate) -> str`: Generate embedding, compare against all known session embeddings, return the display label (name if known, else "Speaker N"). If no match, register new speaker.
- `set_name(speaker_id, name)`: Map a speaker ID to a name.
- `get_display_label(speaker_id) -> str`: Return name if mapped, else the raw speaker ID.
- `reset()`: Clear all embeddings and names.
- `is_active() -> bool`: Whether speaker tracking is enabled (False in minimal).

### How Identification Works

Resemblyzer's `VoiceEncoder` converts audio into a 256-dimensional embedding vector. Two clips from the same person have high cosine similarity (>0.75). Different people are far apart.

For each recorded utterance:
1. Preprocess: convert raw bytes to float32 array at 16kHz
2. `VoiceEncoder.embed_utterance()` → 256-float vector
3. Compute cosine similarity against every known embedding in the session
4. Best match above threshold → that speaker
5. No match above threshold → new speaker, assign "Speaker N", store embedding

Each utterance is judged solely on its voice characteristics. No alternation logic, no pattern assumptions.

### Session Start Behavior

**Minimal profile:** No speaker tracking. No introduction. Ziggy starts listening immediately.

**Standard/Performance profile:** After startup, before entering the wake word loop, Ziggy introduces itself:

> "Hi, I'm Ziggy. What's your name?"

The user's response is:
1. Recorded and embedded → becomes Speaker 1's voice print
2. Transcribed → name extracted and mapped to Speaker 1

If the user doesn't give a name or Ziggy can't extract one, Speaker 1 remains the label. No retry loop — just move on.

When a new voice is detected later in the session, Ziggy asks their name the same way after responding to their query (answer first, ask second — don't block the interaction).

### Transcript Format

The router receives text with the speaker label prepended:

```
Josh: what's the weather in tel aviv
Speaker 2: what time is it
Josh: thanks
Josh: also what time is it in new york
```

If speaker tracking is inactive (minimal profile) or setup failed, no label is prepended — plain text as before.

### System Prompt Addition

Append to existing `SYSTEM_PROMPT`:

```
When transcripts include speaker labels (like "Josh:" or "Speaker 2:"),
you're hearing from different people. Use their names when known.
If someone introduces themselves, remember their name for this conversation.
```

### Profile Integration

Config addition in `RESOURCE_PROFILES`:

```python
"minimal":     { ..., "speaker_tracking": False },
"standard":    { ..., "speaker_tracking": True },
"performance": { ..., "speaker_tracking": True },
```

`VoiceAssistant._setup()` checks this flag and either initializes `SpeakerTracker` or sets it to `None`.

## CPU Cost

Resemblyzer runs on CPU only. Embedding a 5-second clip takes ~50-100ms on the Ryzen 9 5950X. Comparison against N known speakers is negligible (cosine similarity on 256-float vectors). No GPU VRAM used.

## Dependencies

- `resemblyzer` — pip install, ~10MB, depends on numpy and librosa
- No GPU required, no new system packages

## Error Handling

- If resemblyzer fails to install or load: speaker tracking disabled, Ziggy works normally without labels
- If embedding generation fails on a clip (too short, silence, noise): no label prepended for that utterance
- If similarity scores are ambiguous (multiple speakers near threshold): attribute to best match, not "unknown"

## Testing Strategy

- Unit test `SpeakerTracker.identify()` with synthetic audio (different frequency sine waves to simulate distinct "voices")
- Test that same-voice clips produce same speaker ID
- Test that different-voice clips produce different speaker IDs
- Test name mapping: `set_name()` → `get_display_label()` returns name
- Test `reset()` clears all state
- Test minimal profile: tracker is None, transcripts have no labels
- Test transcript formatting with and without speaker labels
- Mock `VoiceEncoder` in all tests — no actual model loading in CI
