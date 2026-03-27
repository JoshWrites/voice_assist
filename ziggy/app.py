"""Ziggy Voice Assistant — main application orchestrator.

Wires together all modules and runs the main event loop.
"""

import os
import queue
import re
import signal
import subprocess
import sys
import threading
import time

from ziggy.config import WAKE_WORD, SHUTDOWN_PHRASE
from ziggy.profile import ProfileManager
from ziggy.stt import SpeechRecognizer
from ziggy.audio import AudioManager
from ziggy.tts import create_tts_engine
from ziggy.backend import detect_backend
from ziggy.backend.msty import MstyBackend
from ziggy.backend.ollama import OllamaBackend
from ziggy.conversation import ConversationManager
from ziggy.router import QueryRouter
from ziggy.tools import ToolRegistry
from ziggy.tools import time_date as time_date_tools
from ziggy.tools import conversions as conversion_tools
from ziggy.tools import web_search as web_search_tools


class VoiceAssistant:
    def __init__(self):
        self.is_listening = True
        self.is_processing = False
        self.conversational_mode = False
        self.last_interaction_time = 0
        self.setup_successful = False

        # Modules (initialized in setup)
        self.profile = ProfileManager()
        self.stt = SpeechRecognizer()
        self.audio = AudioManager(self.stt)
        self.tts = None
        self.backend = None
        self.model = None
        self.conversation = None
        self.router = None
        self.tools = ToolRegistry()

        print("Initializing Ziggy Voice Assistant...")
        self._setup()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup(self):
        try:
            if not self.audio.setup():
                return
            if not self.stt.setup():
                return

            self.profile.detect_and_select()

            if not self._setup_backend():
                return

            self.model = self.backend.get_default_model()
            print(f"  Using model: {self.model}")

            self.tts = create_tts_engine(self.profile.settings)
            self.conversation = ConversationManager(self.profile.settings)

            self._register_tools()

            self.router = QueryRouter(
                self.tools, self.backend, self.model,
                self.profile, self.conversation,
            )

            self.setup_successful = True
            print("All systems ready!")

        except Exception as e:
            print(f"Setup failed: {e}")
            self.setup_successful = False

    def _setup_backend(self):
        """Detect or prompt user to start an LLM backend."""
        self.backend = detect_backend()
        if self.backend:
            return True

        print("No AI backend detected")
        # Use espeak directly for the pre-init prompt
        subprocess.run(
            ["espeak", "-s", "150", "-v", "en",
             "No AI backend detected. Would you like me to start Misty or Ollama?"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        audio_data = self.audio.record_fixed(3)
        if not audio_data:
            return False

        text = self.stt.transcribe(audio_data, self.audio.get_sample_size()).lower()
        if not text:
            return False

        if "msty" in text or "misty" in text:
            self.backend = MstyBackend()
        elif "ollama" in text:
            self.backend = OllamaBackend()
        else:
            # Retry once
            subprocess.run(
                ["espeak", "-s", "150", "-v", "en",
                 "I didn't catch that. Please say Misty or Ollama."],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            audio_data = self.audio.record_fixed(3)
            if not audio_data:
                return False
            text = self.stt.transcribe(audio_data, self.audio.get_sample_size()).lower()
            if "msty" in text or "misty" in text:
                self.backend = MstyBackend()
            elif "ollama" in text:
                self.backend = OllamaBackend()
            else:
                print("Could not determine backend choice")
                return False

        if not self.backend.start():
            return False

        subprocess.run(
            ["espeak", "-s", "150", "-v", "en",
             f"Successfully connected to {self.backend.name}"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        return True

    def _register_tools(self):
        """Register all built-in tools."""
        time_date_tools.register(self.tools)
        conversion_tools.register(self.tools)
        web_search_tools.register(
            self.tools,
            speak_fn=self.speak,
            record_fn=self.audio.record_command,
            stt_fn=lambda audio: self.stt.transcribe(audio, self.audio.get_sample_size()),
        )

    # ------------------------------------------------------------------
    # Speech output (delegates to TTS engine with interruption support)
    # ------------------------------------------------------------------

    def speak(self, text, allow_interruption=True):
        """Speak text, optionally allowing wake-word interruption."""
        try:
            print(f"Speaking: {text}")
            if not allow_interruption or len(text) < 100:
                if not self.tts.speak(text):
                    print("  TTS failed")
                return False

            return self._speak_with_interruption(text)
        except Exception as e:
            print(f"Speech error: {e}")
            return False

    def _speak_with_interruption(self, text):
        """Speak long text sentence-by-sentence, interruptible by wake word."""
        sentences = _split_into_sentences(text)

        interruption_queue = queue.Queue()
        stop_speaking = threading.Event()

        listener = threading.Thread(
            target=self.audio.listen_for_interruption,
            args=(interruption_queue, stop_speaking),
        )
        listener.daemon = True
        listener.start()

        for sentence in sentences:
            if stop_speaking.is_set():
                print("Speech interrupted by wake word")
                break

            process, tmp_path = self.tts.speak_sentence_async(sentence.strip())
            if process is None:
                continue

            while process.poll() is None:
                if stop_speaking.is_set():
                    process.terminate()
                    if tmp_path:
                        try:
                            os.unlink(tmp_path)
                        except OSError:
                            pass
                    print("Speech interrupted mid-sentence")
                    break
                time.sleep(0.1)

            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

        stop_speaking.set()
        try:
            interruption_queue.get_nowait()
            return True
        except queue.Empty:
            return False

    # ------------------------------------------------------------------
    # Voice command handling
    # ------------------------------------------------------------------

    def _handle_voice_command(self):
        try:
            self.is_processing = True
            self.conversational_mode = False

            # Auto-clear stale history
            if time.time() - self.last_interaction_time > 300:
                self.conversation.clear()
            self.last_interaction_time = time.time()

            self.speak("Yes?", allow_interruption=False)

            audio_data = self.audio.record_command(profile_settings=self.profile.settings)
            if not audio_data:
                self.speak("I didn't hear anything", allow_interruption=False)
                return

            command_text = self.stt.transcribe(audio_data, self.audio.get_sample_size())
            if not command_text:
                self.speak("I couldn't understand that", allow_interruption=False)
                return

            print(f"Command: '{command_text}'")
            route_type, response = self.router.route(command_text)

            if route_type == "shutdown":
                self.speak(response, allow_interruption=False)
                self._shutdown()
                return

            if route_type == "ai":
                self.conversation.add_exchange(command_text, response)

            was_interrupted = self.speak(response, allow_interruption=True)
            if was_interrupted:
                self._handle_voice_command()
                return

            # Conversational follow-up loop
            if _contains_question(response):
                self._conversational_loop()

            time.sleep(0.5)
            if "browser" in response.lower() or "search" in response.lower():
                time.sleep(1.5)

        except Exception as e:
            print(f"Command handling error: {e}")
            self.speak("Sorry, I had trouble processing that", allow_interruption=False)
        finally:
            self.is_processing = False
            self.conversational_mode = False
            print("Ready to listen for wake word again...")

    def _conversational_loop(self):
        """Keep listening without wake word while the AI asks questions."""
        self.conversational_mode = True
        while True:
            audio = self.audio.record_command(
                conversational=True, profile_settings=self.profile.settings,
            )
            if not audio:
                break

            text = self.stt.transcribe(audio, self.audio.get_sample_size())
            if not text:
                break

            print(f"User answered: '{text}'")
            if SHUTDOWN_PHRASE in text.lower():
                self.speak("Okay, bye!", allow_interruption=False)
                self._shutdown()
                return

            system_prompt = (
                "You are having a friendly conversation. Respond naturally and keep "
                "the conversation flowing. Feel free to ask follow-up questions or "
                "share related thoughts. Be engaging and personable."
            )
            messages = self.conversation.build_messages(text, system_prompt=system_prompt)
            ai_response = self.backend.query(
                messages, self.model, temperature=0.8,
                max_tokens=self.conversation.profile_settings.get("response_tokens", 1000),
            )
            if not ai_response:
                break

            self.conversation.add_exchange(text, ai_response)
            was_interrupted = self.speak(ai_response, allow_interruption=True)

            if was_interrupted or not _contains_question(ai_response):
                break

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _startup_message(self):
        gb = self.profile.available_memory / 1024
        msg = (
            f"Welcome. Ziggy is ready to assist you, connected to "
            f"{self.backend.name} in {self.profile.settings['name']} mode "
            f"with {gb:.1f} gigabytes available."
        )
        print(msg)
        self.speak(msg, allow_interruption=False)

    def _shutdown(self):
        print("Shutting down Ziggy...")
        self.is_listening = False

        if hasattr(self.backend, "was_started_by_us") and self.backend.was_started_by_us:
            self.speak(
                f"Should I keep {self.backend.name} running after I shut down?",
                allow_interruption=False,
            )
            audio = self.audio.record_fixed(3)
            if audio:
                text = self.stt.transcribe(audio, self.audio.get_sample_size()).lower()
                if not any(w in text for w in ["yes", "yeah", "yep", "keep", "leave"]):
                    print(f"Stopping {self.backend.name}...")
                    self.backend.stop()

        self.audio.terminate()
        print("Goodbye!")
        sys.exit(0)

    def run(self):
        """Main event loop."""
        if not self.setup_successful:
            print("Setup failed — cannot start")
            return

        signal.signal(signal.SIGINT, lambda s, f: self._shutdown())
        signal.signal(signal.SIGTERM, lambda s, f: self._shutdown())

        self._startup_message()

        print(f"\nVoice Assistant Active")
        print(f"Wake word: '{WAKE_WORD}'")
        print(f"Shutdown: '{SHUTDOWN_PHRASE}'")
        print("Press Ctrl+C to stop")
        print("-" * 50)

        try:
            while self.is_listening:
                result = self.audio.listen_for_wake_word(lambda: self.is_listening)
                if result == "shutdown":
                    self.speak("Okay, bye!", allow_interruption=False)
                    self._shutdown()
                elif result is True:
                    self._handle_voice_command()
                elif not self.is_listening:
                    break
                if self.is_listening:
                    time.sleep(0.1)
        except KeyboardInterrupt:
            self._shutdown()
        except Exception as e:
            print(f"Main loop error: {e}")
            self._shutdown()


# ------------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------------

def _split_into_sentences(text):
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return [text]

    cleaned = []
    current = ""
    for sentence in sentences:
        current += sentence + ". "
        if len(current.split()) >= 5 or sentence == sentences[-1]:
            cleaned.append(current.strip())
            current = ""
    return cleaned or [text]


def _contains_question(text):
    if "?" in text:
        return True
    starters = [
        "what", "where", "when", "who", "why", "how",
        "would", "could", "should", "can", "will", "do",
        "does", "did", "is", "are", "was", "were",
        "have", "has", "had", "may", "might",
    ]
    for sentence in text.split("."):
        s = sentence.strip().lower()
        if any(s.startswith(w + " ") for w in starters):
            return True
    return False
