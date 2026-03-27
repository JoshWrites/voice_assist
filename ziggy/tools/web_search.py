"""Web search tools — privacy-focused via DuckDuckGo."""

import subprocess
import time
import urllib.parse

import requests


class WebSearchTool:
    """Handles web searches with permission gating.

    Needs references to the assistant's speak/record helpers, so it's
    initialised with callback functions rather than being purely static.
    """

    def __init__(self, speak_fn, record_fn, stt_fn):
        self._speak = speak_fn
        self._record = record_fn
        self._stt = stt_fn

    def search(self, query):
        if not self._request_permission():
            return "Okay, staying local. Is there anything else I can help you with?"

        self._speak("Would you like me to read you the answer, or open a browser window?",
                     allow_interruption=False)

        audio = self._record(duration=4, conversational=True)
        if audio:
            text = self._stt(audio).lower().strip()
            read_words = ["read", "tell", "say", "speak", "answer"]
            browse_words = ["browser", "open", "window", "firefox", "chrome"]

            if any(w in text for w in read_words):
                return self._fetch_and_read(query)
            elif any(w in text for w in browse_words):
                return self._open_browser(query)

        return self._fetch_and_read(query)

    def _request_permission(self):
        self._speak(
            "I cannot answer that from my local resources. Do you want me to check online?",
            allow_interruption=False,
        )
        audio = self._record(duration=3, conversational=True)
        if not audio:
            return False
        text = self._stt(audio).lower().strip()
        yes_words = ["yes", "yeah", "yep", "okay", "ok", "sure", "go ahead", "please"]
        no_words = ["no", "nope", "don't", "stop", "cancel", "nevermind"]
        if any(w in text for w in yes_words):
            return True
        if any(w in text for w in no_words):
            return False
        return False

    def _fetch_and_read(self, query):
        try:
            self._speak("Let me search for that", allow_interruption=False)
            encoded = urllib.parse.quote_plus(query)
            url = f"https://api.duckduckgo.com/?q={encoded}&format=json&no_html=1&skip_disambig=1"
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                answer = (data.get("AbstractText", "") or data.get("Answer", "")).strip()
                if answer:
                    if len(answer) > 300:
                        answer = answer[:300] + "... would you like me to open a browser for more details?"
                    return answer
                self._speak("I couldn't find a direct answer. Let me open a browser.", allow_interruption=False)
                time.sleep(1)
                return self._open_browser(query)
            return self._open_browser(query)
        except Exception as e:
            print(f"Search error: {e}")
            self._speak("I had trouble searching. Let me open a browser instead.", allow_interruption=False)
            time.sleep(1)
            return self._open_browser(query)

    def _open_browser(self, query):
        try:
            encoded = urllib.parse.quote_plus(query)
            subprocess.Popen(
                ["firefox", f"https://duckduckgo.com/?q={encoded}"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            time.sleep(2)
            return f"Opened browser search for {query}"
        except Exception:
            return "Could not open web browser"


def register(registry, speak_fn, record_fn, stt_fn):
    tool = WebSearchTool(speak_fn, record_fn, stt_fn)
    registry.register(
        name="web_search",
        description="Search the web via DuckDuckGo",
        keywords=["search", "look up"],
        handler=lambda text: tool.search(
            text.lower().replace("search", "").replace("look up", "").strip()
        ),
    )
