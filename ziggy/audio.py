"""Audio I/O — recording, wake word detection, voice activity detection."""

import json
import time

import pyaudio

from ziggy.config import (
    AUDIO_SAMPLE_RATE,
    AUDIO_CHUNK_SIZE,
    AUDIO_CHANNELS,
    AUDIO_INPUT_DEVICE,
    SILENCE_THRESHOLD,
    WAKE_WORD,
    SHUTDOWN_PHRASE,
)


class AudioManager:
    def __init__(self, stt):
        self.stt = stt
        self.sample_rate = AUDIO_SAMPLE_RATE
        self.chunk_size = AUDIO_CHUNK_SIZE
        self.channels = AUDIO_CHANNELS
        self.format = pyaudio.paInt16
        self.audio = None
        self.input_device = AUDIO_INPUT_DEVICE

    def setup(self) -> bool:
        try:
            self.audio = pyaudio.PyAudio()
            if self.input_device is None:
                self.input_device = self._find_best_input()
            if self.input_device is not None:
                name = self.audio.get_device_info_by_index(self.input_device)["name"]
                print(f"  Audio input: {name} (device {self.input_device})")
            else:
                print("  Audio input: system default")
            print("  Audio system initialized")
            return True
        except Exception as e:
            print(f"  Audio init failed: {e}")
            return False

    def _find_best_input(self):
        """Pick the first USB mic, preferring webcam/headset over onboard."""
        usb_devices = []
        for i in range(self.audio.get_device_count()):
            info = self.audio.get_device_info_by_index(i)
            if info["maxInputChannels"] > 0 and "hw:" in info["name"]:
                name = info["name"].lower()
                if "webcam" in name or "usb" in name or "jabra" in name:
                    usb_devices.append((i, info["name"]))
        if usb_devices:
            return usb_devices[0][0]
        return None

    def _open_input_stream(self):
        """Open an audio input stream on the configured device."""
        kwargs = dict(
            format=self.format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
        )
        if self.input_device is not None:
            kwargs["input_device_index"] = self.input_device
        return self.audio.open(**kwargs)

    def get_sample_size(self):
        return self.audio.get_sample_size(self.format)

    def record_command(self, duration=5, conversational=False, profile_settings=None):
        """Record audio until pause detected via VAD."""
        try:
            if conversational:
                print("  Listening for your response...")
            else:
                print("  Recording until pause detected...")

            stream = self._open_input_stream()

            frames = []
            recognizer = self.stt.create_recognizer()

            if conversational:
                max_dur = (profile_settings or {}).get("recording_conversational", 300)
                time.sleep(0.3)
            else:
                max_dur = (profile_settings or {}).get("recording_command", 60)

            last_speech = time.time()
            start = time.time()
            has_speech = False
            last_feedback = start

            while True:
                now = time.time()
                elapsed = now - start

                if elapsed > 30 and (now - last_feedback) > 30:
                    m, s = divmod(int(elapsed), 60)
                    print(f"  Recording: {m}:{s:02d} elapsed...")
                    last_feedback = now

                if elapsed > max_dur:
                    print(f"  Max recording duration reached ({int(max_dur / 60)}m)")
                    break

                data = stream.read(self.chunk_size, exception_on_overflow=False)
                frames.append(data)

                if recognizer.AcceptWaveform(data):
                    result = json.loads(recognizer.Result())
                    if result.get("text"):
                        last_speech = now
                        has_speech = True
                else:
                    partial = json.loads(recognizer.PartialResult())
                    if partial.get("partial"):
                        last_speech = now
                        has_speech = True

                if has_speech and (now - last_speech) > SILENCE_THRESHOLD:
                    dur = now - start
                    if dur < 60:
                        print(f"  Pause detected after {dur:.1f}s")
                    else:
                        m, s = divmod(int(dur), 60)
                        print(f"  Pause detected after {m}:{s:02d}")
                    break

            stream.close()
            return b"".join(frames) if frames else None

        except Exception as e:
            print(f"Recording error: {e}")
            return None

    def record_fixed(self, seconds=3):
        """Record for a fixed duration (used during setup before full VAD)."""
        try:
            if not self.audio:
                self.audio = pyaudio.PyAudio()
            stream = self._open_input_stream()
            frames = []
            for _ in range(0, int(self.sample_rate / self.chunk_size * seconds)):
                data = stream.read(self.chunk_size, exception_on_overflow=False)
                frames.append(data)
            stream.close()
            return b"".join(frames)
        except Exception as e:
            print(f"Recording error: {e}")
            return None

    def listen_for_wake_word(self, is_listening_fn):
        """Block until wake word or shutdown phrase detected.

        Returns: True (wake word), "shutdown", or False (error/stopped).
        """
        max_retries = 3
        retry_count = 0

        while is_listening_fn() and retry_count < max_retries:
            try:
                recognizer = self.stt.create_recognizer()
                stream = None
                for attempt in range(3):
                    try:
                        stream = self._open_input_stream()
                        break
                    except Exception as e:
                        print(f"Audio stream attempt {attempt + 1} failed: {e}")
                        time.sleep(1)

                if not stream:
                    print("  Could not open audio stream")
                    return False

                print(f"  Listening for wake word '{WAKE_WORD}'...")
                retry_count = 0

                while is_listening_fn():
                    try:
                        data = stream.read(self.chunk_size, exception_on_overflow=False)
                        if recognizer.AcceptWaveform(data):
                            result = json.loads(recognizer.Result())
                            text = result.get("text", "").lower().strip()
                        else:
                            partial = json.loads(recognizer.PartialResult())
                            text = partial.get("partial", "").lower().strip()

                        if text:
                            if WAKE_WORD in text:
                                print(f"  Wake word detected: '{text}'")
                                stream.close()
                                return True
                            if SHUTDOWN_PHRASE in text:
                                print(f"  Shutdown detected: '{text}'")
                                stream.close()
                                return "shutdown"
                    except Exception as e:
                        if is_listening_fn():
                            print(f"  Audio read error: {e}")
                            break

                if stream:
                    stream.close()
                return False

            except Exception as e:
                retry_count += 1
                print(f"  Wake word error (attempt {retry_count}): {e}")
                if retry_count < max_retries:
                    time.sleep(2)
                else:
                    print("  Max retries reached")
                    return False
        return False

    def listen_for_interruption(self, interruption_queue, stop_event):
        """Background thread: listen for wake word during speech."""
        try:
            recognizer = self.stt.create_recognizer()
            stream = self._open_input_stream()

            # Wait for TTS to start before listening — avoids self-triggering
            time.sleep(1.5)

            # Buffer ~1 second of audio per detection pass for reliable recognition
            chunks_per_pass = max(1, int(self.sample_rate / self.chunk_size))
            buffer = b""

            while not stop_event.is_set():
                try:
                    data = stream.read(self.chunk_size, exception_on_overflow=False)
                    buffer += data

                    if len(buffer) >= self.chunk_size * chunks_per_pass:
                        if recognizer.AcceptWaveform(buffer):
                            result = json.loads(recognizer.Result())
                            if result.get("text"):
                                transcript = result["text"].lower().strip()
                                if WAKE_WORD in transcript:
                                    interruption_queue.put(True)
                                    stop_event.set()
                                    break
                        buffer = b""

                except Exception as e:
                    if not stop_event.is_set():
                        print(f"Interruption listening error: {e}")
                        break
            stream.close()
        except Exception as e:
            print(f"Interruption listener error: {e}")

    def terminate(self):
        if self.audio:
            self.audio.terminate()
