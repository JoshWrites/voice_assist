"""
Interruption handling for speech synthesis
"""
import threading
import queue
import time
import re
from typing import List, Callable, Optional
from ..audio import AudioFactory


class InterruptibleTTS:
    """Enhanced TTS with interruption capabilities"""
    
    def __init__(self, base_tts, wake_word_detector=None):
        self.base_tts = base_tts
        self.wake_word_detector = wake_word_detector
        self.audio_input = None
        
    def speak_with_interruption(self, text: str, 
                               interruption_detector: Optional[Callable[[], bool]] = None) -> bool:
        """Speak text while listening for interruptions"""
        if len(text) < 100:
            # Short responses - speak normally without interruption
            self.base_tts.speak(text)
            return False
        
        # Break text into sentences for chunked speaking
        sentences = self._split_into_sentences(text)
        
        # Setup interruption detection
        interruption_queue = queue.Queue()
        stop_speaking = threading.Event()
        
        # Use provided detector or default wake word detector
        detector = interruption_detector or self._default_interruption_detector
        
        # Start listening for interruption in background
        listener_thread = threading.Thread(
            target=self._listen_for_interruption,
            args=(interruption_queue, stop_speaking, detector)
        )
        listener_thread.daemon = True
        listener_thread.start()
        
        # Speak each sentence, checking for interruption
        interrupted = False
        for i, sentence in enumerate(sentences):
            if stop_speaking.is_set():
                interrupted = True
                break
            
            print(f"🗣️ Speaking part {i+1}/{len(sentences)}: {sentence[:50]}...")
            
            # Speak sentence
            self.base_tts.speak(sentence.strip())
            
            # Brief pause between sentences
            time.sleep(0.3)
            
            # Check for interruption
            if not interruption_queue.empty():
                interruption_queue.get()  # Clear the queue
                interrupted = True
                break
        
        # Signal listener to stop
        stop_speaking.set()
        
        if interrupted:
            print("🔄 Speech interrupted!")
            self.base_tts.stop_speaking()
        
        return interrupted
    
    def _listen_for_interruption(self, interruption_queue, stop_speaking, detector):
        """Listen for interruption signal in background thread"""
        try:
            if not self.audio_input:
                self.audio_input = AudioFactory.create_input("pyaudio", channels=1, sample_rate=16000)

            self.audio_input.start_stream()

            # Wait for TTS to get started before listening — avoids self-triggering
            time.sleep(1.5)

            # Accumulate ~1 second of audio per detection pass for reliable recognition
            chunks_per_pass = 16  # 16 * 1024 frames @ 16kHz ≈ 1 second
            buffer = b""

            while not stop_speaking.is_set():
                try:
                    data = self.audio_input.read_chunk(1024)
                    buffer += data

                    if len(buffer) >= 1024 * chunks_per_pass:
                        if detector and detector(buffer):
                            interruption_queue.put(True)
                            break
                        buffer = b""

                except Exception as e:
                    print(f"Interruption detection error: {e}")
                    break

        except Exception as e:
            print(f"Interruption listener setup error: {e}")
        finally:
            if self.audio_input:
                self.audio_input.stop_stream()
    
    def _default_interruption_detector(self, audio_data: bytes) -> bool:
        """Default interruption detector using wake word"""
        if not self.wake_word_detector:
            return False
        
        try:
            return self.wake_word_detector.detect(audio_data)
        except Exception:
            return False
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences for chunked speaking"""
        # Split on periods, exclamation marks, question marks
        sentences = re.split(r'[.!?]+', text)
        
        # Clean up and filter empty sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # If no sentences found, return original text
        if not sentences:
            return [text]
        
        # Rejoin sentences that are too short (likely abbreviations)
        cleaned_sentences = []
        i = 0
        while i < len(sentences):
            sentence = sentences[i]
            
            # If sentence is very short and not the last one, try to combine
            while (len(sentence.split()) < 4 and 
                   i + 1 < len(sentences) and 
                   len(cleaned_sentences) > 0):
                i += 1
                sentence += ". " + sentences[i]
            
            cleaned_sentences.append(sentence)
            i += 1
        
        return cleaned_sentences if cleaned_sentences else [text]