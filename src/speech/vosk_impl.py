import json
import numpy as np
from typing import Optional, Union
import vosk
from .interfaces import SpeechRecognitionInterface, WakeWordDetectorInterface


class VoskSpeechRecognition(SpeechRecognitionInterface):
    """Vosk implementation of speech recognition"""
    
    def __init__(self, model_path: str, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.model = None
        self.recognizer = None
        try:
            self.model = vosk.Model(model_path)
            self.recognizer = vosk.KaldiRecognizer(self.model, sample_rate)
        except Exception as e:
            print(f"Error loading Vosk model: {e}")
    
    def recognize(self, audio_data: Union[bytes, np.ndarray],
                 sample_rate: int = 16000) -> Optional[str]:
        """Convert speech to text"""
        if not self.is_ready():
            return None

        try:
            # Reset recognizer to clear any accumulated state from previous calls
            self.recognizer = vosk.KaldiRecognizer(self.model, self.sample_rate)

            # Convert numpy array to bytes if needed
            if isinstance(audio_data, np.ndarray):
                audio_data = audio_data.tobytes()

            # Process audio and get final result
            self.recognizer.AcceptWaveform(audio_data)
            result = json.loads(self.recognizer.FinalResult())
            return result.get('text', '')
        except Exception as e:
            print(f"Recognition error: {e}")
            return None
    
    def is_ready(self) -> bool:
        """Check if the recognizer is ready"""
        return self.model is not None and self.recognizer is not None


class VoskWakeWordDetector(WakeWordDetectorInterface):
    """Vosk-based wake word detection"""
    
    def __init__(self, model_path: str, wake_word: str = "ziggy", 
                 sample_rate: int = 16000, confidence_threshold: float = 0.7):
        self.wake_word = wake_word.lower()
        self.sample_rate = sample_rate
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.recognizer = None
        
        try:
            self.model = vosk.Model(model_path)
            self.recognizer = vosk.KaldiRecognizer(self.model, sample_rate)
        except Exception as e:
            print(f"Error loading Vosk model for wake word: {e}")
    
    def detect(self, audio_data: Union[bytes, np.ndarray],
              sample_rate: int = 16000) -> bool:
        """Detect if wake word is present in audio chunk (streaming, no reset)"""
        if not self.model or not self.recognizer:
            return False

        try:
            # Convert numpy array to bytes if needed
            if isinstance(audio_data, np.ndarray):
                audio_data = audio_data.tobytes()

            # Process chunk and check both partial and final results
            if self.recognizer.AcceptWaveform(audio_data):
                result = json.loads(self.recognizer.Result())
                text = result.get('text', '').lower()
                # Reset after a complete utterance so state doesn't accumulate
                self.recognizer = vosk.KaldiRecognizer(self.model, self.sample_rate)
            else:
                partial = json.loads(self.recognizer.PartialResult())
                text = partial.get('partial', '').lower()

            return self.wake_word in text

        except Exception as e:
            print(f"Wake word detection error: {e}")
            return False
    
    def get_wake_word(self) -> str:
        """Get the configured wake word"""
        return self.wake_word
    
    def set_wake_word(self, wake_word: str) -> None:
        """Set a new wake word"""
        self.wake_word = wake_word.lower()