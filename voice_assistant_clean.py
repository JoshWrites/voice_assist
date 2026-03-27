#!/usr/bin/env python3
"""
Clean Voice Assistant - "Ziggy" 
Complete refactored implementation using modular architecture

Features:
- Custom "ziggy" wake word detection using Vosk
- Smart command routing (local functions vs AI)
- GPU-accelerated AI responses via Ollama/Msty
- espeak/Piper text-to-speech with interruption capability
- Conversation context management
- Resource profile management
- No web search functionality

Usage: python3 voice_assistant_clean.py
"""
import os
import time
import signal
import sys
from pathlib import Path

from src.audio import AudioFactory
from src.speech import SpeechFactory
from src.ai import AIBackendManager
from src.commands import (
    CommandRouter, TimeHandler, DateHandler, 
    MathHandler, ConversionHandler
)
from src.conversation import ConversationManager
from src.resources import ResourceManager


class CleanVoiceAssistant:
    """Clean voice assistant using modular architecture"""
    
    def __init__(self):
        # Core state
        self.wake_word = "ziggy"
        self.shutdown_phrase = "take a break"
        self.is_listening = True
        self.is_processing = False
        
        # Initialize managers
        self.resource_manager = ResourceManager()
        self.conversation_manager = ConversationManager(
            history_limit=self.resource_manager.get_current_profile().history_limit
        )
        
        # Initialize components
        self._initialize_audio()
        self._initialize_speech()
        self._initialize_ai()
        self._initialize_commands()
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _initialize_audio(self):
        """Initialize audio components"""
        self.audio_input = AudioFactory.create_input("pyaudio")
        self.audio_recorder = AudioFactory.create_recorder("pyaudio")
    
    def _initialize_speech(self):
        """Initialize speech components"""
        model_path = "vosk-model-small-en-us-0.15"
        if not os.path.exists(model_path):
            print(f"❌ Vosk model not found at {model_path}")
            print("Please download the Vosk model or update the path")
            sys.exit(1)
        
        self.speech_recognizer = SpeechFactory.create_recognizer(
            "vosk", 
            model_path=model_path
        )
        self.wake_word_detector = SpeechFactory.create_wake_word_detector(
            "vosk",
            model_path=model_path,
            wake_word=self.wake_word
        )
        
        # Initialize TTS
        self.tts = SpeechFactory.create_tts("espeak", voice="en", speed=150)
    
    def _initialize_ai(self):
        """Initialize AI backend"""
        self.ai_manager = AIBackendManager()
    
    def _initialize_commands(self):
        """Initialize command routing"""
        self.command_router = CommandRouter()
        
        # Register local command handlers
        self.command_router.register_handler(TimeHandler())
        self.command_router.register_handler(DateHandler())
        self.command_router.register_handler(MathHandler())
        self.command_router.register_handler(ConversionHandler())
    
    def setup(self):
        """Setup the voice assistant"""
        print("🚀 Initializing Clean Voice Assistant...")
        
        # Setup AI backend
        backend_type = self.ai_manager.detect_backend()
        if not backend_type:
            print("❌ No AI backend detected. Please start Ollama or Msty.")
            return False
        
        backend = self.ai_manager.get_backend()
        print(f"✅ Connected to {backend.get_backend_name()} backend")
        
        # Display resource profile info
        profile = self.resource_manager.get_current_profile()
        memory_status = self.resource_manager.get_memory_status()
        print(f"🎯 Using {profile.name} profile ({memory_status.available/1024:.1f}GB available)")
        
        return True
    
    def listen_for_wake_word(self):
        """Listen for wake word"""
        print("👂 Listening for 'ziggy'...")
        
        try:
            while self.is_listening:
                # Record short audio clips
                audio_data, sample_rate = self.audio_recorder.record(duration=2.0)
                
                # Check for wake word
                if self.wake_word_detector.detect(audio_data, sample_rate):
                    print("👋 Wake word detected!")
                    return True
                    
        except KeyboardInterrupt:
            return False
        
        return False
    
    def record_command(self, conversational=False):
        """Record voice command"""
        profile = self.resource_manager.get_current_profile()
        max_duration = (profile.recording_conversational if conversational 
                       else profile.recording_command)
        
        print(f"🎤 Listening for command (max {max_duration}s)...")
        
        try:
            audio_data, sample_rate = self.audio_recorder.record_until_silence(
                max_duration=max_duration,
                silence_threshold=0.05,
                silence_duration=1.5
            )
            return audio_data, sample_rate
        except Exception as e:
            print(f"Recording error: {e}")
            return None, None
    
    def process_command(self, command_text):
        """Process voice command"""
        print(f"📝 Command: {command_text}")
        
        # Check for shutdown
        if self.shutdown_phrase in command_text.lower():
            return "shutdown", "Okay, bye!"
        
        # Check for conversation management commands
        if any(phrase in command_text.lower() for phrase in 
               ['new conversation', 'start over', 'clear history', 'fresh start']):
            self.conversation_manager.clear_history()
            return "local", "Starting fresh. What would you like to talk about?"
        
        # Check for profile management commands
        profile_response = self._handle_profile_commands(command_text)
        if profile_response:
            return "local", profile_response
        
        # Route command through command router
        route_type, result = self.command_router.route_command(command_text)
        
        if route_type == "local":
            return "local", result.response
        else:
            # Process with AI
            return self._handle_ai_query(command_text)
    
    def _handle_profile_commands(self, text):
        """Handle resource profile management commands"""
        text_lower = text.lower()
        
        # Profile switching commands
        profile_triggers = [
            ('switch to', 'mode'), ('switched to', 'mode'), ('change to', 'mode'),
            ('use', 'mode'), ('set', 'mode'), ('enable', 'mode'),
            ('switch to', 'profile'), ('switched to', 'profile'), ('change to', 'profile'),
            ('use', 'profile'), ('set', 'profile'), ('enable', 'profile')
        ]
        
        for trigger, keyword in profile_triggers:
            if trigger in text_lower and keyword in text_lower:
                # Extract profile name after the trigger word
                profile_words = text_lower.split(trigger)[-1].strip()
                profile_words = profile_words.replace('mode', '').replace('profile', '').strip()
                
                if self.resource_manager.switch_profile(profile_words):
                    # Update conversation manager with new limits
                    new_profile = self.resource_manager.get_current_profile()
                    self.conversation_manager.update_history_limit(new_profile.history_limit)
                    return f"Switched to {new_profile.name} profile. {new_profile.description}"
                else:
                    return "Sorry, I don't recognize that profile. Available profiles are minimal, standard, and performance."
        
        # Profile status commands
        if any(phrase in text_lower for phrase in 
               ['what profile', 'which profile', 'current profile']):
            return self.resource_manager.get_profile_info()
        
        if any(phrase in text_lower for phrase in 
               ['what profiles', 'available profiles', 'list profiles']):
            return self.resource_manager.list_available_profiles()
        
        return None
    
    def _handle_ai_query(self, text):
        """Handle AI query with conversation context"""
        backend = self.ai_manager.get_backend()
        if not backend:
            return "local", "AI backend is not available"
        
        try:
            if self.conversation_manager.is_conversational_mode():
                # Handle as conversational response
                return self._handle_conversational_response(text)
            else:
                # Handle as new query with context
                profile = self.resource_manager.get_current_profile()
                context = self.conversation_manager.get_context(profile.context_tokens)
                
                # Build messages
                messages = [
                    {"role": "system", 
                     "content": "You are a local AI assistant. Answer questions using only your training data. Be helpful, friendly, and concise."}
                ]
                
                # Add conversation context
                messages.extend(context.messages)
                
                # Add current query
                messages.append({"role": "user", "content": text})
                
                # Query AI
                response = backend.query(
                    text, 
                    context=messages[:-1] if len(messages) > 1 else None
                )
                
                if response.error:
                    return "local", f"Sorry, I had trouble processing that: {response.error}"
                
                # Add to conversation history
                self.conversation_manager.add_message("user", text)
                self.conversation_manager.add_message("assistant", response.content)
                
                return "ai", response.content
                
        except Exception as e:
            print(f"AI query error: {e}")
            return "local", "Sorry, I had trouble processing that request"
    
    def _handle_conversational_response(self, text):
        """Handle responses during conversational mode"""
        backend = self.ai_manager.get_backend()
        if not backend:
            return "local", "AI backend is not available"
        
        try:
            profile = self.resource_manager.get_current_profile()
            context = self.conversation_manager.get_context(profile.context_tokens)
            
            # Build conversational messages
            messages = [
                {"role": "system",
                 "content": "You are having a friendly conversation. Respond naturally and keep the conversation flowing. Feel free to ask follow-up questions or share related thoughts. Be engaging and personable."}
            ]
            
            # Add conversation context
            messages.extend(context.messages)
            
            # Add current response
            messages.append({"role": "user", "content": text})
            
            # Query AI
            response = backend.query(
                text,
                context=messages[:-1] if len(messages) > 1 else None
            )
            
            if response.error:
                return "local", f"Sorry, I had trouble with that: {response.error}"
            
            # Add to conversation history
            self.conversation_manager.add_message("user", text)
            self.conversation_manager.add_message("assistant", response.content)
            
            return "ai", response.content
            
        except Exception as e:
            print(f"Conversational AI error: {e}")
            return "local", "Sorry, there was an error continuing our conversation"
    
    def handle_voice_command(self):
        """Handle complete voice interaction after wake word"""
        try:
            self.is_processing = True
            self.conversation_manager.exit_conversational_mode()
            
            # Check if we should clear conversation history due to timeout
            if self.conversation_manager.should_clear_history(timeout_seconds=300):
                self.conversation_manager.clear_history()
                print("🧹 Cleared conversation history (timeout)")
            
            self.tts.speak("Yes?")
            
            # Record the user's command
            audio_data, sample_rate = self.record_command()
            if audio_data is None or len(audio_data) == 0:
                self.tts.speak("I didn't hear anything")
                return
            
            # Convert to text
            command_text = self.speech_recognizer.recognize(audio_data, sample_rate)
            if not command_text:
                self.tts.speak("I couldn't understand that")
                return
            
            # Process command
            route_type, response = self.process_command(command_text)
            print(f"🤖 Response [{route_type}]: {response[:120]}...")

            if route_type == "shutdown":
                self.tts.speak(response)
                self.shutdown()
                return
            
            # Speak the response with interruption capability for long responses
            interruption_detector = lambda data: self.wake_word_detector.detect(data)
            was_interrupted = self.tts.speak_with_interruption(
                response, 
                interruption_detector
            )
            
            if was_interrupted:
                print("🔄 Response was interrupted - processing new command")
                self.handle_voice_command()
                return
            
            # Check if the response contains a question
            if self.command_router.contains_question(response):
                print("❓ Response contains a question - entering conversational mode")
                self.conversation_manager.enter_conversational_mode()
                
                # Continue conversation loop
                self._continue_conversation()
        
        except Exception as e:
            import traceback
            print(f"Command handling error: {e}")
            traceback.print_exc()
            self.tts.speak("Sorry, I had trouble processing that")
        finally:
            self.is_processing = False
            self.conversation_manager.exit_conversational_mode()
            print("👂 Ready to listen for wake word again...")
    
    def _continue_conversation(self):
        """Continue conversational interaction"""
        while self.conversation_manager.is_conversational_mode():
            # Listen for the answer
            audio_data, sample_rate = self.record_command(conversational=True)
            if audio_data is None or len(audio_data) == 0:
                break
            
            answer_text = self.speech_recognizer.recognize(audio_data, sample_rate)
            if not answer_text:
                break
            
            print(f"📝 User answered: '{answer_text}'")
            
            # Check for shutdown
            if self.shutdown_phrase in answer_text.lower():
                self.tts.speak("Okay, bye!")
                self.shutdown()
                return
            
            # Process conversational response
            route_type, follow_up_response = self._handle_conversational_response(answer_text)
            
            # Speak follow-up response
            interruption_detector = lambda data: self.wake_word_detector.detect(data)
            was_interrupted = self.tts.speak_with_interruption(
                follow_up_response,
                interruption_detector
            )
            
            if was_interrupted:
                self.handle_voice_command()
                return
            
            # Continue if AI asks another question
            if not self.command_router.contains_question(follow_up_response):
                self.conversation_manager.exit_conversational_mode()
                break
    
    def startup_message(self):
        """Play welcome message on startup"""
        profile = self.resource_manager.get_current_profile()
        memory_status = self.resource_manager.get_memory_status()
        backend = self.ai_manager.get_backend()
        
        welcome_msg = (
            f"Welcome. Ziggy is ready to assist you, connected to "
            f"{backend.get_backend_name()} in {profile.name} mode with "
            f"{memory_status.available/1024:.1f} gigabytes available."
        )
        print(f"🤖 {welcome_msg}")
        self.tts.speak(welcome_msg)
    
    def shutdown(self):
        """Shutdown the voice assistant"""
        print("👋 Shutting down voice assistant...")
        self.is_listening = False
        
        # Stop AI backend if we started it
        self.ai_manager.stop_backend()
        
        print("✅ Shutdown complete")
        sys.exit(0)
    
    def _signal_handler(self, signum, frame):
        """Handle system signals"""
        print(f"\n🔔 Received signal {signum}")
        self.shutdown()
    
    def run(self):
        """Main run loop"""
        if not self.setup():
            return
        
        self.startup_message()
        print("🎯 Voice assistant ready! Say 'ziggy' to start.")
        
        try:
            while self.is_listening:
                if self.listen_for_wake_word():
                    self.handle_voice_command()
                    
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            self.shutdown()


def main():
    """Main entry point"""
    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    assistant = CleanVoiceAssistant()
    assistant.run()


if __name__ == "__main__":
    main()