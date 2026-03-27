import subprocess
import time
from typing import Optional
from .interfaces import AIBackendInterface, AIBackendManagerInterface
from .ollama_backend import OllamaBackend
from .msty_backend import MstyBackend


class AIBackendManager(AIBackendManagerInterface):
    """Manages AI backend lifecycle and selection"""
    
    def __init__(self, msty_url: str = "http://localhost:10002",
                 ollama_url: str = "http://localhost:11434"):
        self.msty_url = msty_url
        self.ollama_url = ollama_url
        self.backend_process = None
        self.current_backend = None
        self.current_backend_type = None
    
    def detect_backend(self) -> Optional[str]:
        """Detect which backend is running"""
        # Check Msty first
        msty = MstyBackend(self.msty_url)
        if msty.is_available():
            self.current_backend = msty
            self.current_backend_type = "msty"
            return "msty"
        
        # Check Ollama
        ollama = OllamaBackend(self.ollama_url)
        if ollama.is_available():
            self.current_backend = ollama
            self.current_backend_type = "ollama"
            return "ollama"
        
        return None
    
    def start_backend(self, backend_type: str) -> bool:
        """Start a specific backend"""
        try:
            if backend_type == "msty":
                print("🚀 Starting Msty backend...")
                # Check if msty command exists
                result = subprocess.run(['which', 'msty'], capture_output=True, text=True)
                if result.returncode != 0:
                    print("❌ Msty not found. Please install it first.")
                    return False
                
                # Start msty serve in background
                self.backend_process = subprocess.Popen(
                    ['msty', 'serve'],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                
                # Create backend instance
                backend = MstyBackend(self.msty_url)
                
            else:  # ollama
                print("🚀 Starting Ollama backend...")
                # Check if ollama command exists
                result = subprocess.run(['which', 'ollama'], capture_output=True, text=True)
                if result.returncode != 0:
                    print("❌ Ollama not found. Please install it first.")
                    return False
                
                # Start ollama serve in background
                self.backend_process = subprocess.Popen(
                    ['ollama', 'serve'],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                
                # Create backend instance
                backend = OllamaBackend(self.ollama_url)
            
            # Wait for backend to be ready (up to 30 seconds)
            for i in range(30):
                time.sleep(1)
                if backend.is_available():
                    print(f"✅ {backend_type.capitalize()} backend started successfully")
                    self.current_backend = backend
                    self.current_backend_type = backend_type
                    return True
                if i % 5 == 0:
                    print(f"⏳ Waiting for {backend_type} to start... ({i+1}/30)")
            
            print(f"❌ {backend_type.capitalize()} failed to start within 30 seconds")
            return False
            
        except Exception as e:
            print(f"❌ Error starting {backend_type}: {e}")
            return False
    
    def stop_backend(self) -> None:
        """Stop the running backend"""
        # Unload models from VRAM before stopping
        if self.current_backend and hasattr(self.current_backend, 'unload_models'):
            self.current_backend.unload_models()
        if self.backend_process:
            self.backend_process.terminate()
            self.backend_process.wait(timeout=5)
            self.backend_process = None
        self.current_backend = None
        self.current_backend_type = None
    
    def get_backend(self) -> Optional[AIBackendInterface]:
        """Get the current backend instance"""
        return self.current_backend