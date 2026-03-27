import requests
import json
from typing import Dict, List, Optional
from .interfaces import AIBackendInterface, AIModel, AIResponse


class OllamaBackend(AIBackendInterface):
    """Ollama AI backend implementation"""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url.rstrip('/')
        self.backend_name = "Ollama"
    
    def is_available(self) -> bool:
        """Check if the backend is available"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=2)
            if response.status_code == 200:
                data = response.json()
                return 'models' in data
            return False
        except:
            return False
    
    def list_models(self) -> List[AIModel]:
        """List available models"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                models = []
                for model_data in data.get('models', []):
                    model = AIModel(
                        name=model_data.get('name', ''),
                        size=self._format_size(model_data.get('size', 0))
                    )
                    models.append(model)
                return models
            return []
        except Exception as e:
            print(f"Error listing Ollama models: {e}")
            return []
    
    def query(self, prompt: str, model: Optional[str] = None,
             context: Optional[List[Dict[str, str]]] = None,
             stream: bool = False) -> AIResponse:
        """Query the AI backend"""
        try:
            if not model:
                models = self.list_models()
                if not models:
                    return AIResponse(
                        content="",
                        model="",
                        error="No models available"
                    )
                model = models[0].name
            
            # Build messages
            messages = []
            if context:
                messages.extend(context)
            messages.append({"role": "user", "content": prompt})
            
            # Make request
            response = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": model,
                    "messages": messages,
                    "stream": False
                },
                timeout=60
            )
            
            if response.status_code == 200:
                data = response.json()
                return AIResponse(
                    content=data.get('message', {}).get('content', ''),
                    model=model,
                    tokens_used=data.get('total_duration')
                )
            else:
                return AIResponse(
                    content="",
                    model=model,
                    error=f"API error: {response.status_code}"
                )
                
        except Exception as e:
            return AIResponse(
                content="",
                model=model or "",
                error=str(e)
            )
    
    def unload_models(self) -> None:
        """Unload all loaded models from VRAM"""
        try:
            models = self.list_models()
            for model in models:
                requests.post(
                    f"{self.base_url}/api/generate",
                    json={"model": model.name, "keep_alive": 0},
                    timeout=10
                )
                print(f"🧹 Unloaded {model.name} from VRAM")
        except Exception as e:
            print(f"Warning: could not unload models from VRAM: {e}")

    def get_backend_name(self) -> str:
        """Get the name of the backend"""
        return self.backend_name

    def get_backend_url(self) -> str:
        """Get the URL of the backend"""
        return self.base_url
    
    def _format_size(self, size_bytes: int) -> str:
        """Format size in bytes to human readable"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f}{unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f}TB"