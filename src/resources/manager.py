import subprocess
from typing import Dict, Optional
from .interfaces import ResourceManagerInterface, ResourceProfile, MemoryStatus


class ResourceManager(ResourceManagerInterface):
    """Manages system resources and profiles"""
    
    # Predefined resource profiles
    PROFILES = {
        "minimal": ResourceProfile(
            name="Minimal",
            description="Low resource usage for gaming or older systems",
            requirements="4-8GB VRAM",
            context_tokens=8000,
            history_limit=10,
            response_tokens=500,
            recording_conversational=120,  # 2 minutes
            recording_command=30
        ),
        "standard": ResourceProfile(
            name="Standard",
            description="Balanced performance for most users",
            requirements="8-16GB VRAM",
            context_tokens=16000,
            history_limit=20,
            response_tokens=1000,
            recording_conversational=300,  # 5 minutes
            recording_command=60
        ),
        "performance": ResourceProfile(
            name="Performance",
            description="Maximum capabilities for high-end systems",
            requirements="16+ GB VRAM",
            context_tokens=32000,
            history_limit=50,
            response_tokens=2000,
            recording_conversational=600,  # 10 minutes
            recording_command=120
        )
    }
    
    # Profile aliases
    ALIASES = {
        "gaming": "minimal",
        "game": "minimal",
        "low": "minimal",
        "normal": "standard",
        "balanced": "standard",
        "default": "standard",
        "fast": "performance",
        "high": "performance",
        "max": "performance",
        "maximum": "performance"
    }
    
    def __init__(self):
        self.current_profile_name = None
        self.available_memory = 0
        self._detect_initial_profile()
    
    def detect_gpu_memory(self) -> int:
        """Detect available GPU memory in MB"""
        # Try NVIDIA first
        try:
            result = subprocess.run([
                'nvidia-smi', 
                '--query-gpu=memory.total,memory.used', 
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True)
            
            if result.returncode == 0 and result.stdout.strip():
                lines = result.stdout.strip().split('\n')
                if lines and ', ' in lines[0]:
                    total, used = map(int, lines[0].split(', '))
                    available = total - used
                    print(f"🎮 NVIDIA GPU detected: {available}MB available ({total}MB total)")
                    return available
        except FileNotFoundError:
            pass
        
        # Try AMD ROCm
        try:
            result = subprocess.run([
                'rocm-smi', '--showmeminfo', 'vram'
            ], capture_output=True, text=True)

            if result.returncode == 0:
                import re
                # Parse per-GPU totals and find the one with most available VRAM
                gpu_total = {}
                gpu_used = {}
                for line in result.stdout.split('\n'):
                    m = re.search(r'GPU\[(\d+)\].*VRAM Total Memory \(B\):\s*(\d+)', line)
                    if m:
                        gpu_total[m.group(1)] = int(m.group(2))
                    m = re.search(r'GPU\[(\d+)\].*VRAM Total Used Memory \(B\):\s*(\d+)', line)
                    if m:
                        gpu_used[m.group(1)] = int(m.group(2))
                best_available = 0
                best_gpu = None
                for gpu_id, total in gpu_total.items():
                    used = gpu_used.get(gpu_id, 0)
                    available = total - used
                    if available > best_available:
                        best_available = available
                        best_gpu = gpu_id
                if best_gpu is not None:
                    available_mb = best_available // (1024 * 1024)
                    total_mb = gpu_total[best_gpu] // (1024 * 1024)
                    print(f"🔴 AMD GPU[{best_gpu}] selected: {available_mb}MB available ({total_mb}MB total)")
                    return available_mb
        except FileNotFoundError:
            pass
        
        # Fallback: estimate based on system RAM
        try:
            import psutil
            system_ram = psutil.virtual_memory().total // (1024 * 1024)  # Convert to MB
            # Assume integrated graphics uses ~1/8 of system RAM
            estimated_vram = system_ram // 8
            print(f"💻 No discrete GPU detected, estimating {estimated_vram}MB from system RAM")
            return estimated_vram
        except ImportError:
            # Final fallback
            print("⚠️ Could not detect GPU memory, assuming 4GB")
            return 4096
    
    def get_profiles(self) -> Dict[str, ResourceProfile]:
        """Get all available resource profiles"""
        return self.PROFILES.copy()
    
    def get_current_profile(self) -> Optional[ResourceProfile]:
        """Get the current active profile"""
        if self.current_profile_name:
            return self.PROFILES.get(self.current_profile_name)
        return None
    
    def switch_profile(self, profile_name: str) -> bool:
        """Switch to a different profile"""
        profile_name = profile_name.lower()
        
        # Resolve aliases
        if profile_name in self.ALIASES:
            profile_name = self.ALIASES[profile_name]
        
        if profile_name not in self.PROFILES:
            return False
        
        old_profile_name = self.current_profile_name
        self.current_profile_name = profile_name
        
        new_profile = self.PROFILES[profile_name]
        print(f"🔄 Switched from {old_profile_name or 'default'} to {new_profile.name} profile")
        print(f"   {new_profile.description}")
        
        return True
    
    def auto_select_profile(self) -> ResourceProfile:
        """Auto-select profile based on available resources"""
        self.available_memory = self.detect_gpu_memory()
        memory_gb = self.available_memory / 1024
        
        if memory_gb < 8:
            self.current_profile_name = "minimal"
        elif memory_gb < 16:
            self.current_profile_name = "standard"
        else:
            self.current_profile_name = "performance"
        
        profile = self.PROFILES[self.current_profile_name]
        print(f"✅ Auto-selected {profile.name} profile ({memory_gb:.1f}GB available)")
        return profile
    
    def get_memory_status(self) -> MemoryStatus:
        """Get current memory usage status"""
        if self.available_memory == 0:
            self.available_memory = self.detect_gpu_memory()
        
        # Estimate usage based on current profile
        if self.current_profile_name == "minimal":
            estimated_used = 2000  # 2GB estimated
        elif self.current_profile_name == "standard":
            estimated_used = 4000  # 4GB estimated
        else:  # performance
            estimated_used = 6000  # 6GB estimated
        
        total = self.available_memory + estimated_used
        percent = (estimated_used / total) * 100 if total > 0 else 0
        
        return MemoryStatus(
            total=total,
            used=estimated_used,
            available=self.available_memory,
            percent=percent
        )
    
    def _detect_initial_profile(self):
        """Detect initial profile on startup"""
        self.auto_select_profile()
    
    def get_profile_info(self) -> str:
        """Get formatted information about current profile"""
        if not self.current_profile_name:
            return "No profile selected"
        
        profile = self.PROFILES[self.current_profile_name]
        mem_status = self.get_memory_status()
        
        return (
            f"Running in {profile.name} mode, "
            f"using {mem_status.used/1024:.1f} gigabytes of {mem_status.total/1024:.1f} available. "
            f"Can maintain {profile.history_limit} conversation exchanges "
            f"and record up to {profile.recording_conversational//60} minutes."
        )
    
    def list_available_profiles(self) -> str:
        """Get formatted list of available profiles"""
        profiles_info = []
        for name, profile in self.PROFILES.items():
            status = " (current)" if name == self.current_profile_name else ""
            profiles_info.append(f"{profile.name}{status}: {profile.description}")
        
        return "Available profiles: " + ", ".join(profiles_info)