"""Hardware detection and resource profile management."""

import re
import subprocess
import sys

import psutil

from ziggy.config import RESOURCE_PROFILES, PROFILE_ALIASES


class ProfileManager:
    def __init__(self):
        self.current_profile = None
        self.available_memory = 0
        self.settings = {}

    def detect_and_select(self):
        """Auto-detect hardware and pick the right profile."""
        print("  Detecting system resources...")

        # CLI override
        if len(sys.argv) > 2 and sys.argv[1] == "--profile":
            name = sys.argv[2].lower()
            if name in RESOURCE_PROFILES:
                self.current_profile = name
                self.settings = RESOURCE_PROFILES[name].copy()
                print(f"  Using specified profile: {self.settings['name']}")
                return

        self.available_memory = _detect_gpu_memory()
        gb = self.available_memory / 1024

        if gb < 8:
            self.current_profile = "minimal"
        elif gb < 16:
            self.current_profile = "standard"
        else:
            self.current_profile = "performance"

        self.settings = RESOURCE_PROFILES[self.current_profile].copy()
        print(f"  Selected {self.settings['name']} profile ({gb:.1f}GB available)")

    def switch_profile(self, name):
        name = name.lower()
        name = PROFILE_ALIASES.get(name, name)

        if name not in RESOURCE_PROFILES:
            return "I don't recognize that profile. Available profiles are: minimal, standard, and performance."
        if name == self.current_profile:
            return f"I'm already using the {self.settings['name']} profile."

        old_name = self.settings["name"]
        self.current_profile = name
        self.settings = RESOURCE_PROFILES[name].copy()
        return f"Switched from {old_name} to {self.settings['name']} profile. {self.settings['description']}"

    def describe_current(self):
        mem = self._memory_status()
        s = self.settings
        return (
            f"I'm running in {s['name']} mode, "
            f"using {mem['used'] / 1024:.1f} gigabytes of {mem['total'] / 1024:.1f} available. "
            f"I can maintain {s['history_limit']} conversation exchanges "
            f"and record up to {s['recording_conversational'] // 60} minutes."
        )

    def list_profiles(self):
        return (
            "I have three profiles: Minimal for gaming or low resources, "
            "Standard for everyday use, and Performance for extended conversations. "
            f"You're currently using {self.settings['name']} mode."
        )

    def describe_memory(self):
        mem = self._memory_status()
        resp = (
            f"I'm using {mem['used'] / 1024:.1f} gigabytes of "
            f"{mem['total'] / 1024:.1f} available, that's {mem['percent']:.0f} percent. "
        )
        if mem["percent"] > 80:
            resp += "Memory usage is high, you might want to switch to minimal mode."
        return resp

    def _memory_status(self):
        if self.current_profile == "minimal":
            used = 2000
        elif self.current_profile == "standard":
            used = 4000
        else:
            used = 6000
        pct = (used / self.available_memory * 100) if self.available_memory > 0 else 0
        return {
            "total": self.available_memory,
            "used": used,
            "available": self.available_memory - used,
            "percent": pct,
        }


def _detect_gpu_memory():
    """Detect available GPU VRAM in MB."""
    # NVIDIA
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True,
        )
        if result.returncode == 0 and result.stdout.strip():
            line = result.stdout.strip().split("\n")[0]
            if ", " in line:
                total, _ = map(int, line.split(", "))
                print(f"  NVIDIA GPU detected: {total}MB total")
                return total
    except Exception:
        pass

    # AMD ROCm — pick the GPU with the most available VRAM
    try:
        if subprocess.run(["which", "rocm-smi"], capture_output=True).returncode == 0:
            result = subprocess.run(
                ["rocm-smi", "--showmeminfo", "vram"],
                capture_output=True, text=True,
            )
            if result.returncode == 0 and result.stdout:
                gpu_total = {}
                gpu_used = {}
                for line in result.stdout.split("\n"):
                    m = re.search(r"GPU\[(\d+)\].*VRAM Total Memory \(B\):\s*(\d+)", line)
                    if m:
                        gpu_total[m.group(1)] = int(m.group(2))
                    m = re.search(r"GPU\[(\d+)\].*VRAM Total Used Memory \(B\):\s*(\d+)", line)
                    if m:
                        gpu_used[m.group(1)] = int(m.group(2))
                best_available = 0
                best_gpu = None
                for gpu_id, total in gpu_total.items():
                    used = gpu_used.get(gpu_id, 0)
                    avail = total - used
                    if avail > best_available:
                        best_available = avail
                        best_gpu = gpu_id
                if best_gpu is not None:
                    available_mb = best_available // (1024 * 1024)
                    total_mb = gpu_total[best_gpu] // (1024 * 1024)
                    print(f"  AMD GPU[{best_gpu}] selected: {available_mb}MB available ({total_mb}MB total)")
                    return available_mb
    except Exception:
        pass

    # AMD sysfs fallback
    try:
        import glob as glob_mod
        cards = glob_mod.glob("/sys/class/drm/card*/device/mem_info_vram_total")
        if cards:
            with open(cards[0]) as f:
                mb = int(f.read().strip()) // (1024 * 1024)
                print(f"  AMD GPU via sysfs: {mb}MB total ({mb / 1024:.1f}GB)")
                return mb
    except Exception:
        pass

    # Fallback: fraction of system RAM
    ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    estimated = min(ram_gb * 0.25, 8) * 1024
    print(f"  No dedicated GPU detected, using RAM estimate: {int(estimated)}MB")
    return int(estimated)
