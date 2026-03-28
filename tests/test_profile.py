"""Tests for ziggy.profile — hardware detection and profile management."""

from unittest.mock import patch, MagicMock

from ziggy.profile import ProfileManager, _detect_gpu_memory
from ziggy.config import RESOURCE_PROFILES


class TestProfileManager:
    def _make(self, memory_mb=8192):
        pm = ProfileManager()
        pm.available_memory = memory_mb
        pm.current_profile = "standard"
        pm.settings = RESOURCE_PROFILES["standard"].copy()
        return pm

    def test_switch_profile_valid(self):
        pm = self._make()
        result = pm.switch_profile("minimal")
        assert pm.current_profile == "minimal"
        assert "Switched" in result
        assert "Minimal" in result

    def test_switch_profile_alias(self):
        pm = self._make()
        result = pm.switch_profile("gaming")
        assert pm.current_profile == "minimal"

    def test_switch_profile_already_active(self):
        pm = self._make()
        result = pm.switch_profile("standard")
        assert "already" in result.lower()

    def test_switch_profile_invalid(self):
        pm = self._make()
        result = pm.switch_profile("turbo")
        assert "don't recognize" in result.lower()
        assert pm.current_profile == "standard"

    def test_describe_current(self):
        pm = self._make(memory_mb=16384)
        result = pm.describe_current()
        assert "Standard" in result
        assert "gigabytes" in result

    def test_list_profiles(self):
        pm = self._make()
        result = pm.list_profiles()
        assert "Minimal" in result
        assert "Standard" in result
        assert "Performance" in result

    def test_describe_memory(self):
        pm = self._make(memory_mb=16384)
        result = pm.describe_memory()
        assert "gigabytes" in result

    def test_describe_memory_high_usage_warning(self):
        pm = self._make(memory_mb=4000)
        pm.current_profile = "performance"
        pm.settings = RESOURCE_PROFILES["performance"].copy()
        result = pm.describe_memory()
        assert "high" in result.lower()

    @patch("ziggy.profile.subprocess.run")
    def test_detect_and_select_minimal(self, mock_run):
        # No GPU found, low RAM fallback
        mock_run.side_effect = FileNotFoundError
        with patch("ziggy.profile.psutil.virtual_memory") as mock_vm:
            mock_vm.return_value = MagicMock(total=4 * 1024**3)  # 4GB RAM
            pm = ProfileManager()
            pm.detect_and_select()
            assert pm.current_profile == "minimal"

    @patch("ziggy.profile.subprocess.run")
    def test_detect_and_select_performance(self, mock_run):
        # NVIDIA GPU with 24GB
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "24576, 1024"
        mock_run.return_value = mock_result
        pm = ProfileManager()
        pm.detect_and_select()
        assert pm.current_profile == "performance"


class TestDetectGpuMemory:
    @patch("ziggy.profile.subprocess.run")
    def test_nvidia_detection(self, mock_run):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "16384, 2048"
        mock_run.return_value = mock_result
        result = _detect_gpu_memory()
        assert result == 16384

    @patch("ziggy.profile.subprocess.run")
    def test_amd_multi_gpu_picks_best(self, mock_run):
        """Should pick the GPU with the most available VRAM."""
        which_result = MagicMock(returncode=0)
        rocm_output = (
            "GPU[0] : VRAM Total Memory (B): 17179869184\n"
            "GPU[0] : VRAM Total Used Memory (B): 10737418240\n"
            "GPU[1] : VRAM Total Memory (B): 17179869184\n"
            "GPU[1] : VRAM Total Used Memory (B): 2147483648\n"
        )
        rocm_result = MagicMock(returncode=0, stdout=rocm_output)

        def side_effect(cmd, **kwargs):
            if cmd[0] == "nvidia-smi":
                raise FileNotFoundError
            if "which" in cmd:
                return which_result
            return rocm_result

        mock_run.side_effect = side_effect
        result = _detect_gpu_memory()
        # GPU[1] has 17179869184 - 2147483648 = 15032385536 bytes = 14336 MB
        assert result == 14336

    @patch("ziggy.profile.subprocess.run")
    def test_fallback_to_ram_estimate(self, mock_run):
        mock_run.side_effect = FileNotFoundError
        with patch("ziggy.profile.psutil.virtual_memory") as mock_vm, \
             patch("glob.glob", return_value=[]):
            mock_vm.return_value = MagicMock(total=32 * 1024**3)
            result = _detect_gpu_memory()
            # min(32 * 0.25, 8) * 1024 = 8192
            assert result == 8192
