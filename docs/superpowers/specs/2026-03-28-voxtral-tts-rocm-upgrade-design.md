# Voxtral TTS on AMD RDNA3 via ROCm 7.2.1 Upgrade

## Goal

Get Voxtral-4B-TTS running on the RX 7900 XTX (gfx1100) as Ziggy's performance-profile TTS engine, by upgrading ROCm from 6.4.1 to 7.2.1 and installing vLLM with ROCm support.

## Current State

- **OS:** Ubuntu 24.04.4 LTS, kernel 6.11.0-29-generic
- **ROCm:** 6.4.1 (installed from jammy/22.04 packages on noble/24.04)
- **GPU[0]:** RX 5700 XT (8GB, gfx1010/RDNA1)
- **GPU[1]:** RX 7900 XTX (24GB, gfx1100/RDNA3) — primary inference GPU
- **Ollama:** 0.8.0, running on ROCm, used as LLM backend
- **Open WebUI:** running on port 8000
- **TTS:** Piper (neural, working), espeak (fallback). Voxtral not available.
- **vLLM:** 0.18.0 CUDA build installed (non-functional on AMD)

## Target State

- **ROCm:** 7.2.1 (native Ubuntu 24.04.4 support)
- **vLLM:** 0.18.0+rocm700 wheel in dedicated venv
- **vLLM-Omni:** latest, for Voxtral TTS pipeline
- **Voxtral-4B-TTS-2603:** serving on port 8091, GPU 1 only
- **Ziggy:** auto-selects Voxtral TTS in performance profile

## Staged Upgrade Plan

### Stage 1: ROCm 7.2.1 Upgrade

**Pre-flight:**
1. Stop Ollama (`systemctl stop ollama` or `ollama stop`)
2. Note any other GPU-using services

**Remove ROCm 6.4.1:**
3. Run `amdgpu-uninstall` to remove existing ROCm + amdgpu-dkms driver
4. Clean up any remaining rocm packages: `sudo apt autoremove`

**Install ROCm 7.2.1:**
5. Download: `wget https://repo.radeon.com/amdgpu-install/7.2.1/ubuntu/noble/amdgpu-install_7.2.1.70201-1_all.deb`
6. Install: `sudo apt install ./amdgpu-install_7.2.1.70201-1_all.deb`
7. Run: `sudo amdgpu-install --usecase=rocm`
8. Reboot

**Verify:**
9. `rocminfo` — both GPUs visible (gfx1010 + gfx1100)
10. `rocm-smi --showmeminfo vram` — VRAM totals correct
11. `/opt/rocm/.info/version` — shows 7.2.1

**Gate:** Do not proceed to Stage 2 until all three verifications pass.

### Stage 2: Validate Existing Stack

1. Start Ollama, run a test query to confirm LLM inference works
2. Confirm Open WebUI responds on port 8000
3. Run Ziggy test suite: `.venv/bin/python -m pytest tests/ -v`

**Gate:** All three must pass. If Ollama fails, troubleshoot before proceeding.

### Stage 3: vLLM + Voxtral Installation

**Clean venv:**
1. Remove old venv: `rm -rf ~/.local/share/vllm-env`
2. Create fresh: `python3 -m venv ~/.local/share/vllm-env`

**Install vLLM ROCm wheel:**
3. `pip install vllm==0.18.0+rocm700 --extra-index-url https://wheels.vllm.ai/rocm/0.18.0/rocm700`
4. `pip install git+https://github.com/vllm-project/vllm-omni.git --upgrade`

**Configure:**
5. Stage config at `~/.local/share/vllm-env/voxtral_tts_gpu1.yaml`:
   - Both stages: `devices: "0"` (relative to CUDA_VISIBLE_DEVICES)
   - Stage 0: `enforce_eager: true`, `gpu_memory_utilization: 0.8`
   - Stage 1: `enforce_eager: true`, `gpu_memory_utilization: 0.1`

**Launch:**
6. Environment variables:
   - `CUDA_VISIBLE_DEVICES=1` (expose only the 7900 XTX)
   - `VLLM_USE_TRITON_FLASH_ATTN=0` (no Flash Attention on RDNA3)
   - `HSA_OVERRIDE_GFX_VERSION=11.0.0`
7. Command:
   ```
   vllm serve mistralai/Voxtral-4B-TTS-2603 \
     --stage-configs-path ~/.local/share/vllm-env/voxtral_tts_gpu1.yaml \
     --omni --port 8091 --trust-remote-code --enforce-eager
   ```
8. First run downloads model (~8GB from HuggingFace)

**Verify:**
9. `curl http://localhost:8091/v1/models` — shows Voxtral model
10. Test TTS:
    ```
    curl http://localhost:8091/v1/audio/speech \
      -H "Content-Type: application/json" \
      -d '{"model":"mistralai/Voxtral-4B-TTS-2603","input":"Hello world","voice":"casual_male"}' \
      --output test.wav && aplay test.wav
    ```

**Gate:** Must hear audio output from test.wav.

### Stage 3.5: Ziggy Integration Validation

1. Run `main.py` — should show:
   - `AMD GPU[1] selected: ~23000MB available`
   - `Selected Performance profile`
   - `Voxtral TTS ready (voice: casual_male)`
2. Run test suite: `.venv/bin/python -m pytest tests/ -v` — all pass

## Architecture Notes

- **Port 8091** chosen to avoid conflict with Open WebUI (8000) and Ollama (11434)
- **GPU isolation:** vLLM sees only GPU 1 via `CUDA_VISIBLE_DEVICES=1`. Ollama handles its own GPU selection.
- **VRAM budget:** Voxtral uses ~21GB of 24GB on GPU 1. Ollama's models load/unload dynamically and can share the remaining VRAM or use GPU 0.
- **Ziggy TTS fallback chain:** Voxtral (port 8091) -> Piper -> espeak. If vLLM is down, Piper takes over automatically.
- **Ziggy port config:** Already updated in `ziggy/tts/voxtral.py` (DEFAULT_URL = "http://localhost:8091")

## What We're NOT Changing

- RX 5700 XT configuration (stays as-is)
- Ollama configuration (auto-detects ROCm)
- Kernel version (6.11 is supported by ROCm 7.2.1)
- Open WebUI setup
- Ziggy's test suite or core logic

## Rollback

- **ROCm upgrade fails:** Reinstall ROCm 6.4.1 via AMDGPU installer for Ubuntu 22.04
- **Ollama breaks on 7.2.1:** Reinstall Ollama (`curl -fsSL https://ollama.com/install.sh | sh`)
- **vLLM won't start:** Delete venv, fall back to Piper TTS (already working)
