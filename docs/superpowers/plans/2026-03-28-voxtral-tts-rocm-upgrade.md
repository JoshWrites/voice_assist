# Voxtral TTS on AMD RDNA3 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade ROCm from 6.4.1 to 7.2.1, install vLLM with ROCm support, and get Voxtral-4B-TTS serving on the RX 7900 XTX as Ziggy's performance-profile TTS engine.

**Architecture:** Staged upgrade — ROCm first, validate existing stack, then layer vLLM + Voxtral on top. Each stage is gated by verification before proceeding. vLLM runs in an isolated venv, pinned to GPU 1 via `CUDA_VISIBLE_DEVICES=1`.

**Tech Stack:** ROCm 7.2.1, vLLM 0.18.0+rocm700, vLLM-Omni, Voxtral-4B-TTS-2603, Python 3.12

---

## File Structure

| File | Role |
|---|---|
| `~/.local/share/vllm-env/` | Dedicated venv for vLLM + Voxtral (recreated clean) |
| `~/.local/share/vllm-env/voxtral_tts_gpu1.yaml` | Stage config targeting GPU 1 with enforce_eager (already exists) |
| `ziggy/tts/voxtral.py` | Ziggy's Voxtral TTS client (port already updated to 8091) |
| `tests/test_tts.py` | Existing TTS tests (no changes needed) |

**Note:** This plan is primarily infrastructure/system work. The only code file already modified is `ziggy/tts/voxtral.py` (port changed to 8091 in a prior session). The stage config yaml also already exists with correct settings. This plan focuses on the system-level upgrade and validation steps.

---

### Task 1: Pre-flight — Record Current State

**Context:** Before touching anything, capture the current working state so we can verify restoration later and have rollback data.

- [ ] **Step 1: Record current GPU and ROCm state**

```bash
rocminfo | grep -E "Name:|Marketing Name:" > /tmp/rocm-preflight.txt
rocm-smi --showmeminfo vram >> /tmp/rocm-preflight.txt
cat /opt/rocm/.info/version >> /tmp/rocm-preflight.txt
echo "---" >> /tmp/rocm-preflight.txt
ollama --version >> /tmp/rocm-preflight.txt
cat /tmp/rocm-preflight.txt
```

Expected: Shows gfx1010 + gfx1100, ROCm 6.4.1, Ollama 0.8.0.

- [ ] **Step 2: Verify Ollama is working before we start**

```bash
ollama run qwen2.5-coder:14b-instruct-q4_K_M "Say hello in one sentence" --verbose 2>&1 | tail -5
```

Expected: A response from the model. If this fails, do NOT proceed — fix Ollama first.

- [ ] **Step 3: Record what ROCm packages are installed**

```bash
dpkg -l | grep -E "rocm|amdgpu" | awk '{print $2, $3}' > /tmp/rocm-packages-before.txt
wc -l /tmp/rocm-packages-before.txt
```

Expected: A list of packages for rollback reference.

---

### Task 2: Stop GPU Services

**Context:** The ROCm upgrade replaces the kernel driver (amdgpu-dkms). All GPU-using services must be stopped first.

- [ ] **Step 1: Stop Ollama**

```bash
sudo systemctl stop ollama 2>/dev/null; ollama stop 2>/dev/null; echo "Ollama stopped"
```

Expected: "Ollama stopped" (one of the two commands will work).

- [ ] **Step 2: Check for other GPU processes**

```bash
rocm-smi --showpids 2>/dev/null || echo "No GPU processes found"
```

Expected: No active GPU processes, or only display-related ones.

---

### Task 3: Remove ROCm 6.4.1

**Context:** Remove the existing ROCm stack. The `amdgpu-uninstall` script handles removing both ROCm userspace and the amdgpu-dkms kernel module.

- [ ] **Step 1: Run the uninstaller**

```bash
sudo amdgpu-uninstall -y
```

Expected: Packages removed. May take a few minutes. If `amdgpu-uninstall` is not found, use:
```bash
sudo apt remove --purge amdgpu-dkms rocm-core 2>/dev/null
sudo apt autoremove -y
```

- [ ] **Step 2: Clean up remaining packages**

```bash
sudo apt autoremove -y
dpkg -l | grep -E "rocm|amdgpu" | awk '{print $2}' | head -10
```

Expected: Few or no ROCm/amdgpu packages remaining. Some config packages may linger — that's fine.

- [ ] **Step 3: Remove old repo config if present**

```bash
sudo rm -f /etc/apt/sources.list.d/amdgpu.list /etc/apt/sources.list.d/rocm.list 2>/dev/null
sudo rm -f /etc/apt/preferences.d/rocm-pin-600 2>/dev/null
echo "Old repo configs cleaned"
```

Expected: "Old repo configs cleaned".

---

### Task 4: Install ROCm 7.2.1

**Context:** Download and install the ROCm 7.2.1 AMDGPU installer for Ubuntu 24.04 (noble), then use it to install the full ROCm stack.

- [ ] **Step 1: Download the amdgpu-install package**

```bash
cd /tmp
wget https://repo.radeon.com/amdgpu-install/7.2.1/ubuntu/noble/amdgpu-install_7.2.1.70201-1_all.deb
ls -la amdgpu-install_7.2.1.70201-1_all.deb
```

Expected: File downloaded, roughly 10-20KB (it's just the repo setup package).

- [ ] **Step 2: Install the repo package**

```bash
sudo apt install -y /tmp/amdgpu-install_7.2.1.70201-1_all.deb
```

Expected: Package installs, sets up AMD's apt repos for noble.

- [ ] **Step 3: Update package lists**

```bash
sudo apt update 2>&1 | tail -5
```

Expected: Repo lists updated, should see `repo.radeon.com` entries.

- [ ] **Step 4: Install ROCm with amdgpu-install**

```bash
sudo amdgpu-install --usecase=rocm -y
```

Expected: This installs amdgpu-dkms (kernel driver), ROCm runtime, rocm-smi, rocminfo, and HIP libraries. Will take several minutes. The DKMS module compiles against your kernel.

- [ ] **Step 5: Reboot**

```bash
sudo reboot
```

Expected: System reboots. Reconnect after reboot.

---

### Task 5: Verify ROCm 7.2.1 — GATE

**Context:** After reboot, verify the new ROCm is functional before touching anything else. This is the Stage 1 gate — all three checks must pass.

- [ ] **Step 1: Check ROCm version**

```bash
cat /opt/rocm/.info/version
```

Expected: `7.2.1` (or `7.2.1-xxxx`).

- [ ] **Step 2: Check both GPUs are visible**

```bash
rocminfo | grep -E "Name:|Marketing Name:" | grep -v CPU
```

Expected: Shows both `gfx1010` (RX 5700 XT) and `gfx1100` (RX 7900 XTX).

- [ ] **Step 3: Check VRAM detection**

```bash
rocm-smi --showmeminfo vram
```

Expected: GPU[0] ~8573MB total, GPU[1] ~25753MB total. Same as before upgrade.

- [ ] **Step 4: Compare with pre-flight**

```bash
cat /tmp/rocm-preflight.txt
```

Compare GPU names and VRAM totals. They should match (version will differ).

**GATE: If any check fails, STOP. Do not proceed to Task 6. Troubleshoot or rollback:**
```bash
# Rollback: reinstall ROCm 6.4.1
# wget https://repo.radeon.com/amdgpu-install/6.4.1/ubuntu/jammy/amdgpu-install_6.4.1.60401-1_all.deb
# sudo apt install ./amdgpu-install_6.4.1.60401-1_all.deb
# sudo amdgpu-install --usecase=rocm -y
# sudo reboot
```

---

### Task 6: Validate Ollama — GATE

**Context:** Ollama must still work after the ROCm upgrade. This is the Stage 2 gate.

- [ ] **Step 1: Start Ollama**

```bash
sudo systemctl start ollama 2>/dev/null || ollama serve &
sleep 3
ollama list | head -5
```

Expected: Ollama starts, shows available models.

- [ ] **Step 2: Run a test query**

```bash
ollama run qwen2.5-coder:14b-instruct-q4_K_M "Say hello in one sentence" 2>&1 | head -5
```

Expected: A coherent response from the model. This confirms GPU inference works on ROCm 7.2.1.

- [ ] **Step 3: Verify Open WebUI**

```bash
curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/
```

Expected: `200` (or `301`/`302` redirect). Confirms Open WebUI is still serving.

- [ ] **Step 4: Run Ziggy test suite**

```bash
cd /home/levine/Documents/Repos/voice_assist
.venv/bin/python -m pytest tests/ -v 2>&1 | tail -5
```

Expected: `111 passed` (all tests pass).

**GATE: If Ollama fails, reinstall it before proceeding:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

---

### Task 7: Clean vLLM Environment

**Context:** The existing vllm-env has the CUDA build which won't work. Remove it and create a fresh venv.

- [ ] **Step 1: Remove old venv**

```bash
rm -rf ~/.local/share/vllm-env
```

Expected: Directory removed.

- [ ] **Step 2: Create fresh venv**

```bash
python3 -m venv ~/.local/share/vllm-env
~/.local/share/vllm-env/bin/pip install --upgrade pip 2>&1 | tail -1
```

Expected: "Successfully installed pip-XX.X.X".

---

### Task 8: Install vLLM ROCm Wheel

**Context:** Install the ROCm-specific vLLM wheel from AMD's wheel index. This bundles ROCm-compatible PyTorch.

- [ ] **Step 1: Install vLLM ROCm wheel**

```bash
~/.local/share/vllm-env/bin/pip install vllm==0.18.0+rocm700 \
  --extra-index-url https://wheels.vllm.ai/rocm/0.18.0/rocm700 2>&1 | tail -5
```

Expected: "Successfully installed ... vllm-0.18.0+rocm700 ...". This is a large download (~2GB+), will take a few minutes.

- [ ] **Step 2: Verify vLLM sees the GPU**

```bash
CUDA_VISIBLE_DEVICES=1 HSA_OVERRIDE_GFX_VERSION=11.0.0 \
  ~/.local/share/vllm-env/bin/python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'ROCm: {torch.version.hip}')
print(f'GPUs: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
" 2>&1
```

Expected: Shows PyTorch with ROCm, 1 GPU visible (Radeon RX 7900 XTX).

- [ ] **Step 3: Install vllm-omni**

```bash
~/.local/share/vllm-env/bin/pip install \
  "git+https://github.com/vllm-project/vllm-omni.git" --upgrade 2>&1 | tail -5
```

Expected: "Successfully installed ... vllm-omni-...".

---

### Task 9: Write Stage Config

**Context:** Create the Voxtral stage config targeting GPU 1 with RDNA3-compatible settings. The file existed before but was deleted with the venv.

- [ ] **Step 1: Write the stage config**

Write this file to `~/.local/share/vllm-env/voxtral_tts_gpu1.yaml`:

```yaml
async_chunk: true
stage_args:
  - stage_id: 0
    stage_type: llm
    runtime:
      process: true
      devices: "0"
    engine_args:
      max_num_seqs: 32
      model_stage: audio_generation
      model_arch: VoxtralTTSForConditionalGeneration
      worker_type: ar
      worker_cls: vllm_omni.worker.gpu_ar_worker.GPUARWorker
      scheduler_cls: vllm_omni.core.sched.omni_ar_scheduler.OmniARScheduler
      gpu_memory_utilization: 0.8
      enforce_eager: true
      trust_remote_code: true
      async_scheduling: true
      engine_output_type: latent
      enable_prefix_caching: false
      tokenizer_mode: mistral
      config_format: mistral
      load_format: mistral
      skip_mm_profiling: true
      enable_chunked_prefill: false
      max_model_len: 4096
      custom_process_next_stage_input_func: vllm_omni.model_executor.stage_input_processors.voxtral_tts.generator2tokenizer_async_chunk
    output_connectors:
      to_stage_1: connector_of_shared_memory
    is_comprehension: true
    final_output: false
    final_output_type: text
    default_sampling_params:
      temperature: 0.0
      top_p: 1.0
      top_k: -1
      max_tokens: 2048
      seed: 42
      detokenize: True
      repetition_penalty: 1.1
  - stage_id: 1
    stage_type: llm
    runtime:
      process: true
      devices: "0"
    engine_args:
      max_num_seqs: 32
      model_stage: audio_tokenizer
      model_arch: VoxtralTTSForConditionalGeneration
      worker_type: generation
      worker_cls: vllm_omni.worker.gpu_generation_worker.GPUGenerationWorker
      scheduler_cls: vllm_omni.core.sched.omni_generation_scheduler.OmniGenerationScheduler
      async_scheduling: false
      gpu_memory_utilization: 0.1
      enforce_eager: true
      trust_remote_code: true
      enable_prefix_caching: false
      skip_mm_profiling: true
      engine_output_type: audio
      tokenizer_mode: mistral
      config_format: mistral
      load_format: mistral
      max_num_batched_tokens: 65536
      max_model_len: 65536
    engine_input_source: [0]
    is_comprehension: false
    final_output: true
    final_output_type: audio
    input_connectors:
      from_stage_0: connector_of_shared_memory
    tts_args:
      max_instructions_length: 500
    default_sampling_params:
      temperature: 0.9
      top_p: 0.8
      top_k: 40
      max_tokens: 2048
      seed: 42
      detokenize: True
      repetition_penalty: 1.05

runtime:
  enabled: true
  defaults:
    window_size: -1
    max_inflight: 1

  connectors:
    connector_of_shared_memory:
      name: SharedMemoryConnector
      extra:
        shm_threshold_bytes: 65536
        codec_streaming: true
        connector_get_sleep_s: 0.01
        connector_get_max_wait_first_chunk: 3000
        connector_get_max_wait: 300
        codec_chunk_frames: 25
        codec_chunk_frames_at_begin: 5
        codec_left_context_frames: 25

  edges:
    - from: 0
      to: 1
      window_size: -1
```

- [ ] **Step 2: Verify the file**

```bash
cat ~/.local/share/vllm-env/voxtral_tts_gpu1.yaml | grep enforce_eager
```

Expected: Two lines showing `enforce_eager: true`.

---

### Task 10: Launch Voxtral TTS Server

**Context:** Start the vLLM server with Voxtral. First run will download the model (~8GB). The server runs on port 8091, pinned to GPU 1.

- [ ] **Step 1: Launch vLLM serve**

```bash
CUDA_VISIBLE_DEVICES=1 \
VLLM_USE_TRITON_FLASH_ATTN=0 \
HSA_OVERRIDE_GFX_VERSION=11.0.0 \
nohup ~/.local/share/vllm-env/bin/vllm serve mistralai/Voxtral-4B-TTS-2603 \
  --stage-configs-path ~/.local/share/vllm-env/voxtral_tts_gpu1.yaml \
  --omni --port 8091 --trust-remote-code --enforce-eager \
  > /tmp/vllm-voxtral.log 2>&1 &

echo "vLLM PID: $!"
```

Expected: Process starts in background. First run downloads the model — monitor with:
```bash
tail -f /tmp/vllm-voxtral.log
```

Wait until you see a line like `Uvicorn running on http://0.0.0.0:8091` or `Application startup complete`.

- [ ] **Step 2: Verify the model endpoint**

```bash
curl -s http://localhost:8091/v1/models | python3 -m json.tool
```

Expected: JSON listing showing `mistralai/Voxtral-4B-TTS-2603` in the models array.

---

### Task 11: Test Voxtral TTS Audio — GATE

**Context:** Generate actual speech audio and play it. This is the Stage 3 gate.

- [ ] **Step 1: Generate test audio**

```bash
curl -s http://localhost:8091/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"mistralai/Voxtral-4B-TTS-2603","input":"Hello, this is Ziggy speaking with the Voxtral voice.","voice":"casual_male","response_format":"wav"}' \
  --output /tmp/voxtral-test.wav

ls -la /tmp/voxtral-test.wav
```

Expected: WAV file created, should be several KB to a few hundred KB.

- [ ] **Step 2: Play the audio**

```bash
aplay /tmp/voxtral-test.wav
```

Expected: You hear spoken audio saying "Hello, this is Ziggy speaking with the Voxtral voice."

- [ ] **Step 3: Check VRAM usage**

```bash
rocm-smi --showmeminfo vram
```

Expected: GPU[1] shows significant VRAM usage (~18-21GB used out of ~25GB).

**GATE: If no audio, check `/tmp/vllm-voxtral.log` for errors. If vLLM won't start on ROCm, fall back to Piper — delete the venv and Ziggy will auto-select Piper.**

---

### Task 12: Validate Ziggy Integration

**Context:** Run Ziggy and confirm it auto-detects the 7900 XTX, selects Performance profile, and uses Voxtral TTS.

- [ ] **Step 1: Run Ziggy**

```bash
cd /home/levine/Documents/Repos/voice_assist
.venv/bin/python main.py
```

Expected output should include:
```
AMD GPU[1] selected: ~23000MB available (~24000MB total)
Selected Performance profile
Checking for Voxtral TTS server...
Voxtral TTS ready (voice: casual_male)
```

Stop Ziggy with Ctrl+C after confirming the startup messages.

- [ ] **Step 2: Run the test suite**

```bash
.venv/bin/python -m pytest tests/ -v 2>&1 | tail -5
```

Expected: `111 passed`.

- [ ] **Step 3: Commit all changes**

```bash
cd /home/levine/Documents/Repos/voice_assist
git add -A
git commit -m "Add Voxtral TTS setup: ROCm 7.2.1 upgrade plan, stage config, port update

- Update Voxtral default port to 8091 (avoid Open WebUI on 8000)
- Add design spec and implementation plan for ROCm upgrade
- Stage config uses enforce_eager for RDNA3 compatibility"
```
