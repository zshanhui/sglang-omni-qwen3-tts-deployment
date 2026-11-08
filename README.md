# Qwen3-TTS on SGLang-Omni — runbook runner

`run_qwen3_tts.sh` automates PLAN.md §4–§8 for a single-GPU box.
Validated against `sglang-omni` **v0.2.0**.

## Quick start (on the RunPod A6000)

```bash
# 1. get the script onto the box (or clone the repo you keep it in)
scp run_qwen3_tts.sh root@<pod>:/workspace/

# 2. install everything (idempotent, ~5 min + 3.4 GB model download)
./run_qwen3_tts.sh install

# 3. start the server — blocks until /health is ready (graph capture: 3–8 min)
./run_qwen3_tts.sh up

# 4. in a second shell
./run_qwen3_tts.sh verify
./run_qwen3_tts.sh bench

# 5. stop
./run_qwen3_tts.sh down
```

`./run_qwen3_tts.sh all` chains install → up → verify → bench → down.

## Commands

| Command | Does | Notes |
|---|---|---|
| `install` | preflight, venv, `sglang-omni==0.2.0`, qwen-tts extras, model download, repo clone | Idempotent. Verifies `transformers` is still 5.x afterwards. |
| `up` | starts `sgl-omni serve`, waits for `/health` | Backgrounded with a pid file; survives shell exit. |
| `verify` | `/health`, `/v1/audio/voices`, smoke synthesis, streaming TTFA | Writes `run/smoke.wav`. |
| `bench` | SeedTTS benchmark: RTF, TTFA, throughput; then CER via Qwen3-ASR | **Stops the TTS server** for the ASR pass (one GPU). |
| `down` | graceful stop + sweep leftover stage processes | |
| `status` | health + GPU memory | |
| `all` | the whole pipeline | |

## Defaults (48 GB RTX A6000, PLAN.md §9)

```
topology          preprocessing | engine | vocoder   (three processes)
max_running_requests   64
cuda_graph_max_bs      64
gpu_memory_fraction    0.72 engine / 0.05 frontend / 0.15 vocoder
```

**24 GB cards (4090 / L4)** — single process, 16 concurrent:

```bash
ISOLATE_STAGES=0 MAX_RUNNING_REQUESTS=16 CUDA_GRAPH_MAX_BS=16 GPU_MEM_FRACTION=0.80 \
  ./run_qwen3_tts.sh up
```

The script auto-detects VRAM < 40 GB and applies this profile for you.

## Env knobs

| Var | Default | Purpose |
|---|---|---|
| `PORT` | `8000` | server port |
| `VOICE` | `Vivian` | one of `Serena Vivian Uncle_Fu Ryan Aiden Ono_Anna Sohee Eric Dylan` |
| `MODEL_PATH` | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | swap for `-Base` to clone voices |
| `BENCH_LANG` | `zh` | `zh` or `en` |
| `BENCH_CONCURRENCY` | `64` | matches `max_running_requests` |
| `BENCH_MAX_SAMPLES` | `200` | use `2020` for the full ZH set |
| `BENCH_SKIP_ASR` | `0` | `1` = skip CER scoring (keeps the server running) |
| `READY_TIMEOUT` | `1200` | seconds to wait for `/health` |
| `SKIP_MODEL_DOWNLOAD` | `0` | `1` = assume the checkpoint is already cached |

## Gotchas this script already handles

- **`--no-deps` on `qwen-tts`.** Installing it with dependencies replaces
  Transformers 5.19 with the 4.57.3 pin the runtime is built against. The script
  installs bare and then asserts `transformers` is still 5.x, failing loudly if not.
- **PEP 668.** It never installs into a system interpreter; it creates and enters
  `~/.venvs/sglang-omni` first. (No-op inside the `hongccc/sglang-omni:dev` image.)
- **`examples/` is not in the pip wheel.** The benchmark needs the git clone, so
  `install` clones to `~/sglang-omni`.
- **The benchmark stops/starts servers.** `--use-existing-server` is used for the
  generate pass; the ASR pass needs the whole card, so the TTS server is stopped first.
- **`--config` is intentionally not passed.** The YAML holds only `config_cls` +
  `model_path`; the architecture auto-detects from `config.json`.

## Expected output on an A6000

The script reports these; the H200 reference is QPS 13.3 / RTF 0.208 / TTFA 105 ms
at concurrency 16 (PLAN.md Appendix A).

| Metric | A6000 expectation |
|---|---|
| per-stream RTF @ conc 16 | ~0.3–0.5 (768 GB/s vs H200's 4.8 TB/s) |
| TTFA | ~150–400 ms |
| aggregate | ~12 audio-s/s ≈ 12 h of speech per GPU-hour |
| cost | ~$0.0005 per audio-minute @ $0.33/hr |

**Treat every row as an estimate until `bench` prints your own numbers.**

## CuTeDSL note

Qwen3-TTS's fast transformer blocks require SM90/SM100/SM103 with ≥96 SMs and do
not support MIG. The A6000 is **SM86**, so it silently uses the default blocks —
correct output, just slower. This is expected and is the same for 4090 (SM89) and
A100 (SM80). `install` prints the detected compute capability so you can confirm.
