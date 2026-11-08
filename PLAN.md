# Plan: Qwen3-TTS on a minimal viable GPU, served with SGLang

Status: draft · Date: 2026-11-08 · Owner: lishanhui
Goal: stand up a self-hosted, OpenAI-compatible Chinese TTS endpoint with good output
speed on the cheapest card that does not compromise quality, then measure it properly.

---

## 1. Decisions up front

| Decision | Choice | Why |
|---|---|---|
| Runtime | `sglang-omni` **v0.2.0** | TTS is not in SGLang core; `sglang-omni` is the multi-stage runtime that exposes `/v1/audio/speech` and reuses SGLang's paged-KV engine |
| Model | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | best measured Mandarin (0.984% CER non-stream / 0.927% stream, 0/2,020 samples >50% CER) and the fastest under load |
| Not the 0.6B | — | measured **RTF > 1 at concurrency 16** (codec pipeline bound). Fewer params ≠ more output. Also 0.6B does not reliably honor `instructions` |
| Card | **1× RTX 4090 24 GB** | best $/audio; decode is bandwidth-bound (4090 ≈ 1008 GB/s vs L4 ≈ 300 GB/s at the same VRAM) |
| VRAM floor | 16 GB (4–8 streams) · **24 GB (16–24 streams)** · 48 GB (64+) | see §9 |
| License | Apache-2.0 | commercially usable, no negotiation needed |

**Explicitly out of scope for this phase:** voice cloning, S2-Pro, MOSS, Ming, multi-GPU.
Those are evaluated separately; this plan is one model, one card, one endpoint.

### Expected performance (headroom to validate in §8)

Reference: 1× H200 141 GB, BF16, TP1, concurrency 16, 1.7B-CustomVoice.
QPS 13.3 · RTF 0.208 (ZH) · TTFA 105 ms · corpus CER 0.984%.

On a 4090 expect roughly **1/4 – 1/5** of that: ≈15–20 audio-seconds per second,
TTFA ~150–300 ms, ~1 hour of speech per ~4 minutes of GPU time.
**Treat as an estimate until measured.**

---

## 2. Success criteria

- [ ] `/health` returns 200 and `/v1/audio/voices` lists the 9 built-in speakers
- [ ] A 200-character Chinese news paragraph synthesizes with no tonal or polyphone errors on a 50-item hand-check
- [ ] Streaming: zero underruns >50 ms at target concurrency (reference run had 6.2% of ZH streams underrun, max 2,072 ms — this is the real failure mode)
- [ ] Measured RTF ≤ 0.6 at concurrency 16 on the chosen card
- [ ] Cost per audio-minute recorded
- [ ] One-command restart from a clean box (`make up`)

---

## 3. Where to rent the card

Prices below were pulled live from provider APIs on **2026-11-08** and move constantly
(spot markets can swing ±50% within a day). Re-check before committing.

### RTX 4090 (24 GB) — single GPU, on-demand/spot

| Provider | Price | Notes |
|---|---|---|
| **Vast.ai** | min **$0.121** · p10 $0.322 · **median $0.402** · p90 $0.443 (n=64 offers) | cheapest, but host quality varies wildly — filter on `reliability2 ≥ 0.99` |
| **RunPod** | **$0.34 community** / $0.89 secure | community is the sweet spot; better UX than Vast |
| Thunder Compute / Salad / TensorDock | ~$0.25–0.40 | smaller fleets, worth a look for burst |

### Neighbours worth knowing about

| GPU | VRAM | Vast.ai min / median | Why it matters |
|---|---|---|---|
| RTX 4090 | 24 GB | $0.121 / $0.402 | this plan |
| RTX 5090 | 32 GB | $0.213 / $0.497 | +8 GB, SM120, more headroom for 64 streams |
| RTX A6000 | 48 GB | $0.308 / $0.429 | cheapest 48 GB; SM86 — fine for Qwen3-TTS, see §10 |
| **A100 SXM4** | **80 GB** | **$0.296** / $0.802 | 80 GB for under a 4090's median price. **Works for Qwen3-TTS**, refuses S2-Pro (SM80) |
| L40S | 48 GB | $0.417 / $0.602 | Ada, 48 GB |

### For the ASEAN audience (this project's actual target)

Latency to end users matters more than $0.05/hr. 4090 offers seen in-region:

| Location | Price | Network |
|---|---|---|
| Hong Kong | $0.322 | ⚠️ 21 Mbps — too slow to pull 3.4 GB weights comfortably |
| Vietnam | $0.374 | 900 Mbps |
| Thailand | $0.401 / $0.402 / $0.442 | 562 / 248 / 892 Mbps |
| South Korea | $0.402 | 248 Mbps |
| Japan | $0.414 – $0.455 | 474–801 Mbps |

**Recommendation:** start on RunPod community 4090 at $0.34/hr (predictable, good tooling),
then move to a Vast in-region host (TH/VN) once the benchmark is green and latency
to real users is the binding constraint. Verify `reliability2 ≥ 0.99` and
`inet_down ≥ 500 Mbps` — you need to download 3.4 GB of weights per cold start.

### Cloud vs. buy

Renting dominates for this phase: a 4090 at $0.34/hr is ~$245/mo at 24/7, and a
purchase carries depreciation, power, and a dead-capital risk before the model is even
validated. Revisit purchase only after §8 shows the throughput you need.

---

## 4. Step 1 — Host prerequisites

Ubuntu 22.04+, NVIDIA driver with CUDA 12.x, Docker + NVIDIA Container Toolkit.
8 vCPU, 16 GB RAM, 30 GB disk.

```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi   # must print the GPU
```

## 5. Step 2 — Container

```bash
docker pull hongccc/sglang-omni:dev
# pin for reproducibility — :dev moves with main
docker image inspect --format '{{index .RepoDigests 0}}' hongccc/sglang-omni:dev | tee .image-digest

docker run -it --shm-size 32g --gpus all --ipc host --network host --privileged \
  hongccc/sglang-omni:dev /bin/zsh
```

## 6. Step 3 — Install

```bash
pip install --upgrade pip && pip install uv
uv venv .venv -p 3.12
source .venv/bin/activate
uv pip install --prerelease=allow "sglang-omni==0.2.0"
```

### 6.1 Qwen3-TTS extras — the step that bites everyone

```bash
apt-get update && apt-get install -y sox
uv pip install --no-deps sox einops
uv pip install --no-deps qwen-tts==0.1.1
```

`--no-deps` is **mandatory on both lines**, for two independent reasons:

1. `qwen-tts==0.1.1` pins **Transformers 4.57.3**; installing with deps silently replaces
   the Transformers **5.19** / SGLang **0.5.23** stack the runtime is built against.
2. `sox` pulls a newer **numpy**, which violates `numba`'s `numpy<=2.4` ceiling → breaks
   `librosa` → `import qwen_tts` fails with *"Numba needs NumPy 2.4 or less"*.

Do not "fix" a `TypeError` from inside `qwen_tts` by installing its own pin. The pinned
5.19 stack is the supported configuration; sglang-omni shims the API differences.

Gate before proceeding:

```bash
uv pip check
python -c "import torch, transformers, qwen_tts; print(torch.__version__, transformers.__version__)"
```

`transformers` must still report **5.x**. If it reports 4.57.3, recreate the venv.

## 7. Step 4 — Download and launch

```bash
hf download Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice     # 3.4 GB bf16
```

```bash
sgl-omni serve \
  --model-path Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --tts_engine.engine.max_running_requests 16 \
  --tts_engine.engine.max_queued_requests 32 \
  --tts_engine.engine.cuda_graph_max_bs 16 \
  --mem-fraction-static 0.80 \
  --port 8000
```

Flag rationale:

| Flag | Why |
|---|---|
| `max_running_requests 16` | default is 64 → OOM on 24 GB. **The single most important flag at this size.** |
| `cuda_graph_max_bs 16` | graph capture ladder follows the running bound; keeps capture from eating the KV pool |
| `mem-fraction-static 0.80` | default 0.85 leaves too little headroom for the decoder's float32 upsampling spike (vocoder shares the process by default) |
| no `--preprocessing.process` | isolating preprocessing costs **+2.2 GB**. Right at 48 GB+, wrong at 24 GB |
| no `--config` | the YAML holds only `config_cls` + `model_path`; architecture auto-detects from `config.json`. Note `examples/` is **not shipped in the pip wheel** — clone the repo if you want the reference YAMLs |

Startup: **CUDA-graph capture takes several minutes**; `/health` returns **503 `starting`**
until the server's own warmup request completes. `SGLANG_OMNI_WARMUP_TIMEOUT`
(default 600 s) bounds it. Use `--skip-server-warmup` for iteration only.

## 8. Step 5 — Verify, then measure

### 5a. Smoke test

```bash
curl -s localhost:8000/health
curl -s localhost:8000/v1/models
curl -s localhost:8000/v1/audio/voices | python -m json.tool
```

Expect speakers: `Serena, Vivian, Uncle_Fu, Ryan, Aiden, Ono_Anna, Sohee, Eric, Dylan`
(`Eric` = Sichuan, `Dylan` = Beijing; `Vivian` is default).

```bash
curl -X POST localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "input": "今天上午，国家统计局发布了第三季度经济数据。",
    "voice": "Vivian",
    "language": "Chinese",
    "instructions": "以新闻播报的语气，清晰、平稳、语速适中。",
    "response_format": "wav"
  }' --output news.wav
```

### 5b. Streaming

```bash
curl -N -X POST localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input":"第一句。第二句。","voice":"Vivian","language":"Chinese",
       "stream":true,"response_format":"pcm"}' \
| ffplay -f s16le -ar 24000 -ch_layout mono -
```

If output is static, the payload is float32 → retry with `-f f32le`.

### 5c. Benchmark

The benchmark lives in the repo, not the wheel:

```bash
git clone --depth 1 https://github.com/sgl-project/sglang-omni.git && cd sglang-omni
python -m benchmarks.eval.benchmark_tts_seedtts \
  --port 8000 --lang zh --max-concurrency 16 --warmup 8 --max-samples 200
```

Run it again with `--stream` and track **underrun counts**, not just mean RTF.
Watch VRAM concurrently:

```bash
nvidia-smi --query-gpu=memory.used --format=csv -l 1
```

Scale `max_running_requests` upward until VRAM ~90% or p95 latency breaks.

---

## 9. Step 6 — Sizing and tuning

### Sizing ladder

| GPU | VRAM | `max_running_requests` | Isolate preprocessing? | Concurrent streams |
|---|---|---|---|---|
| T4 / A4000 | 16 GB | 8 | no | 4–8 |
| **RTX 4090** | **24 GB** | **16** | **no** | **16–24** |
| 5090 / A6000 | 32–48 GB | 32 | no | 32–48 |
| L40S / A6000 48 GB | 48 GB | 64 | yes (+2.2 GB) | 64+ |

Production layout at 48 GB+ (removes interpreter contention that caps a single
replica above ~32 concurrent):

```bash
sgl-omni serve --model-path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --preprocessing.process tts_frontend --preprocessing.gpu 0 \
  --preprocessing.gpu_memory_fraction 0.05 \
  --tts_engine.gpu_memory_fraction 0.75 \
  --tts_engine.engine.mem_fraction_static 0.75 \
  --tts_engine.engine.max_running_requests 64 \
  --vocoder.process vocoder --vocoder.gpu_memory_fraction 0.12
```

### Tuning matrix, in order of payoff

| Symptom | Lever |
|---|---|
| OOM at startup | `max_running_requests` 16→8, `cuda_graph_max_bs`, `mem-fraction-static` →0.75 |
| First audio feels slow | lower `initial_codec_chunk_frames` (default 5; `0` starts at 25) |
| Playback stutters | **raise** `initial_codec_chunk_frames` and/or buffer client-side |
| Long paragraphs spike VRAM | stream them instead of `stream:false` — peak scales with chunk length, not model size |
| Rare endless audio | cap `max_new_tokens` (runaways: ~0.2% at 0.6B; 2 per ~121k zh requests at 1.7B-CustomVoice) |
| Unreproducible A/B | pin `seed` **and** the container digest — prefix-cache state and batch composition change output |

---

## 10. Risks and gotchas

| Risk | Impact | Mitigation |
|---|---|---|
| **SM89 fallback** | Qwen3-TTS's CuTeDSL fast blocks need SM90/SM100/SM103 with ≥96 SMs; a 4090 falls back to default blocks (correct, just slower) | accept for phase 1; compare H100 only if throughput-bound |
| **No MIG** | CuTeDSL path explicitly does not support MIG / SM-partitioned contexts | don't slice the card expecting several mini-servers |
| Install drift | `qwen-tts` deps destroy the Transformers 5.19 stack | `--no-deps`, `uv pip check` gate, recreate on failure |
| Startup cost | minutes of graph capture + warmup; `/health` 503 meanwhile | don't put it behind a fast-autoscale loop; pre-warm |
| Chinese streaming underruns | reference run: 6.2% of ZH streams underran >50 ms, max 2,072 ms | tune the chunk ramp against client buffer before shipping |
| Price volatility | spot 4090 prices swing ±50%/day | pin an on-demand tier for anything user-facing |
| Model pinning | `:dev` image and HF `main` both move | pin image digest + HF revision |

---

## 11. Local dev without a GPU (optional, do this first)

Native MLX runtime on Apple Silicon, 0.6B CustomVoice, no CUDA needed:

```bash
git clone --depth 1 https://github.com/sgl-project/sglang-omni.git && cd sglang-omni
sglang_omni_mlx/native/scripts/build_runtime.sh ~/omni-runtime
hf download mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-bf16 \
  --revision 6415d95f88be018ff9e46813119dc3bc12261328 --local-dir ~/qwen3-tts-0.6b-cv
~/omni-runtime/bin/qwen3_tts_server --model-path ~/qwen3-tts-0.6b-cv \
  --model-name Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice --port 8000
```

Measured on an M5 / 24 GB: 0.46 s to healthy, 0.22 s p50 first audio, RTF 0.50,
peak 5.6 GB for a 5 s utterance and 13.2 GB for a 20 s one.
**Requests are serialized (1 at a time, 2 queued, rest 503); body limit 1 MiB.**

Use it to validate `instructions`, speaker choice, and text normalization on real
news copy before renting anything.

---

## 12. Open questions

1. Which Chinese voice do we standardize on for news — `Vivian`, `Uncle_Fu`, or a cloned anchor?
2. Do we need polyphone (多音字) *deterministic* control? If yes, this plan is insufficient
   (no pinyin control in Qwen3-TTS) and we need MOSS-TTS Local alongside — see separate eval.
3. Is 16–24 concurrent streams enough for the 10–50 user target, or do we need the 48 GB tier day one?
4. Do we host in-region (TH/VN) for latency, or US/EU for cost, with a CDN in front of the audio?

---

## 13. Next actions

- [ ] Rent a RunPod community 4090 @ $0.34/hr
- [ ] Execute §4–§8, record: RTF, TTFA, underruns, peak VRAM, cost/audio-minute
- [ ] Build the 50-item polyphone/tonal hand-check set from real news copy
- [ ] Decide on §12 open questions 1 and 2
- [ ] If throughput-bound: re-run on A100 80 GB ($0.296–0.80/hr) and compare $/audio-minute
- [ ] Then evaluate MOSS-TTS Local for the teaching path

---

## Appendix A — Reference numbers

1.7B CustomVoice on 1× H200 141 GB, BF16, TP1, concurrency 16
(`max_running_requests 64`, `gpu_memory_fraction` 0.85 engine / 0.10 vocoder):

| Metric | Non-stream ZH | Stream ZH |
|---|---|---|
| Corpus CER | 0.984% | 0.927% |
| Samples >50% CER | 0 | 0 |
| RTF mean | 0.2079 | 0.3332 |
| TTFA mean | — | 104.7 ms |
| QPS | 13.307 | 8.333 |
| No underrun >50 ms | — | 93.76% |

0.6B Base, same setup, for contrast: corpus WER 1.07% (excl. 2 runaways / 1,088),
2 samples >50% WER, and **RTF > 1 at concurrency 16**.

## Appendix B — Files this plan should produce

```
exp-mvi-tts-fishs2/
├── PLAN.md            # this file
├── Makefile           # up / serve / verify / bench / down
├── .image-digest      # pinned container digest
├── scripts/verify.sh  # health + voices + smoke synthesis
└── benchmarks/        # results.jsonl from §8
```
