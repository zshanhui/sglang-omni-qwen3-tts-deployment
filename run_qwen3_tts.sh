#!/usr/bin/env bash
# ==============================================================================
# Qwen3-TTS on SGLang-Omni — one-shot runbook for a single 48 GB RTX A6000.
#
# Implements steps 1-8 of PLAN.md:
#   01 Install (docker image / venv / qwen-tts extras)   -> `install`
#   02 Start server (48 GB three-process layout)         -> `up`
#   03 Verify (health, voices, smoke synth, streaming)   -> `verify`
#   04 Benchmark (RTF / throughput / CER)                -> `bench`
#   05 Teardown                                          -> `down`
#
# Run this INSIDE the sglang-omni container (or any Linux box with the venv).
# Every step is idempotent: re-running `up` restarts, `verify`/`bench` skip
# work that already succeeded.
#
#   ./run_qwen3_tts.sh install      # once per fresh container
#   ./run_qwen3_tts.sh up           # start the server (blocks ~3-8 min: graph capture)
#   ./run_qwen3_tts.sh verify       # in another shell
#   ./run_qwen3_tts.sh bench
#   ./run_qwen3_tts.sh down
#
#   ./run_qwen3_tts.sh all          # install + up + verify + bench + down
#
# Tune with env vars, e.g.
#   MAX_RUNNING_REQUESTS=32 GPU_MEM_FRACTION=0.72 ./run_qwen3_tts.sh up
# ==============================================================================
set -Eeuo pipefail

# ------------------------------- configuration -------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${LOG_DIR:-$SCRIPT_DIR/logs}"
RUN_DIR="${RUN_DIR:-$SCRIPT_DIR/run}"
VENV_DIR="${VENV_DIR:-$HOME/.venvs/sglang-omni}"
OMNI_REPO="${OMNI_REPO:-$HOME/sglang-omni}"          # cloned for benchmarks/ + examples/
OMNI_VERSION="${OMNI_VERSION:-0.2.0}"
DOCKER_IMAGE="${DOCKER_IMAGE:-hongccc/sglang-omni:dev}"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
VOICE="${VOICE:-Vivian}"
LANG_TAG="${LANG_TAG:-Chinese}"

# --- 48 GB A6000 sizing (PLAN.md §9). 4090 users: see README note at bottom. --
# Three-stage isolation costs ~2.2 GB extra but removes the interpreter
# contention that caps a single replica above ~32 concurrent requests.
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-64}"
MAX_QUEUED_REQUESTS="${MAX_QUEUED_REQUESTS:-64}"
CUDA_GRAPH_MAX_BS="${CUDA_GRAPH_MAX_BS:-64}"
GPU_MEM_FRACTION="${GPU_MEM_FRACTION:-0.72}"          # engine share
PREPROCESS_FRACTION="${PREPROCESS_FRACTION:-0.05}"    # split frontend share
VOCODER_FRACTION="${VOCODER_FRACTION:-0.15}"          # vocoder process share
ISOLATE_STAGES="${ISOLATE_STAGES:-1}"                 # 0 = single-process (24 GB cards)

# --- benchmark ---------------------------------------------------------------
BENCH_LANG="${BENCH_LANG:-zh}"
BENCH_CONCURRENCY="${BENCH_CONCURRENCY:-64}"
BENCH_MAX_SAMPLES="${BENCH_MAX_SAMPLES:-200}"
BENCH_WARMUP="${BENCH_WARMUP:-8}"
BENCH_OUT="${BENCH_OUT:-$SCRIPT_DIR/bench-results}"
SERVER_LOG="$LOG_DIR/tts_server.log"
PID_FILE="$RUN_DIR/tts_server.pid"

# -------------------------------- utilities ----------------------------------
C_RESET=$'\033[0m'; C_BOLD=$'\033[1m'; C_DIM=$'\033[2m'
C_RED=$'\033[31m'; C_GRN=$'\033[32m'; C_YEL=$'\033[33m'; C_BLU=$'\033[34m'

step()  { printf '\n%s==> %s%s\n' "$C_BOLD$C_BLU" "$*" "$C_RESET"; }
info()  { printf '%s    %s%s\n' "$C_DIM" "$*" "$C_RESET"; }
ok()    { printf '%s  ok %s%s\n' "$C_GRN" "$*" "$C_RESET"; }
warn()  { printf '%s warn %s%s\n' "$C_YEL" "$*" "$C_RESET"; }
die()   { printf '\n%serr  %s%s\n' "$C_RED" "$*" "$C_RESET" >&2; exit 1; }

trap 'die "failed at line $LINENO: $BASH_COMMAND"' ERR

need() { command -v "$1" >/dev/null 2>&1 || die "missing required command: $1"; }

have_cuda() { command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; }

gpu_name() {
  nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo unknown
}
gpu_vram_mib() {
  nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 || echo 0
}

# Create the venv if there is none, then activate it. Used by install (to place
# packages) and by up/bench (to find them).
ensure_env() {
  if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    info "using active venv: $VIRTUAL_ENV"
    return
  fi
  if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
    info "creating venv: $VENV_DIR"
    python3 -m venv "$VENV_DIR"
  fi
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
  info "activated venv: $VENV_DIR"
}

# In-container venv is already active; on a bare host we need to enter it.
activate_env() {
  if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    info "using active venv: $VIRTUAL_ENV"
    return
  fi
  if [[ -f "$VENV_DIR/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
    info "activated venv: $VENV_DIR"
    return
  fi
  die "no active venv and $VENV_DIR not found — run '$0 install' first"
}

server_up() {
  curl -fsS --max-time 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1
}

wait_ready() {
  local timeout="${1:-900}" start now
  start=$(date +%s)
  info "waiting for $HOST:$PORT (graph capture can take several minutes)…"
  while :; do
    if server_up; then
      ok "server ready after $(( $(date +%s) - start ))s"
      return 0
    fi
    if [[ -f "$PID_FILE" ]] && ! kill -0 "$(cat "$PID_FILE" 2>/dev/null)" 2>/dev/null; then
      warn "server process exited; last 40 log lines:"
      tail -40 "$SERVER_LOG" 2>/dev/null || true
      return 1
    fi
    now=$(date +%s)
    (( now - start > timeout )) && { warn "timed out after ${timeout}s"; tail -40 "$SERVER_LOG" 2>/dev/null || true; return 1; }
    sleep 5
  done
}

# ------------------------------ 01 install -----------------------------------
do_checks() {
  step "01a Preflight"
  need curl
  local name vram
  name=$(gpu_name); vram=$(gpu_vram_mib)
  info "GPU:  $name (${vram} MiB)"
  if have_cuda; then
    ok "CUDA device(s) visible"
  else
    warn "nvidia-smi unavailable — the server will fail to start without a GPU"
  fi

  case "$name" in
    *"RTX A6000"*|*"A6000"*)
      ok "RTX A6000 detected — 48 GB layout is correct" ;;
    *4090*|*L4*|*A10*|*L40*)
      warn "$name has <=24/48 GB class VRAM. If it is 24 GB, use: ISOLATE_STAGES=0 MAX_RUNNING_REQUESTS=16 GPU_MEM_FRACTION=0.80 $0 up" ;;
    *A100*|*H100*|*H200*)
      info "$name detected — you can raise MAX_RUNNING_REQUESTS and still isolate stages" ;;
    *) info "unrecognised GPU '$name' — continuing with the 48 GB profile" ;;
  esac

  if (( vram > 0 && vram < 40000 )); then
    warn "VRAM ${vram} MiB is below the 48 GB profile; falling back to the 24 GB layout for this run"
    ISOLATE_STAGES=0; MAX_RUNNING_REQUESTS=16; CUDA_GRAPH_MAX_BS=16; GPU_MEM_FRACTION=0.80
  fi

  # SM gate: Qwen3-TTS CuTeDSL fast blocks need SM90/100/103 and >=96 SMs.
  local cc
  cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 || echo "")
  if [[ -n "$cc" ]]; then
    info "compute capability: $cc"
    case "$cc" in
      9.0|10.0|10.3) ok "CuTeDSL fast path available" ;;
      *) info "SM$cc -> default transformer blocks (correct output, just slower; expected on A6000/4090/A100)" ;;
    esac
  fi
}

do_install() {
  do_checks
  step "01b Install sglang-omni $OMNI_VERSION"
  need python3
  # PEP 668: never touch a system interpreter. Create/enter the venv first so
  # every later step (uv, qwen-tts, hf download) lands in the same place. No-op
  # inside the recommended docker image, which already ships an active venv.
  ensure_env
  if python3 -c "import sglang_omni" >/dev/null 2>&1; then
    ok "sglang_omni already importable"
  else
    if ! command -v uv >/dev/null 2>&1; then
      python3 -m pip install --upgrade pip uv
    fi
    uv pip install --prerelease=allow "sglang-omni==$OMNI_VERSION"
    ok "installed sglang-omni $OMNI_VERSION into ${VIRTUAL_ENV:-system}"
  fi

  step "01c Install Qwen3-TTS extras (--no-deps is mandatory)"
  info "qwen-tts pins transformers 4.57.3; installing with deps would replace the supported 5.19 stack"
  info "sox drags a newer numpy past numba's <=2.4 ceiling, which breaks librosa -> import qwen_tts"
  if python3 -c "import qwen_tts" >/dev/null 2>&1; then
    ok "qwen_tts already installed"
  else
    if command -v apt-get >/dev/null 2>&1 && [[ "${EUID:-$(id -u)}" -eq 0 ]]; then
      apt-get update -qq && apt-get install -y -qq sox || warn "apt-get sox failed; the python sox shim may still work"
    elif command -v apt-get >/dev/null 2>&1; then
      warn "not root — skipping apt-get sox; install it manually if sox is missing"
    fi
    if ! command -v uv >/dev/null 2>&1; then
      python3 -m pip install uv
    fi
    uv pip install --no-deps sox einops
    uv pip install --no-deps qwen-tts==0.1.1
    ok "installed qwen-tts 0.1.1 + sox + einops without deps"
  fi

  step "01d Verify the dependency stack was not clobbered"
  local tf_ver
  tf_ver=$(python3 -c "import transformers; print(transformers.__version__)" 2>/dev/null || echo "missing")
  info "transformers: $tf_ver"
  if [[ "$tf_ver" == 4.5* ]]; then
    die "transformers got downgraded to $tf_ver — the supported stack is 5.x. Delete $VENV_DIR and re-run install."
  fi
  python3 -c "import qwen_tts" >/dev/null 2>&1 || die "import qwen_tts still fails (check the numba/numpy ceiling)"
  ok "stack intact: transformers $tf_ver, qwen_tts imports"

  mkdir -p "$LOG_DIR" "$RUN_DIR"

  if [[ "${SKIP_MODEL_DOWNLOAD:-0}" != "1" ]]; then
    step "01e Download checkpoint ($MODEL_PATH, ~3.4 GB)"
    if command -v hf >/dev/null 2>&1; then
      hf download "$MODEL_PATH" >/dev/null && ok "checkpoint cached"
    else
      python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('$MODEL_PATH')
" && ok "checkpoint cached"
    fi
  fi

  step "01f Clone sglang-omni repo (benchmarks + example configs are not in the wheel)"
  if [[ -d "$OMNI_REPO/.git" ]]; then
    ok "repo present: $OMNI_REPO"
  else
    need git
    git clone --depth 1 https://github.com/sgl-project/sglang-omni.git "$OMNI_REPO"
    ok "cloned to $OMNI_REPO"
  fi

  step "Install complete"
  info "next: $0 up"
}

# --------------------------------- 02 up -------------------------------------
do_up() {
  activate_env
  mkdir -p "$LOG_DIR" "$RUN_DIR"

  if server_up; then
    ok "server already healthy on $PORT"
    return
  fi
  if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE" 2>/dev/null)" 2>/dev/null; then
    warn "a server process is running but not healthy; restarting"
    do_down
  fi

  step "02 Start Qwen3-TTS"
  local args=(
    serve
    --model-path "$MODEL_PATH"
    --host "$HOST"
    --port "$PORT"
    --tts_engine.engine.max_running_requests "$MAX_RUNNING_REQUESTS"
    --tts_engine.engine.max_queued_requests "$MAX_QUEUED_REQUESTS"
    --tts_engine.engine.cuda_graph_max_bs "$CUDA_GRAPH_MAX_BS"
  )
  if [[ "$ISOLATE_STAGES" == "1" ]]; then
    info "topology: preprocessing | engine | vocoder (three processes, 48 GB layout)"
    args+=(
      --preprocessing.process tts_frontend
      --preprocessing.gpu 0
      --preprocessing.gpu_memory_fraction "$PREPROCESS_FRACTION"
      --tts_engine.gpu_memory_fraction "$GPU_MEM_FRACTION"
      --tts_engine.engine.mem_fraction_static "$GPU_MEM_FRACTION"
      --vocoder.process vocoder
      --vocoder.gpu_memory_fraction "$VOCODER_FRACTION"
    )
  else
    info "topology: single process (<=24 GB layout)"
    args+=( --mem-fraction-static "$GPU_MEM_FRACTION" )
  fi

  info "max_running_requests=$MAX_RUNNING_REQUESTS  cuda_graph_max_bs=$CUDA_GRAPH_MAX_BS"
  info "logging to $SERVER_LOG"

  # nohup + setsid so the server survives this shell; pid file lets `down` find it
  nohup sgl-omni "${args[@]}" >"$SERVER_LOG" 2>&1 &
  echo $! >"$PID_FILE"
  ok "launched (pid $(cat "$PID_FILE"))"

  wait_ready "${READY_TIMEOUT:-1200}" || die "server did not become healthy; see $SERVER_LOG"
  curl -fsS "http://127.0.0.1:$PORT/v1/models" | head -c 400; echo
}

# ------------------------------- 03 verify -----------------------------------
do_verify() {
  need curl
  server_up || die "server not healthy on $PORT — run '$0 up' first"

  step "03a Health + voices"
  curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null && ok "/health 200"
  local voices
  voices=$(curl -fsS "http://127.0.0.1:$PORT/v1/audio/voices" | python3 -c "
import json,sys
d=json.load(sys.stdin)
print(' '.join(d.get('voices') or []))
" 2>/dev/null || true)
  if [[ -z "$voices" ]]; then
    warn "could not parse /v1/audio/voices"
  else
    info "voices: $voices"
    for want in Vivian Serena Uncle_Fu Ryan; do
      [[ " $voices " == *" $want "* ]] && ok "found speaker $want" || warn "speaker $want missing"
    done
  fi

  step "03b Smoke synthesis (Chinese news line)"
  local out="$RUN_DIR/smoke.wav"
  local payload
  payload=$(python3 - "$MODEL_PATH" "$VOICE" "$LANG_TAG" <<'PY'
import json, sys
model, voice, lang = sys.argv[1:4]
print(json.dumps({
    "model": model,
    "input": "今天上午，国家统计局发布了第三季度经济数据。",
    "voice": voice,
    "language": lang,
    "instructions": "以新闻播报的语气，清晰、平稳、语速适中。",
    "response_format": "wav",
}, ensure_ascii=False))
PY
)
  local t0 t1
  t0=$(python3 -c 'import time;print(time.time())')
  curl -fsS -X POST "http://127.0.0.1:$PORT/v1/audio/speech" \
    -H 'Content-Type: application/json' -d "$payload" -o "$out"
  t1=$(python3 -c 'import time;print(time.time())')
  python3 - "$out" "$t0" "$t1" <<'PY'
import sys, wave
path, t0, t1 = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
with wave.open(path) as w:
    dur = w.getnframes() / w.getframerate()
    print(f"    {path}: {dur:.2f}s audio @ {w.getframerate()} Hz, {w.getnchannels()}ch")
print(f"    wall {(t1-t0):.2f}s -> RTF {(t1-t0)/dur if dur else float('nan'):.3f}")
PY
  ok "smoke synthesis ok -> $out"

  step "03c Streaming (first-chunk latency + underrun sanity)"
  python3 - "$PORT" "$MODEL_PATH" "$VOICE" "$LANG_TAG" <<'PY'
import json, sys, time, urllib.request
port, model, voice, lang = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
body = json.dumps({"model": model, "input": "第一句。第二句。第三句。",
                   "voice": voice, "language": lang,
                   "stream": True, "response_format": "pcm"}).encode()
req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/audio/speech", data=body,
                             headers={"Content-Type": "application/json"})
t0, first, total, chunks = time.time(), None, 0, 0
with urllib.request.urlopen(req, timeout=300) as r:
    while True:
        c = r.read(65536)
        if not c: break
        if first is None: first = time.time()
        total += len(c); chunks += 1
# 24 kHz mono, 2 bytes/sample (s16le) unless the payload is float32
secs = total / (24000 * 2)
print(f"    chunks={chunks} bytes={total} TTFA={(first-t0)*1000:.0f}ms "
      f"audio~{secs:.2f}s (s16le assumption) RTF={(time.time()-t0)/secs if secs else float('nan'):.3f}")
PY
  ok "streaming ok"
  info "listening tip: curl the same body | ffplay -f s16le -ar 24000 -ch_layout mono -"
}

# -------------------------------- 04 bench -----------------------------------
do_bench() {
  activate_env
  server_up || die "server not healthy on $PORT — run '$0 up' first"
  [[ -d "$OMNI_REPO/benchmarks" ]] || die "benchmarks missing — run '$0 install' (clones $OMNI_REPO)"

  step "04a Benchmark: $BENCH_LANG, concurrency $BENCH_CONCURRENCY, $BENCH_MAX_SAMPLES samples"
  info "generate-only first (measures RTF/TTFA without spending GPU time on ASR scoring)"
  (
    cd "$OMNI_REPO"
    python3 -m benchmarks.eval.benchmark_tts_seedtts \
      --use-existing-server \
      --base-url "http://127.0.0.1:$PORT" \
      --model "$MODEL_PATH" \
      --voice "$VOICE" \
      --lang "$BENCH_LANG" \
      --concurrency "$BENCH_CONCURRENCY" \
      --warmup "$BENCH_WARMUP" \
      --max-samples "$BENCH_MAX_SAMPLES" \
      --output-dir "$BENCH_OUT" \
      --generate-only
  ) || die "benchmark (generate) failed"

  if [[ "${BENCH_SKIP_ASR:-0}" == "1" ]]; then
    warn "BENCH_SKIP_ASR=1 -> skipping CER scoring"
  else
    step "04b CER scoring with Qwen3-ASR-1.7B"
    info "this STOPS our TTS server to free the GPU, starts an ASR server on the same port,"
    info "transcribes the saved WAVs, then leaves the GPU idle. Re-run 'up' afterwards."
    do_down
    (
      cd "$OMNI_REPO"
      python3 -m benchmarks.eval.benchmark_tts_seedtts \
        --lang "$BENCH_LANG" \
        --output-dir "$BENCH_OUT" \
        --transcribe-only \
        --port "$PORT"
    ) || die "CER scoring failed"
    ok "reports in $BENCH_OUT"
    info "restart the TTS server with: $0 up"
  fi

  step "04c Latest results"
  find "$BENCH_OUT" -name '*.json' -o -name '*.jsonl' -o -name '*.md' 2>/dev/null | tail -8 | while read -r f; do
    printf '    %s\n' "$f"
  done
}

# -------------------------------- 05 down ------------------------------------
do_down() {
  step "05 Stop server"
  local pid=""
  [[ -f "$PID_FILE" ]] && pid=$(cat "$PID_FILE" 2>/dev/null || true)

  local stopped=1
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    kill -TERM "$pid" 2>/dev/null || true
    for _ in $(seq 1 30); do
      kill -0 "$pid" 2>/dev/null || { stopped=0; break; }
      sleep 1
    done
    if (( stopped )); then
      warn "graceful stop timed out; sending SIGKILL"
      kill -KILL "$pid" 2>/dev/null || true
    fi
    ok "stopped pid $pid"
  fi
  rm -f "$PID_FILE"

  # stage processes are children of separate process groups; sweep by cmdline
  if command -v pkill >/dev/null 2>&1; then
    pkill -f "sgl-omni serve .*$MODEL_PATH" 2>/dev/null && ok "swept leftover stage processes" || true
  fi

  step "05b GPU state"
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv 2>/dev/null || true
}

# --------------------------------- main --------------------------------------
usage() {
  cat <<EOF
Qwen3-TTS on SGLang-Omni — runbook runner

usage: $0 <command>

  install   preflight, install sglang-omni + qwen-tts extras, download model, clone repo
  up        start the server (default 48 GB three-process layout)
  verify    health, voices, smoke synthesis, streaming latency
  bench     SeedTTS benchmark (RTF/TTFA) + optional CER via Qwen3-ASR
  down      stop the server and free the GPU
  status    show health + GPU memory
  all       install -> up -> verify -> bench -> down

env knobs (current):
  MODEL_PATH=$MODEL_PATH
  PORT=$PORT   VOICE=$VOICE
  MAX_RUNNING_REQUESTS=$MAX_RUNNING_REQUESTS   CUDA_GRAPH_MAX_BS=$CUDA_GRAPH_MAX_BS
  ISOLATE_STAGES=$ISOLATE_STAGES  GPU_MEM_FRACTION=$GPU_MEM_FRACTION
  BENCH_LANG=$BENCH_LANG  BENCH_CONCURRENCY=$BENCH_CONCURRENCY  BENCH_MAX_SAMPLES=$BENCH_MAX_SAMPLES
  BENCH_OUT=$BENCH_OUT

24 GB cards (4090 / L4):
  ISOLATE_STAGES=0 MAX_RUNNING_REQUESTS=16 CUDA_GRAPH_MAX_BS=16 GPU_MEM_FRACTION=0.80 $0 up
EOF
}

do_status() {
  if server_up; then ok "healthy on $PORT"; else warn "not healthy on $PORT"; fi
  [[ -f "$PID_FILE" ]] && info "pid: $(cat "$PID_FILE")"
  nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv 2>/dev/null || true
}

main() {
  case "${1:-}" in
    install) do_install ;;
    up)      do_up ;;
    verify)  do_verify ;;
    bench)   do_bench ;;
    down)    do_down ;;
    status)  do_status ;;
    all)
      do_install; do_up; do_verify; do_bench; do_down
      step "Done — results in $BENCH_OUT"
      ;;
    ""|-h|--help|help) usage ;;
    *) usage; die "unknown command: $1" ;;
  esac
}

main "$@"
