#!/usr/bin/env bash
# ==============================================================================
# Push run_qwen3_tts.sh (+ README) to a remote GPU host and optionally start it.
#
#   ./sync.sh                          # show the runpod SFTP/ssh command to copy from
#   ./sync.sh root@1.2.3.4 -p 40022    # scp to that host
#   ./sync.sh root@1.2.3.4 -p 40022 --install
#   ./sync.sh <host> [-p PORT] [-i KEY] [--remote-dir DIR] [--install|--up]
#
# RunPod puts the host + port on the pod's "Connect" panel, e.g.
#   ssh root@213.173.111.42 -p 40188 -i ~/.ssh/id_ed25519
# Copy those exact values. RunPod images already have curl, git, python3.12,
# nvidia-smi and an active venv, so `--install` inside the image is usually a
# no-op except for the qwen-tts extras and the model download.
# ==============================================================================
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FILES=(run_qwen3_tts.sh README.md)

HOST=""; PORT="22"; KEY=""; REMOTE_DIR="/workspace"; ACTION=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -p|--port)       PORT="$2"; shift 2 ;;
    -i|--identity)   KEY="$2"; shift 2 ;;
    -d|--remote-dir) REMOTE_DIR="$2"; shift 2 ;;
    --install)       ACTION="install"; shift ;;
    --up)            ACTION="up"; shift ;;
    --verify)        ACTION="verify"; shift ;;
    -h|--help)       sed -n '3,14p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    -*)              echo "unknown flag: $1" >&2; exit 2 ;;
    *)               HOST="$1"; shift ;;
  esac
done

C_B=$'\033[1m'; C_D=$'\033[2m'; C_R=$'\033[0m'; C_G=$'\033[32m'; C_Y=$'\033[33m'

if [[ -z "$HOST" ]]; then
  cat <<EOF

${C_B}No host given — here is what to copy from the RunPod panel.${C_R}

  1. Open the pod → ${C_B}Connect${C_R} → "Direct TCP ports" / SSH.
     You get a line like:

       ${C_D}ssh root@213.173.111.42 -p 40188 -i ~/.ssh/id_ed25519${C_R}

  2. Re-run this script with those values:

       ./sync.sh root@213.173.111.42 -p 40188

     or just paste the pod's own command and run ${C_B}scp${C_R} yourself:

       scp -P 40188 -i ~/.ssh/id_ed25519 run_qwen3_tts.sh README.md \\
           root@213.173.111.42:/workspace/

  3. Then on the pod:

       ${C_D}cd /workspace && chmod +x run_qwen3_tts.sh && ./run_qwen3_tts.sh install${C_R}

${C_B}If you are running inside the pod's web terminal / Jupyter,${C_R}
you do not need ssh at all — see "No ssh: paste method" in README.md.

EOF
  exit 0
fi

SSH_OPTS=(-p "$PORT" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20)
SCP_OPTS=(-P "$PORT" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20)
[[ -n "$KEY" ]] && { SSH_OPTS+=(-i "$KEY"); SCP_OPTS+=(-i "$KEY"); }

for f in "${FILES[@]}"; do
  [[ -f "$SCRIPT_DIR/$f" ]] || { echo "missing $SCRIPT_DIR/$f" >&2; exit 1; }
done

echo "${C_B}==>${C_R} checking ssh to $HOST:$PORT"
if ! ssh "${SSH_OPTS[@]}" "$HOST" 'echo ok' 2>/dev/null | grep -q ok; then
  echo "${C_Y}warn${C_R} ssh failed. Common causes on RunPod:"
  echo "   - wrong port (it is NOT 22; it is the per-pod port from the Connect panel)"
  echo "   - your public key is not in the pod (RunPod injects the keys on your account)"
  echo "   - the pod is stopped"
  echo
  echo "Fallback: use the Jupyter web terminal in the RunPod UI and paste the script."
  exit 1
fi
echo "${C_G}  ok${C_R} reachable"

echo "${C_B}==>${C_R} creating $REMOTE_DIR"
ssh "${SSH_OPTS[@]}" "$HOST" "mkdir -p '$REMOTE_DIR'"

echo "${C_B}==>${C_R} copying ${FILES[*]}"
scp "${SCP_OPTS[@]}" "${FILES[@]/#/$SCRIPT_DIR/}" "$HOST:$REMOTE_DIR/"
ssh "${SSH_OPTS[@]}" "$HOST" "chmod +x '$REMOTE_DIR/run_qwen3_tts.sh'"
echo "${C_G}  ok${C_R} copied to $HOST:$REMOTE_DIR"

# quick remote sanity check
echo "${C_B}==>${C_R} remote preflight"
ssh "${SSH_OPTS[@]}" "$HOST" "cd '$REMOTE_DIR' && ./run_qwen3_tts.sh --help | head -3"

if [[ -n "$ACTION" ]]; then
  echo "${C_B}==>${C_R} running: ./run_qwen3_tts.sh $ACTION"
  exec ssh "${SSH_OPTS[@]}" -t "$HOST" "cd '$REMOTE_DIR' && ./run_qwen3_tts.sh $ACTION"
fi

cat <<EOF

${C_G}done.${C_R} Next, on the pod:

  ssh ${SSH_OPTS[*]} $HOST
  cd $REMOTE_DIR && ./run_qwen3_tts.sh install

or from here:

  ./sync.sh $HOST -p $PORT --install

EOF
