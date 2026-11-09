#!/usr/bin/env python3
# ==============================================================================
# tts.py — call the Qwen3-TTS inference API from INSIDE the pod.
#
# Zero dependencies beyond the stdlib (urllib). No `pip install requests`.
# Works against the server started by ./run_qwen3_tts.sh up.
#
#   ./tts.py "今天上午，国家统计局发布了第三季度经济数据。"
#   ./tts.py -o news.wav "第一条。" "第二条。" "第三条。"      # one file per line
#   ./tts.py --voice Uncle_Fu --voice Dylan --split-by voice "…"
#   ./tts.py --stream -o live.wav "长段落…"                   # streamed, still a wav
#   ./tts.py --play "第一句。"                                # play as it arrives
#   ./tts.py --batch lines.txt                                 # one request, 32 items
#   ./tts.py --list-voices
#   ./tts.py --check                                           # health + models
#   cat article.txt | ./tts.py --stdin -o article.wav
#
# Every request prints a one-line report (latency, audio length, RTF, finish
# reason) to stderr so piped/captured audio stays clean.
# ==============================================================================
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import wave
from dataclasses import dataclass, field
from typing import Iterable, Iterator

DEFAULT_BASE = os.environ.get("TTS_BASE_URL", "http://127.0.0.1:8000")
DEFAULT_MODEL = os.environ.get(
    "TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
)
DEFAULT_VOICE = os.environ.get("TTS_VOICE", "Vivian")
DEFAULT_LANG = os.environ.get("TTS_LANG", "Chinese")
PCM_RATE = 24000      # sglang-omni client/audio.py DEFAULT_SAMPLE_RATE
PCM_WIDTH = 2         # s16le, mono — speech_streaming.py PCM_SAMPLE_WIDTH_BYTES
PCM_CHANNELS = 1

DIM, BOLD, RESET = "\033[2m", "\033[1m", "\033[0m"


def log(msg: str) -> None:
    print(f"{DIM}{msg}{RESET}", file=sys.stderr, flush=True)


def die(msg: str, code: int = 1) -> "NoReturn":  # noqa: F821
    print(f"error: {msg}", file=sys.stderr)
    raise SystemExit(code)


# ----------------------------------------------------------------------------- http
def _request(url: str, payload: dict | None, timeout: float, stream: bool = False):
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers)
    try:
        return urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as exc:
        body = exc.read(2000).decode("utf-8", "replace")
        detail = body
        try:
            parsed = json.loads(body)
            detail = parsed.get("detail") or parsed.get("error") or body
        except Exception:
            pass
        die(f"HTTP {exc.code} from {url}: {detail}")
    except urllib.error.URLError as exc:
        die(
            f"cannot reach {url}: {exc.reason}\n"
            "  is the server up?   ./run_qwen3_tts.sh status\n"
            "  start it with:      ./run_qwen3_tts.sh up"
        )


def get_json(base: str, path: str, timeout: float = 30.0) -> dict:
    with _request(base + path, None, timeout) as r:
        return json.loads(r.read())


# ------------------------------------------------------------------------- payload
def build_payload(args: argparse.Namespace, text: str, voice: str) -> dict:
    p: dict = {
        "model": args.model,
        "input": text,
        "voice": voice,
        "language": args.language,
        "response_format": args.format,
        "stream": args.stream,
    }
    if args.instructions:
        p["instructions"] = args.instructions
    if args.speed != 1.0:
        p["speed"] = args.speed
    if args.max_new_tokens:
        p["max_new_tokens"] = args.max_new_tokens
    if args.seed is not None:
        p["seed"] = args.seed
    if args.temperature is not None:
        p["temperature"] = args.temperature
    if args.top_p is not None:
        p["top_p"] = args.top_p
    if args.chunk_frames is not None:
        p["initial_codec_chunk_frames"] = args.chunk_frames
    return p


@dataclass
class Stats:
    t0: float
    ttfa_ms: float | None = None
    audio_s: float = 0.0
    finish: str | None = None
    request_id: str | None = None
    extra: dict = field(default_factory=dict)

    def report(self, label: str) -> None:
        wall = time.time() - self.t0
        rtf = (wall / self.audio_s) if self.audio_s else float("nan")
        bits = [f"wall {wall:5.2f}s", f"audio {self.audio_s:5.2f}s", f"RTF {rtf:5.3f}"]
        if self.ttfa_ms is not None:
            bits.insert(0, f"TTFA {self.ttfa_ms:5.0f}ms")
        if self.finish:
            bits.append(f"finish={self.finish}")
            if self.finish == "length":
                bits.append("(!) hit max_new_tokens — possible runaway")
        log(f"{label}: " + "  ".join(bits))


# ----------------------------------------------------------------------- encoding
def write_wav(path: str, pcm: bytes, rate: int = PCM_RATE) -> float:
    if len(pcm) % 2:
        pcm = pcm[:-1]
    with wave.open(path, "wb") as w:
        w.setnchannels(PCM_CHANNELS)
        w.setsampwidth(PCM_WIDTH)
        w.setframerate(rate)
        w.writeframes(pcm)
    return len(pcm) / (PCM_WIDTH * PCM_CHANNELS * rate)


def parse_sse(line: str) -> dict | None:
    if not line.startswith("data:"):
        return None
    try:
        return json.loads(line[5:].strip())
    except json.JSONDecodeError:
        return None


# ------------------------------------------------------------------------- calling
def synth(args: argparse.Namespace, text: str, voice: str, out: str | None) -> Stats:
    payload = build_payload(args, text, voice)
    stats = Stats(t0=time.time())
    url = args.base.rstrip("/") + "/v1/audio/speech"

    if args.stream:
        return _synth_stream(args, url, payload, out, stats)

    with _request(url, payload, args.timeout) as r:
        body = r.read()
        stats.request_id = r.headers.get("X-Request-Id")
        stats.finish = r.headers.get("X-Finish-Reason")
    wall = time.time() - stats.t0
    # non-streamed body is an encoded container (wav/mp3/flac) — measure via header
    stats.audio_s = _duration_of(body, args.format) or 0.0
    stats.ttfa_ms = wall * 1000
    if out:
        with open(out, "wb") as f:
            f.write(body)
        log(f"wrote {out} ({len(body)/1024:.0f} KiB)")
    else:
        sys.stdout.buffer.write(body)
        sys.stdout.buffer.flush()
    return stats


def _duration_of(body: bytes, fmt: str) -> float | None:
    if fmt == "wav":
        try:
            import io

            with wave.open(io.BytesIO(body)) as w:
                return w.getnframes() / w.getframerate()
        except Exception:
            return None
    if fmt == "pcm":
        return len(body) / (PCM_WIDTH * PCM_CHANNELS * PCM_RATE)
    return None


def _synth_stream(
    args: argparse.Namespace, url: str, payload: dict, out: str | None, stats: Stats
) -> Stats:
    """Stream PCM, optionally decode SSE, optionally play as it arrives."""
    player = None
    if args.play:
        player = _open_player()

    pcm = bytearray()
    sse_wav_header: bytes | None = None  # when response_format=wav + sse

    with _request(url, payload, args.timeout, stream=True) as r:
        stats.request_id = r.headers.get("X-Request-Id")
        content_type = (r.headers.get("Content-Type") or "").lower()
        is_sse = "event-stream" in content_type

        while True:
            raw = r.read(16384)
            if not raw:
                break
            if stats.ttfa_ms is None:
                stats.ttfa_ms = (time.time() - stats.t0) * 1000

            if is_sse:
                for line in raw.decode("utf-8", "replace").splitlines():
                    evt = parse_sse(line)
                    if not evt:
                        continue
                    etype = evt.get("type")
                    if etype == "speech.audio.delta":
                        chunk = base64.b64decode(evt["audio"])
                        if evt.get("response_format") == "wav" and sse_wav_header is None:
                            # first packet is a 44-byte streaming wav header
                            sse_wav_header = chunk[:44]
                            chunk = chunk[44:]
                        pcm.extend(chunk)
                        if player:
                            player.stdin.write(chunk)
                    elif etype == "speech.audio.done":
                        stats.audio_s = float(evt.get("duration") or 0.0)
                        stats.finish = evt.get("finish_reason")
                    elif etype.endswith(".error"):
                        die(f"stream error: {evt.get('error')}")
            else:
                pcm.extend(raw)
                if player:
                    player.stdin.write(raw)

    if player:
        player.stdin.close()
        player.wait()

    stats.audio_s = stats.audio_s or len(pcm) / (PCM_WIDTH * PCM_CHANNELS * PCM_RATE)
    if out:
        if args.format == "wav":
            write_wav(out, bytes(pcm))
        else:
            with open(out, "wb") as f:
                f.write(pcm)
        log(f"wrote {out} ({len(pcm)/1024:.0f} KiB, {args.format})")
    elif not args.play:
        sys.stdout.buffer.write(pcm)
        sys.stdout.buffer.flush()
    return stats


def _open_player():
    if not shutil.which("ffplay"):
        die("--play needs ffplay (apt-get install -y ffmpeg)")
    return subprocess.Popen(
        ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet",
         "-f", "s16le", "-ar", str(PCM_RATE), "-ch_layout", "mono", "-"],
        stdin=subprocess.PIPE,
    )


def batch(args: argparse.Namespace, texts: list[str], voice: str) -> None:
    items = [{"input": t} for t in texts]
    payload = {
        "model": args.model,
        "voice": voice,
        "language": args.language,
        "response_format": args.format,
        "items": items,
    }
    if args.instructions:
        payload["instructions"] = args.instructions
    t0 = time.time()
    with _request(args.base.rstrip("/") + "/v1/audio/speech/batch",
                  payload, args.timeout) as r:
        doc = json.loads(r.read())
    wall = time.time() - t0

    log(f"batch: {doc.get('succeeded')}/{doc.get('total')} ok in {wall:.2f}s "
        f"({doc.get('failed')} failed)")
    outdir = args.outdir or "."
    os.makedirs(outdir, exist_ok=True)
    for item in doc.get("results", []):
        idx = item.get("index")
        if item.get("status") != "ok" or not item.get("audio_data"):
            log(f"  [{idx}] FAILED: {item.get('error')}")
            continue
        fmt = item.get("format") or args.format
        path = os.path.join(outdir, f"batch_{idx:03d}.{fmt}")
        raw = base64.b64decode(item["audio_data"])
        with open(path, "wb") as f:
            f.write(raw)
        log(f"  [{idx}] {path} ({len(raw)/1024:.0f} KiB) finish={item.get('finish_reason')}")


# ------------------------------------------------------------------------- input
def split_text(text: str, mode: str) -> list[str]:
    if mode == "none":
        return [text.strip()] if text.strip() else []
    if mode == "line":
        return [ln.strip() for ln in text.splitlines() if ln.strip()]
    # sentence mode: split after CJK/latin terminators, keep them attached
    parts = re.split(r"(?<=[。！？!?；;\.])\s*", text)
    return [p.strip() for p in parts if p.strip()]


def read_input(args: argparse.Namespace) -> str:
    if args.stdin or not args.text:
        log("reading text from stdin…")
        return sys.stdin.read()
    return " ".join(args.text)


# ------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Call the Qwen3-TTS /v1/audio/speech API from inside the pod.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="examples:\n"
        '  ./tts.py "今天上午，国家统计局发布了第三季度经济数据。"\n'
        "  ./tts.py -o news.wav --split-by sentence article.txt\n"
        "  ./tts.py --stream --play \"第一句。第二句。\"\n"
        "  ./tts.py --voice Dylan --instructions '北京口音，语速稍慢' \"…\"\n",
    )
    ap.add_argument("text", nargs="*", help="text to synthesize (or use --stdin)")
    ap.add_argument("-o", "--out", help="output file; omit to write audio to stdout")
    ap.add_argument("--outdir", default=".", help="directory for --batch outputs")
    ap.add_argument("--base", default=DEFAULT_BASE, help=f"server base URL (default {DEFAULT_BASE})")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--voice", action="append", help="speaker; repeat to cycle per line")
    ap.add_argument("--language", default=DEFAULT_LANG)
    ap.add_argument("--instructions", help="free-text style (1.7B honours this)")
    ap.add_argument("--format", default="wav", choices=["wav", "pcm", "mp3", "flac"])
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--stream", action="store_true", help="request streaming PCM")
    ap.add_argument("--play", action="store_true", help="play while streaming (or after)")
    ap.add_argument("--batch", metavar="FILE", help="one request, up to 32 items, from FILE")
    ap.add_argument("--split-by", choices=["none", "line", "sentence"], default="none",
                    help="split input before synthesizing (default: whole text, one request)")
    ap.add_argument("--chunk-frames", type=int, default=None,
                    help="initial_codec_chunk_frames: lower = faster first audio")
    ap.add_argument("--max-new-tokens", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--stdin", action="store_true", help="read the text from stdin")
    ap.add_argument("--list-voices", action="store_true")
    ap.add_argument("--check", action="store_true", help="health + models, then exit")
    ap.add_argument("--quiet", action="store_true", help="suppress the stderr report")
    args = ap.parse_args()

    if args.quiet:
        global log
        log = lambda *_: None  # noqa: E731

    if args.check:
        try:
            h = get_json(args.base, "/health")
            log(f"health: {json.dumps(h)[:200]}")
        except SystemExit:
            raise
        m = get_json(args.base, "/v1/models")
        print(json.dumps(m, indent=2))
        return

    if args.list_voices:
        v = get_json(args.base, "/v1/audio/voices")
        print("\n".join(v.get("voices") or []))
        return

    voices = args.voice or [DEFAULT_VOICE]

    if args.batch:
        with open(args.batch, encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        if len(lines) > 32:
            die(f"{len(lines)} lines > 32-item server cap; split the file")
        return batch(args, lines, voices[0])

    raw = read_input(args)
    chunks = split_text(raw, args.split_by)
    if not chunks:
        die("no text to synthesize")

    started = time.time()
    total_audio = 0.0
    for i, chunk in enumerate(chunks):
        voice = voices[i % len(voices)]
        out = args.out
        if out and len(chunks) > 1:
            stem, ext = os.path.splitext(out)
            out = f"{stem}_{i:03d}{ext or '.' + args.format}"
        st = synth(args, chunk, voice, out)
        total_audio += st.audio_s
        st.report(f"[{i+1}/{len(chunks)}] {voice}")

    if len(chunks) > 1:
        wall = time.time() - started
        log(f"TOTAL {len(chunks)} chunks, {total_audio:.2f}s audio in {wall:.2f}s "
            f"(RTF {wall/total_audio if total_audio else float('nan'):.3f})")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
