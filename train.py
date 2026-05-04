#!/usr/bin/env python3
"""
Train the LLM by crawling the web in real time.

Usage:
  python3 train.py <duration> <detail>

  duration   How long to train:  30s  5m  2h  1d  1h30m
  detail     How deep to learn:  low  medium  high
                                 (or a number 1-10)

Examples:
  python3 train.py 30m low        # quick demo, simple Wikipedia
  python3 train.py 2h medium      # good general knowledge
  python3 train.py 8h high        # deep crawl, larger model
"""
import sys
import os
import re
import time
import queue
import threading
import signal

sys.path.insert(0, os.path.dirname(__file__))


# ── Duration parser ───────────────────────────────────────────────────────────
def parse_duration(s: str) -> int:
    """Parse a human duration string into seconds. e.g. '1h30m' -> 5400"""
    s = s.strip().lower()
    total = 0
    for value, unit in re.findall(r"(\d+(?:\.\d+)?)\s*([smhd])", s):
        v = float(value)
        total += {"s": 1, "m": 60, "h": 3600, "d": 86400}[unit] * v
    if total == 0:
        # Try bare number — treat as minutes
        try:
            total = float(s) * 60
        except ValueError:
            print(f"ERROR: Cannot parse duration '{s}'")
            print("Examples: 30s  5m  2h  1d  1h30m")
            sys.exit(1)
    return int(total)


def format_duration(seconds: int) -> str:
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds//60}m {seconds%60:02d}s"
    return f"{seconds//3600}h {(seconds%3600)//60:02d}m"


# ── Detail level ──────────────────────────────────────────────────────────────
def parse_detail(s: str) -> str:
    """Normalise detail arg to 'low' / 'medium' / 'high'."""
    s = s.strip().lower()
    _ALIASES = {
        "low": "low", "min": "low", "light": "low", "easy": "low", "basic": "low",
        "medium": "medium", "mid": "medium", "normal": "medium", "standard": "medium",
        "high": "high", "heavy": "high", "max": "high", "deep": "high", "full": "high",
    }
    if s in _ALIASES:
        return _ALIASES[s]
    try:
        n = int(s)
        if n <= 3:  return "low"
        if n <= 6:  return "medium"
        return "high"
    except ValueError:
        pass
    print(f"ERROR: Cannot parse detail level '{s}'")
    print("Use: low / medium / high  (or aliases: min, mid, max, light, heavy, deep)")
    sys.exit(1)


# ── Config by detail level ────────────────────────────────────────────────────
_DETAIL_CONFIG = {
    #          model   seq   batch  lr      warmup
    "low":    ("small",  256,  4,   5e-4,   50),
    "medium": ("small",  512,  4,   3e-4,   100),
    "high":   ("medium", 1024, 2,   2e-4,   200),
}


# ── Progress bar ──────────────────────────────────────────────────────────────
def progress_bar(elapsed: int, total: int, width: int = 40) -> str:
    pct   = min(elapsed / total, 1.0)
    done  = int(pct * width)
    bar   = "█" * done + "░" * (width - done)
    return f"[{bar}] {pct*100:.0f}%  {format_duration(elapsed)} / {format_duration(total)}"


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    duration_str = sys.argv[1]
    detail_str   = sys.argv[2]

    duration = parse_duration(duration_str)
    detail   = parse_detail(detail_str)

    model_size, seq_len, batch_size, max_lr, warmup = _DETAIL_CONFIG[detail]

    print()
    print("━" * 60)
    print("  TensorFlow LLM  —  Web-Crawl Training")
    print("━" * 60)
    print(f"  Duration   : {format_duration(duration)}")
    print(f"  Detail     : {detail}")
    print(f"  Model      : {model_size}")
    print(f"  Seq len    : {seq_len}")
    print(f"  Batch size : {batch_size}")
    print("━" * 60)
    print()

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    from tokenizer.byte_tokenizer import ByteTokenizer
    tokenizer = ByteTokenizer()

    # ── Model config ──────────────────────────────────────────────────────────
    from model.config import ModelConfig
    config = getattr(ModelConfig, model_size)()

    # ── Shared state ──────────────────────────────────────────────────────────
    text_queue  = queue.Queue(maxsize=500)
    stop_event  = threading.Event()

    # ── Web crawler ───────────────────────────────────────────────────────────
    from crawler.crawler import WebCrawler
    print("  Starting web crawler...")
    crawler = WebCrawler(
        detail=detail,
        out_queue=text_queue,
        stop_event=stop_event,
    )
    crawl_threads = crawler.start()

    # ── Trainer ───────────────────────────────────────────────────────────────
    from training.stream_trainer import StreamTrainer
    print("  Initialising model...")
    trainer = StreamTrainer(
        config=config,
        tokenizer=tokenizer,
        text_queue=text_queue,
        stop_event=stop_event,
        seq_len=seq_len,
        batch_size=batch_size,
        max_lr=max_lr,
        warmup_steps=warmup,
        output_dir="checkpoints",
        log_every=10,
        save_every=200,
    )

    # ── Timer thread — prints progress and stops training when time is up ─────
    start_time = time.time()

    def timer_loop():
        while not stop_event.is_set():
            elapsed = int(time.time() - start_time)
            remaining = duration - elapsed
            if remaining <= 0:
                print(f"\n  Time's up! ({format_duration(duration)})")
                stop_event.set()
                break
            bar = progress_bar(elapsed, duration)
            stats = crawler.stats
            print(
                f"\r  {bar}  |  pages: {stats.pages_fetched}  "
                f"queue: {text_queue.qsize():<4}",
                end="", flush=True,
            )
            time.sleep(2)

    timer = threading.Thread(target=timer_loop, daemon=True)
    timer.start()

    # Graceful Ctrl-C
    def _sigint(sig, frame):
        print("\n\n  Interrupted — saving checkpoint...")
        stop_event.set()
    signal.signal(signal.SIGINT, _sigint)

    print()
    print(f"  Training for {format_duration(duration)}. Press Ctrl-C to stop early.")
    print()

    # ── Run (blocks until stop_event is set) ──────────────────────────────────
    trainer.run()

    timer.join(timeout=3)
    for t in crawl_threads:
        t.join(timeout=2)

    stats = crawler.stats
    print()
    print("━" * 60)
    print("  Done!")
    print(f"  Pages crawled : {stats.pages_fetched}")
    print(f"  Data fetched  : {stats.bytes_fetched / 1024:.0f} KB")
    print(f"  Tokens seen   : {trainer.total_tokens:,}")
    print(f"  Steps trained : {trainer.global_step}")
    print(f"  Checkpoint    : checkpoints/ckpts")
    print("━" * 60)
    print()
    print("  To chat with the trained model:")
    print("    python3 generate.py")
    print()


if __name__ == "__main__":
    main()
