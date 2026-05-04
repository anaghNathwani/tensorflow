#!/usr/bin/env python3
"""
Interactive terminal chat interface for TensorFlow LLM.

Usage:
  python3 generate.py --checkpoint checkpoints/ckpts
  python3 generate.py --checkpoint checkpoints/ckpts --size medium
  python3 generate.py --checkpoint checkpoints/ckpts --temperature 0.9
"""
import argparse
import sys
import os
import shutil

sys.path.insert(0, os.path.dirname(__file__))

# ── ANSI codes ────────────────────────────────────────────────────────────────
RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
CYAN   = "\033[36m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"

_USE_COLOR = sys.stdout.isatty()


def c(*codes: str) -> str:
    """Return ANSI prefix only when stdout is a real tty."""
    return ("".join(codes) if _USE_COLOR else "")


def cprint(text: str, *codes: str, **kw):
    print(c(*codes) + text + (RESET if _USE_COLOR else ""), **kw)


def tw() -> int:
    return shutil.get_terminal_size((80, 24)).columns


def hr(char: str = "─"):
    print(c(DIM) + char * tw() + (RESET if _USE_COLOR else ""))


# ── Banner / help ─────────────────────────────────────────────────────────────
def print_banner():
    title = "TensorFlow LLM  -  Interactive Chat"
    pad   = max((tw() - len(title)) // 2, 0)
    print()
    hr("=")
    cprint(" " * pad + title, BOLD, CYAN)
    hr("=")
    cprint("  /help  /clear  /settings  /set <k> <v>  /reset  /quit", DIM)
    hr()
    print()


def print_help():
    cprint("\nCommands", BOLD)
    rows = [
        ("/help",           "Show this message"),
        ("/clear",          "Clear the screen"),
        ("/settings",       "Show generation settings"),
        ("/set key value",  "Change a setting  (e.g. /set temperature 0.9)"),
        ("/reset",          "Clear conversation history"),
        ("/quit | /exit",   "Exit"),
    ]
    col = max(len(r[0]) for r in rows) + 4
    for cmd, desc in rows:
        print("  " + c(CYAN, BOLD) + cmd + RESET + " " * (col - len(cmd)) + c(DIM) + desc + RESET)
    print()


def print_settings(settings: dict):
    cprint("\nSettings", BOLD)
    col = max(len(k) for k in settings) + 4
    for k, v in settings.items():
        print("  " + c(CYAN) + k + RESET + " " * (col - len(k)) + str(v))
    print()


# ── Model loading ─────────────────────────────────────────────────────────────
def load_model(args):
    import tensorflow as tf
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    tf.get_logger().setLevel("ERROR")

    # ── Auto-bootstrap if no checkpoint exists ────────────────────────────────
    from pathlib import Path
    has_ckpt = tf.train.latest_checkpoint(args.checkpoint) is not None
    if not has_ckpt:
        cprint("No checkpoint found — running bootstrap training on bundled data...", YELLOW)
        import bootstrap
        bootstrap.main(epochs=2, quiet=False)

    if args.tokenizer:
        cprint("Loading tokenizer...", DIM)
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    else:
        from tokenizer.byte_tokenizer import ByteTokenizer
        tokenizer = ByteTokenizer()
        cprint("Using built-in byte tokenizer (no download needed)", DIM)

    cprint("Building model...   ", DIM)
    from model.config import ModelConfig
    from model.transformer import TensorFlowLLM

    config = ModelConfig.from_json(args.config) if args.config \
             else getattr(ModelConfig, args.size)()

    tf.keras.mixed_precision.set_global_policy("mixed_bfloat16")
    model = TensorFlowLLM(config, name="llm")

    # Trigger graph build
    dummy = tf.zeros((1, 1), dtype=tf.int32)
    model(dummy, use_cache=False, training=False)

    ckpt   = tf.train.Checkpoint(model=model)
    latest = tf.train.latest_checkpoint(args.checkpoint)
    if latest:
        ckpt.restore(latest).expect_partial()
        cprint(f"Checkpoint: {latest}", DIM)
    else:
        cprint("WARNING: no checkpoint found — using random weights", YELLOW)

    cprint(f"Model ready  ({model.count_params()/1e6:.1f}M params)\n", DIM)
    return model, tokenizer, config


# ── Generation ────────────────────────────────────────────────────────────────
def stream_response(model, tokenizer, prompt: str,
                    history: list[dict], settings: dict) -> str:
    from inference.generate import Generator

    gen = Generator(
        model, tokenizer,
        max_new_tokens     = settings["max_tokens"],
        temperature        = settings["temperature"],
        top_k              = settings["top_k"],
        top_p              = settings["top_p"],
        repetition_penalty = settings["repetition_penalty"],
    )

    # Build conversation context
    context = ""
    for turn in history:
        context += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n\n"
    context += f"User: {prompt}\nAssistant:"

    print()
    print(c(BOLD, GREEN) + "Assistant" + RESET)
    print()

    full   = ""
    col    = 0
    wrap_w = tw() - 4

    sys.stdout.write("  ")
    for token in gen._stream(context):
        full += token
        for ch in token:
            if ch == "\n":
                sys.stdout.write("\n  ")
                col = 0
            else:
                sys.stdout.write(ch)
                col += 1
                if col >= wrap_w and ch == " ":
                    sys.stdout.write("\n  ")
                    col = 0
        sys.stdout.flush()

    sys.stdout.write("\n\n")
    sys.stdout.flush()
    return full


# ── Input prompt (ANSI-safe) ──────────────────────────────────────────────────
def prompt_input(label: str) -> str:
    """Print a coloured label then read input — keeps ANSI out of input() itself."""
    sys.stdout.write(c(BOLD, YELLOW) + label + RESET)
    sys.stdout.flush()
    return input()


# ── Arg parsing ───────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",  default="checkpoints/ckpts")
    p.add_argument("--config",      default=None)
    p.add_argument("--size",        default="small", choices=["small","medium","large"])
    p.add_argument("--tokenizer",   default=None,
                   help="HuggingFace tokenizer name/path (optional). "
                        "Omit to use the built-in byte tokenizer — no login needed.")
    p.add_argument("--max_tokens",  type=int,   default=256)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k",       type=int,   default=50)
    p.add_argument("--top_p",       type=float, default=0.95)
    p.add_argument("--rep_penalty", type=float, default=1.1, dest="repetition_penalty")
    p.add_argument("--system",      default=None)
    p.add_argument("--no_history",  action="store_true")
    return p.parse_args()


# ── Main loop ─────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    print_banner()
    model, tokenizer, _ = load_model(args)

    settings = {
        "temperature":        args.temperature,
        "top_k":              args.top_k,
        "top_p":              args.top_p,
        "max_tokens":         args.max_tokens,
        "repetition_penalty": args.repetition_penalty,
    }

    history: list[dict] = []

    # Load system prompt from limits/
    from limits.loader import load_system_prompt
    system_prompt = args.system or load_system_prompt()
    if system_prompt:
        cprint("Limits loaded from limits/", DIM)

    cprint("Ready — type a message and press Enter. /help for commands.\n", DIM)

    while True:
        try:
            user_input = prompt_input("You > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            cprint("Bye!", DIM)
            sys.exit(0)

        if not user_input:
            continue

        if user_input.startswith("/"):
            parts = user_input.split()
            cmd   = parts[0].lower()

            if cmd in ("/quit", "/exit"):
                cprint("Bye!", DIM)
                sys.exit(0)
            elif cmd == "/help":
                print_help()
            elif cmd == "/clear":
                os.system("clear")
                print_banner()
            elif cmd == "/settings":
                print_settings(settings)
            elif cmd == "/reset":
                history = []
                cprint("History cleared.\n", DIM)
            elif cmd == "/set":
                if len(parts) < 3:
                    cprint("Usage: /set <key> <value>\n", YELLOW)
                else:
                    key, val = parts[1], parts[2]
                    if key not in settings:
                        cprint(f"Unknown key '{key}'. Valid: {list(settings)}\n", YELLOW)
                    else:
                        try:
                            settings[key] = type(settings[key])(val)
                            cprint(f"  {key} = {settings[key]}\n", DIM)
                        except ValueError:
                            cprint(f"Invalid value '{val}'\n", YELLOW)
            else:
                cprint(f"Unknown command. Type /help.\n", YELLOW)
            continue

        prompt = (system_prompt + "\n\n" + user_input) if system_prompt else user_input

        response = stream_response(
            model, tokenizer, prompt,
            history=[] if args.no_history else history,
            settings=settings,
        )

        if not args.no_history:
            history.append({"user": user_input, "assistant": response})

        hr()
        print()


if __name__ == "__main__":
    main()
