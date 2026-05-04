"""Load limits/ files and assemble them into a system prompt."""
from pathlib import Path

_DIR = Path(__file__).parent


def _read_rules(filename: str) -> list[str]:
    path = _DIR / filename
    if not path.exists():
        return []
    lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            lines.append(line)
    return lines


def load_system_prompt() -> str:
    """Return the assembled system prompt from all limits/ files."""
    parts = []

    persona_path = _DIR / "persona.txt"
    if persona_path.exists():
        persona = persona_path.read_text(encoding="utf-8").strip()
        if persona:
            parts.append(persona)

    allowed = _read_rules("allowed.txt")
    if allowed:
        parts.append("You should:\n" + "\n".join(f"- {r}" for r in allowed))

    denied = _read_rules("denied.txt")
    if denied:
        parts.append("You must never:\n" + "\n".join(f"- {r}" for r in denied))

    return "\n\n".join(parts)
