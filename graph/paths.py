"""Shared utility for generating dated output file paths with numeric suffixes."""
from datetime import date
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent.parent / "output"


def dated_path(name: str, ext: str = "md") -> Path:
    """
    Return a Path like output/YYYY-MM-DD-N-<name>.<ext>.

    If output/YYYY-MM-DD-1-<name>.<ext> already exists, increments N until a
    free slot is found.
    """
    OUTPUT_DIR.mkdir(exist_ok=True)
    today = date.today().isoformat()
    n = 1
    while True:
        path = OUTPUT_DIR / f"{today}-{n}-{name}.{ext}"
        if not path.exists():
            return path
        n += 1
