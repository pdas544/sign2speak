from __future__ import annotations

from pathlib import Path
import runpy


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    source_script = project_root / "train_lstm.py"
    runpy.run_path(str(source_script), run_name="__main__")


if __name__ == "__main__":
    main()
