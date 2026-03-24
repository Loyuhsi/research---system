#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


IN_DIR = Path("/workspace/in")
OUT_DIR = Path("/workspace/out")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tasks = sorted(IN_DIR.glob("*"))
    result = {
        "status": "ok",
        "tasks_seen": [task.name for task in tasks],
        "note": "RD-Agent sandbox scaffold is wired. Replace runner.py with the official RD-Agent entrypoint when Docker daemon and image build are available.",
    }
    (OUT_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
