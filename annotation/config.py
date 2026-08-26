from __future__ import annotations

import os
from pathlib import Path


def load_env_file(path: str | Path) -> int:
    """Load simple KEY=VALUE entries without overriding process environment."""
    loaded = 0
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                raise ValueError(f"{path}:{line_number}: expected KEY=VALUE")
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if not key.isidentifier():
                raise ValueError(f"{path}:{line_number}: invalid environment variable name")
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                value = value[1:-1]
            if key not in os.environ:
                os.environ[key] = value
                loaded += 1
    return loaded
