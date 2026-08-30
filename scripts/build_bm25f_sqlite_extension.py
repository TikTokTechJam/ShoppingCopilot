"""Build the small native SQLite FTS5 BM25F extension."""

from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
import sysconfig
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "native" / "bm25f.c"


def _default_output() -> Path:
    system = platform.system().casefold()
    if system == "darwin":
        return ROOT / "native" / "build" / "bm25f.dylib"
    if system == "windows":
        return ROOT / "native" / "build" / "bm25f.dll"
    return ROOT / "native" / "build" / "bm25f.so"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=_default_output())
    args = parser.parse_args()

    compiler = shutil.which("clang") or shutil.which("cc")
    if compiler is None:
        raise SystemExit("A C compiler (clang or cc) is required to build bm25f")
    if not SOURCE.is_file():
        raise SystemExit(f"Native source is missing: {SOURCE}")

    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [compiler, "-O3", "-fPIC", "-Wall", "-Wextra"]
    # On macOS, adding the SDK sysroot can cause clang to select the SDK's
    # sqlite3ext.h.  Apple's header defines SQLITE_OMIT_LOAD_EXTENSION, which
    # disables SQLITE_EXTENSION_INIT1/2 and leaves sqlite3_* calls as unsafe
    # dynamic symbols.  Use the header shipped with the active Python SQLite
    # runtime instead; its extension API matches the connection that loads it.
    include_candidates = (
        Path(sysconfig.get_path("include")),
        Path(sysconfig.get_config_var("prefix") or "") / "include",
    )
    for sqlite_include in include_candidates:
        if (sqlite_include / "sqlite3ext.h").is_file():
            command.extend(["-I", str(sqlite_include)])
            break
    if platform.system() == "Darwin":
        sdk = subprocess.check_output(
            ["xcrun", "--show-sdk-path"], text=True
        ).strip()
        command.extend(["-dynamiclib", "-undefined", "dynamic_lookup", "-isysroot", sdk])
    elif platform.system() == "Windows":
        raise SystemExit("The portable build command currently targets macOS/Linux")
    else:
        command.append("-shared")
    command.extend([str(SOURCE), "-o", str(output)])
    print("[bm25f] building native extension:", " ".join(command), flush=True)
    subprocess.run(command, check=True, cwd=ROOT)
    print(f"[bm25f] wrote {output}", flush=True)


if __name__ == "__main__":
    main()
