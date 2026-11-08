"""Generate parser-compatible synthetic screenplay samples for local testing."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.demo_data import synthetic_screenplays


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=1, help="Number of samples to generate (1-50).")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("samples") / "generated",
        help="Output directory (default: samples/generated).",
    )
    args = parser.parse_args()
    samples = synthetic_screenplays(args.count)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    destinations = [args.output_dir / name for name, _ in samples]
    existing = [path for path in destinations if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing sample file(s): "
            + ", ".join(path.name for path in existing)
        )
    for path, (_, content) in zip(destinations, samples, strict=True):
        path.write_bytes(content)
    print(f"Generated {len(destinations)} synthetic sample(s) in {args.output_dir}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
