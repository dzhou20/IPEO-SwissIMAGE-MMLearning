#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable

import rasterio
from rasterio.errors import RasterioIOError


def _iter_paths(data_dir: Path, csv_path: Path | None) -> Iterable[Path]:
    if csv_path is None:
        yield from sorted(data_dir.glob("*.tif"))
        return

    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        if "id" not in reader.fieldnames:
            raise ValueError("CSV must contain an 'id' column.")
        for row in reader:
            tif = data_dir / f"{row['id']}.tif"
            yield tif


def _check_tif(path: Path) -> str | None:
    if not path.exists():
        return "missing"
    try:
        with rasterio.open(path) as src:
            _ = src.read()
    except RasterioIOError:
        return "rasterio_error"
    except Exception:
        return "other_error"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Check .tif integrity by reading files.")
    parser.add_argument("--data-dir", default="data", help="Directory with .tif files.")
    parser.add_argument("--csv", default=None, help="Optional CSV with an 'id' column.")
    parser.add_argument("--output", default=None, help="Optional output file for bad paths.")
    parser.add_argument("--max", type=int, default=None, help="Max number of files to check.")
    parser.add_argument("--print-every", type=int, default=500, help="Progress print interval.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    csv_path = Path(args.csv) if args.csv else None
    out_path = Path(args.output) if args.output else None

    total = 0
    bad = []
    for path in _iter_paths(data_dir, csv_path):
        total += 1
        if args.max is not None and total > args.max:
            break
        status = _check_tif(path)
        if status is not None:
            bad.append((str(path), status))
        if total % args.print_every == 0:
            print(f"[info] checked={total}, bad={len(bad)}")

    print(f"[info] done: checked={total}, bad={len(bad)}")
    if bad:
        print("[info] sample bad files:")
        for path, status in bad[:10]:
            print(f"  {path} :: {status}")

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for path, status in bad:
                f.write(f"{path}\t{status}\n")
        print(f"[info] wrote bad list to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
