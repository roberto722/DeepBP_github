#!/usr/bin/env python3
"""Converte il JSON di inferenza in un file Excel con metriche per immagine.

Input atteso (estratto):
{
  "summary": {...},
  "samples": [
    {
      "name": "...",
      "metrics": {"mse": ..., "mae": ..., "ssim": ..., "psnr": ...},
      "inference_speed": {"sample_seconds": ...}
    }
  ]
}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


COLUMNS = ["name", "mse", "mae", "ssim", "psnr", "sample_seconds"]


def _safe_get(dct: dict[str, Any], *keys: str) -> Any:
    current: Any = dct
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def extract_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    samples = payload.get("samples")
    if not isinstance(samples, list):
        raise ValueError("Il campo 'samples' deve essere una lista nel JSON di input.")

    rows: list[dict[str, Any]] = []
    for sample in samples:
        if not isinstance(sample, dict):
            continue

        rows.append(
            {
                "name": sample.get("name"),
                "mse": _safe_get(sample, "metrics", "mse"),
                "mae": _safe_get(sample, "metrics", "mae"),
                "ssim": _safe_get(sample, "metrics", "ssim"),
                "psnr": _safe_get(sample, "metrics", "psnr"),
                "sample_seconds": _safe_get(sample, "inference_speed", "sample_seconds"),
            }
        )

    return rows


def write_excel(rows: list[dict[str, Any]], output_path: Path) -> None:
    try:
        from openpyxl import Workbook
    except ImportError as exc:
        raise SystemExit(
            "Dipendenza mancante: openpyxl. Installa con 'pip install openpyxl'."
        ) from exc

    workbook = Workbook()
    worksheet = workbook.active
    worksheet.title = "metrics"

    worksheet.append(COLUMNS)
    for row in rows:
        worksheet.append([row.get(column) for column in COLUMNS])

    workbook.save(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estrae metriche per-sample da JSON e le salva in formato Excel (.xlsx)."
    )
    parser.add_argument("input_json", type=Path, help="Percorso del file JSON di input")
    parser.add_argument(
        "output_xlsx",
        type=Path,
        nargs="?",
        help="Percorso del file .xlsx di output (default: stesso nome dell'input)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path: Path = args.input_json
    output_path: Path = args.output_xlsx or input_path.with_suffix(".xlsx")

    with input_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    rows = extract_rows(payload)
    write_excel(rows, output_path)

    print(f"Creato file Excel: {output_path}")
    print(f"Righe esportate: {len(rows)}")


if __name__ == "__main__":
    main()