import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
from zipfile import ZipFile
import xml.etree.ElementTree as ET
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hand_motor_map import (
    FINGER_MOTOR_IDS_BY_HAND, LEGACY_TO_CURRENT_MOTOR_IDS, MAPPING_VERSION,
    WRIST_MOTOR_IDS_BY_HAND,
)


FINGERS = ["index", "little", "middle", "ring", "thumb"]
XLSX_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"

GLOVE_Q_SLICE = slice(8, 28)  # I:AB, zero-based [8:28]
GLOVE_Q_SLICES_BY_HAND = {"left": GLOVE_Q_SLICE, "right": slice(28, 48)}
MOTOR_CMD_IGNORED_ROWS = set(WRIST_MOTOR_IDS_BY_HAND["left"])
MOTOR_FEEDBACK_IGNORED_ROWS = {1, *(mid + 1 for mid in MOTOR_CMD_IGNORED_ROWS)}


def motor_row_layout(hand, motor_map):
    current_ids = list(FINGER_MOTOR_IDS_BY_HAND[hand])
    if motor_map == "legacy":
        inverse = {new: old for old, new in LEGACY_TO_CURRENT_MOTOR_IDS[hand].items()}
        source_ids = sorted(inverse[mid] for mid in current_ids)
        order = [source_ids.index(inverse[mid]) for mid in current_ids]
    else:
        source_ids = current_ids
        order = list(range(len(current_ids)))
    cmd_ignored = set(range(1, 29)) - set(source_ids)
    feedback_ignored = {1, *(mid + 1 for mid in cmd_ignored)}
    return cmd_ignored, feedback_ignored, order


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert raw collection result xlsx files into q_20/m_24 training CSV."
    )
    parser.add_argument("--hand", choices=["left", "right"], default="left")
    parser.add_argument("--motor-map", choices=["current", "legacy"], default="current",
                        help="Wiring used by the source recording; output always uses current channels.")
    parser.add_argument(
        "--collection",
        type=str,
        default="left_collection",
        help="Collection directory containing <hand>_hand/<finger>/result.",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default="preprocessed_raw.csv",
        help="Output CSV path for valid training samples.",
    )
    parser.add_argument(
        "--invalid-csv",
        type=str,
        default="preprocessed_raw_invalid.csv",
        help="Output CSV path for skipped/invalid sample records.",
    )
    parser.add_argument(
        "--summary-json",
        type=str,
        default="preprocessed_raw_summary.json",
        help="Output JSON path for preprocessing summary.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=14.0,
        help="Invalid sample threshold: any abs(motor_cmd - motor_feedback) greater than this value.",
    )
    return parser.parse_args()


def column_index(cell_ref: str) -> int:
    letters = "".join(ch for ch in cell_ref if ch.isalpha())
    index = 0
    for ch in letters:
        index = index * 26 + ord(ch.upper()) - ord("A") + 1
    return index - 1


def load_shared_strings(zf: ZipFile) -> List[str]:
    try:
        data = zf.read("xl/sharedStrings.xml")
    except KeyError:
        return []

    root = ET.fromstring(data)
    strings = []
    for item in root.findall(f"{XLSX_NS}si"):
        parts = [node.text or "" for node in item.iter(f"{XLSX_NS}t")]
        strings.append("".join(parts))
    return strings


def cell_text(cell: ET.Element, shared_strings: Sequence[str]) -> str:
    value = cell.find(f"{XLSX_NS}v")
    if value is None:
        inline = cell.find(f"{XLSX_NS}is")
        if inline is None:
            return ""
        return "".join(node.text or "" for node in inline.iter(f"{XLSX_NS}t"))

    text = value.text or ""
    if cell.attrib.get("t") == "s":
        return shared_strings[int(text)]
    return text


def read_xlsx_text_matrix(path: Path) -> List[List[str]]:
    with ZipFile(path) as zf:
        shared_strings = load_shared_strings(zf)
        worksheet_names = sorted(
            name
            for name in zf.namelist()
            if name.startswith("xl/worksheets/sheet") and name.endswith(".xml")
        )
        if not worksheet_names:
            raise ValueError(f"No worksheet found in {path}")

        root = ET.fromstring(zf.read(worksheet_names[0]))
        sheet_data = root.find(f"{XLSX_NS}sheetData")
        if sheet_data is None:
            return []

        matrix: List[List[str]] = []
        for row in sheet_data.findall(f"{XLSX_NS}row"):
            values: List[str] = []
            last_index = -1
            for cell in row.findall(f"{XLSX_NS}c"):
                current_index = column_index(cell.attrib.get("r", "A1"))
                while last_index + 1 < current_index:
                    values.append("")
                    last_index += 1

                text = cell_text(cell, shared_strings).strip()
                values.append(text)
                last_index = current_index
            matrix.append(values)
        return matrix


def to_float(value: str, path: Path, row: int, col: int) -> float:
    if value == "":
        return 0.0
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"Non-numeric value at {path} row={row} col={col}: {value!r}") from exc


def first_row_slice(path: Path, value_slice: slice, expected_dim: int) -> List[float]:
    matrix = read_xlsx_text_matrix(path)
    if not matrix:
        raise ValueError(f"Empty worksheet: {path}")
    values = [
        to_float(value, path, 1, col_index + 1)
        for col_index, value in enumerate(matrix[0][value_slice], start=value_slice.start)
    ]
    if len(values) != expected_dim:
        raise ValueError(f"Expected {expected_dim} values from {path}, got {len(values)}")
    return values


def first_column_without_rows(path: Path, ignored_rows: set[int], expected_dim: int) -> List[float]:
    matrix = read_xlsx_text_matrix(path)
    values = [
        to_float(row[0], path, row_index, 1)
        for row_index, row in enumerate(matrix, start=1)
        if row_index not in ignored_rows and row
    ]
    if len(values) != expected_dim:
        raise ValueError(f"Expected {expected_dim} first-column values from {path}, got {len(values)}")
    return values


def collect_sample_paths(finger_dir: Path) -> Dict[str, Dict[str, Path]]:
    result_dir = finger_dir / "result"
    if not result_dir.exists():
        result_dir = finger_dir / "results"

    folder_candidates = {
        "glove_data": ["glove_data", "glove data"],
        "motor_cmd": ["motor_cmd", "motor cmd"],
        "motor_feedback": ["motor_feedback", "motor feedback"],
    }

    samples: Dict[str, Dict[str, Path]] = {}
    for kind, names in folder_candidates.items():
        folder = next((result_dir / name for name in names if (result_dir / name).exists()), None)
        if folder is None:
            continue
        for path in sorted(folder.glob("*.xlsx")):
            samples.setdefault(path.stem, {})[kind] = path
    return samples


def max_abs_diff_on_nonzero_cmd(
    cmd_values: Sequence[float],
    feedback_values: Sequence[float],
) -> Tuple[float, int, int]:
    checked = [
        (index, abs(cmd - feedback))
        for index, (cmd, feedback) in enumerate(zip(cmd_values, feedback_values))
        if cmd != 0.0
    ]
    if not checked:
        return 0.0, -1, 0

    max_index, max_diff = max(checked, key=lambda item: item[1])
    return max_diff, max_index, len(checked)


def write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    collection_dir = Path(args.collection)
    hand_dir = collection_dir / f"{args.hand}_hand"
    if not hand_dir.exists():
        raise FileNotFoundError(f"Cannot find hand directory: {hand_dir}")
    cmd_ignored, feedback_ignored, motor_order = motor_row_layout(args.hand, args.motor_map)

    valid_rows: List[Dict] = []
    invalid_rows: List[Dict] = []
    summary = {
        "collection": str(collection_dir),
        "hand": args.hand,
        "source_motor_map": args.motor_map,
        "motor_mapping_version": MAPPING_VERSION,
        "motor_ids": list(FINGER_MOTOR_IDS_BY_HAND[args.hand]),
        "threshold": args.threshold,
        "valid_count": 0,
        "invalid_count": 0,
        "by_finger": {},
    }

    for finger in FINGERS:
        finger_dir = hand_dir / finger
        finger_summary = {"seen": 0, "valid": 0, "invalid": 0}
        summary["by_finger"][finger] = finger_summary
        if not finger_dir.exists():
            continue

        samples = collect_sample_paths(finger_dir)
        for point_id, paths in sorted(samples.items()):
            finger_summary["seen"] += 1
            missing = [name for name in ("glove_data", "motor_cmd", "motor_feedback") if name not in paths]
            invalid_base = {
                "session_id": collection_dir.name,
                "trial_id": f"{finger}_{point_id}",
                "active_finger": finger,
                "point_id": point_id,
            }

            if missing:
                invalid_rows.append({**invalid_base, "reason": "missing_file", "detail": "|".join(missing)})
                finger_summary["invalid"] += 1
                continue

            try:
                q_values = first_row_slice(paths["glove_data"], GLOVE_Q_SLICES_BY_HAND[args.hand], 20)
                cmd_values = first_column_without_rows(paths["motor_cmd"], cmd_ignored, 24)
                feedback_values = first_column_without_rows(
                    paths["motor_feedback"], feedback_ignored, 24
                )
                cmd_values = [cmd_values[i] for i in motor_order]
                feedback_values = [feedback_values[i] for i in motor_order]
            except Exception as exc:
                invalid_rows.append({**invalid_base, "reason": "parse_error", "detail": str(exc)})
                finger_summary["invalid"] += 1
                continue

            max_diff, max_diff_dim, checked_dims = max_abs_diff_on_nonzero_cmd(
                cmd_values,
                feedback_values,
            )
            if max_diff > args.threshold:
                invalid_rows.append(
                    {
                        **invalid_base,
                        "reason": "motor_mismatch",
                        "detail": (
                            f"dim={max_diff_dim},max_abs_diff={max_diff:.6f},"
                            f"checked_nonzero_cmd_dims={checked_dims}"
                        ),
                    }
                )
                finger_summary["invalid"] += 1
                continue

            row = {
                "session_id": collection_dir.name,
                "trial_id": f"{finger}_{point_id}",
                "active_finger": finger,
                "point_id": int(point_id),
            }
            row.update({f"q_{i}": value for i, value in enumerate(q_values)})
            row.update({f"m_{i}": value for i, value in enumerate(feedback_values)})
            valid_rows.append(row)
            finger_summary["valid"] += 1

    summary["valid_count"] = len(valid_rows)
    summary["invalid_count"] = len(invalid_rows)

    output_fields = (
        ["session_id", "trial_id", "active_finger", "point_id"]
        + [f"q_{i}" for i in range(20)]
        + [f"m_{i}" for i in range(24)]
    )
    invalid_fields = ["session_id", "trial_id", "active_finger", "point_id", "reason", "detail"]

    write_csv(Path(args.out_csv), output_fields, valid_rows)
    write_csv(Path(args.invalid_csv), invalid_fields, invalid_rows)
    Path(args.summary_json).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"valid samples: {len(valid_rows)}")
    print(f"invalid samples: {len(invalid_rows)}")
    print(f"wrote: {args.out_csv}")
    print(f"wrote: {args.invalid_csv}")
    print(f"wrote: {args.summary_json}")


if __name__ == "__main__":
    main()
