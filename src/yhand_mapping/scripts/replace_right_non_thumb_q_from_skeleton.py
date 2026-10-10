#!/usr/bin/env python3
"""Replace q_0..q_3 for right index/middle/ring/little from Glove Skeleton xlsx.

This script processes the four non-thumb right-hand fingers. By default it reads:
  preprocessed_right_index_clean.csv
  preprocessed_right_middle_clean.csv
  preprocessed_right_ring_clean.csv
  preprocessed_right_little_clean.csv

and writes:
  preprocessed_right_index_clean_skeleton_q.csv
  preprocessed_right_middle_clean_skeleton_q.csv
  preprocessed_right_ring_clean_skeleton_q.csv
  preprocessed_right_little_clean_skeleton_q.csv

It also writes one stacked 16-angle table by default:
  preprocessed_right_non_thumb_clean_skeleton_q16.csv

Each q16 row corresponds to one source CSV row and its matching skeleton xlsx.
The 16 output angle columns are:
  index_q_0, index_q_1, index_q_2, index_q_3,
  middle_q_0, middle_q_1, middle_q_2, middle_q_3,
  ring_q_0, ring_q_1, ring_q_2, ring_q_3,
  little_q_0, little_q_1, little_q_2, little_q_3.

Output joint angle sequence for every finger:
  q_0 = proximal node local +X out of metacarpal node local XZ plane,
        positive toward metacarpal node +Y
  q_1 = proximal node local +X projected to metacarpal node local XZ plane,
        angle from metacarpal node +X, positive toward metacarpal node -Z
  q_2 = intermediate node local +X projected to proximal node local XZ plane,
        angle from proximal node +X, positive toward proximal node -Z
  q_3 = distal node local +X projected to intermediate node local XZ plane,
        angle from intermediate node +X, positive toward intermediate node -Z

Node chains:
  index:  Node5  -> Node6  -> Node7  -> Node8  -> Node9
  middle: Node10 -> Node11 -> Node12 -> Node13 -> Node14
  ring:   Node15 -> Node16 -> Node17 -> Node18 -> Node19
  little: Node20 -> Node21 -> Node22 -> Node23 -> Node24
"""

import argparse
import csv
import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile


DEFAULT_COLLECTION_ROOT = Path("/home/ubuntu/new_hand_data/20260428_215534_right_collection")


@dataclass(frozen=True)
class FingerSpec:
    name: str
    nodes: tuple[int, int, int, int, int]
    node_names: tuple[str, str, str, str, str]


FINGER_SPECS = {
    "index": FingerSpec(
        name="index",
        nodes=(5, 6, 7, 8, 9),
        node_names=("Index Metacarpal", "Index Proximal", "Index Intermediate", "Index Distal", "Index Tip"),
    ),
    "middle": FingerSpec(
        name="middle",
        nodes=(10, 11, 12, 13, 14),
        node_names=("Middle Metacarpal", "Middle Proximal", "Middle Intermediate", "Middle Distal", "Middle Tip"),
    ),
    "ring": FingerSpec(
        name="ring",
        nodes=(15, 16, 17, 18, 19),
        node_names=("Ring Metacarpal", "Ring Proximal", "Ring Intermediate", "Ring Distal", "Ring Tip"),
    ),
    "little": FingerSpec(
        name="little",
        nodes=(20, 21, 22, 23, 24),
        node_names=("Little Metacarpal", "Little Proximal", "Little Intermediate", "Little Distal", "Little Tip"),
    ),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute right-hand non-thumb q_0..q_3 from skeleton nodes and replace CSV q columns."
    )
    parser.add_argument(
        "--fingers",
        default="index,middle,ring,little",
        help="comma-separated subset of: index,middle,ring,little",
    )
    parser.add_argument(
        "--input-template",
        default="preprocessed_right_{finger}_clean.csv",
        help="input CSV template; {finger} is replaced with each finger name",
    )
    parser.add_argument(
        "--output-template",
        default="preprocessed_right_{finger}_clean_skeleton_q.csv",
        help="output CSV template; {finger} is replaced with each finger name",
    )
    parser.add_argument(
        "--combined-output",
        default="preprocessed_right_non_thumb_clean_skeleton_q16.csv",
        help=(
            "optional stacked CSV with 16 non-thumb q columns computed from each source skeleton frame; "
            "set to empty string to disable"
        ),
    )
    parser.add_argument(
        "--skeleton-root",
        default=str(DEFAULT_COLLECTION_ROOT),
        help="collection root containing right_hand/<finger>/result/Glove Skeleton Data",
    )
    parser.add_argument("--strict", action="store_true", help="fail if any skeleton xlsx is missing or invalid")
    return parser.parse_args()


def parse_fingers(text: str) -> list[str]:
    fingers = [x.strip() for x in text.split(",") if x.strip()]
    unknown = [x for x in fingers if x not in FINGER_SPECS]
    if unknown:
        raise ValueError(f"unknown fingers: {unknown}; expected one or more of {sorted(FINGER_SPECS)}")
    return fingers


def skeleton_dir_for_finger(skeleton_root: Path, finger: str) -> Path:
    return skeleton_root / "right_hand" / finger / "result" / "Glove Skeleton Data"


def col_to_idx(cell_ref: str) -> int:
    letters = "".join(ch for ch in cell_ref if ch.isalpha())
    out = 0
    for ch in letters:
        out = out * 26 + ord(ch.upper()) - ord("A") + 1
    return out - 1


def read_shared_strings(zf: ZipFile) -> list[str]:
    try:
        data = zf.read("xl/sharedStrings.xml")
    except KeyError:
        return []
    ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    root = ET.fromstring(data)
    strings = []
    for si in root.findall("a:si", ns):
        strings.append("".join(t.text or "" for t in si.findall(".//a:t", ns)))
    return strings


def read_xlsx_first_row(path: Path) -> list[float]:
    ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with ZipFile(path) as zf:
        shared = read_shared_strings(zf)
        root = ET.fromstring(zf.read("xl/worksheets/sheet1.xml"))
        first_row = root.find(".//a:sheetData/a:row", ns)
        if first_row is None:
            raise ValueError(f"no rows in xlsx: {path}")

        values: list[str] = []
        for cell in first_row.findall("a:c", ns):
            idx = col_to_idx(cell.attrib.get("r", "A1"))
            while len(values) <= idx:
                values.append("")
            v = cell.find("a:v", ns)
            value = "" if v is None or v.text is None else v.text
            if cell.attrib.get("t") == "s" and value:
                value = shared[int(value)]
            values[idx] = value

    if len(values) != 176:
        raise ValueError(f"expected 176 values, got {len(values)} in {path}")
    return [float(v) for v in values]


def node(data: list[float], node_idx: int) -> list[float]:
    start = 1 + node_idx * 7
    return data[start : start + 7]


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def norm(v):
    return math.sqrt(dot(v, v))


def quat_normalize(q):
    n = norm(q)
    if n < 1e-12:
        raise ValueError("zero-length quaternion")
    return [x / n for x in q]


def quat_inverse(q):
    x, y, z, w = quat_normalize(q)
    return [-x, -y, -z, w]


def quat_multiply(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return [
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ]


def quat_rotate(q, v):
    qn = quat_normalize(q)
    vq = [v[0], v[1], v[2], 0.0]
    return quat_multiply(quat_multiply(qn, vq), quat_inverse(qn))[:3]


def compute_spread_from_child_x_out_of_parent_xz(parent_q, child_q, child_label: str) -> float:
    # R_parent_child = inverse(R_parent) * R_child
    # A = R_parent_child * [1, 0, 0] = child local +X in parent coordinates.
    # spread = asin(A_y / ||A||), positive toward parent local +Y.
    relative_q = quat_multiply(quat_inverse(parent_q), quat_normalize(child_q))
    child_x_in_parent = quat_rotate(relative_q, [1.0, 0.0, 0.0])
    n = norm(child_x_in_parent)
    if n < 1e-12:
        raise ValueError(f"zero-length {child_label} local +X vector")
    ay = max(-1.0, min(1.0, child_x_in_parent[1] / n))
    return math.degrees(math.asin(ay))


def compute_child_x_projected_xz_angle(parent_q, child_q, child_label: str, parent_label: str) -> float:
    # R_parent_child = inverse(R_parent) * R_child
    # A = R_parent_child * [1, 0, 0] = child local +X in parent coordinates.
    # Project A onto parent local XZ plane, then measure the signed angle
    # from parent local +X toward parent local -Z.
    #
    # q = atan2(-A_z, A_x)
    relative_q = quat_multiply(quat_inverse(parent_q), quat_normalize(child_q))
    child_x_in_parent = quat_rotate(relative_q, [1.0, 0.0, 0.0])
    ax = child_x_in_parent[0]
    az = child_x_in_parent[2]
    projected_norm = math.sqrt(ax * ax + az * az)
    if projected_norm < 1e-12:
        raise ValueError(
            f"{child_label} local +X projection on {parent_label} local XZ plane is zero-length"
        )
    return math.degrees(math.atan2(-az, ax))


def compute_finger_q(data: list[float], spec: FingerSpec) -> list[float]:
    metacarpal_idx, proximal_idx, intermediate_idx, distal_idx, _tip_idx = spec.nodes
    metacarpal_name, proximal_name, intermediate_name, distal_name, _tip_name = spec.node_names

    metacarpal_q = node(data, metacarpal_idx)[3:7]
    proximal_q = node(data, proximal_idx)[3:7]
    intermediate_q = node(data, intermediate_idx)[3:7]
    distal_q = node(data, distal_idx)[3:7]

    spread = compute_spread_from_child_x_out_of_parent_xz(
        metacarpal_q,
        proximal_q,
        proximal_name,
    )
    mcp = compute_child_x_projected_xz_angle(
        metacarpal_q,
        proximal_q,
        proximal_name,
        metacarpal_name,
    )
    pip = compute_child_x_projected_xz_angle(
        proximal_q,
        intermediate_q,
        intermediate_name,
        proximal_name,
    )
    dip = compute_child_x_projected_xz_angle(
        intermediate_q,
        distal_q,
        distal_name,
        intermediate_name,
    )
    return [spread, mcp, pip, dip]


def compute_non_thumb_q16(data: list[float]) -> dict[str, float]:
    out = {}
    for finger, spec in FINGER_SPECS.items():
        values = compute_finger_q(data, spec)
        for i, value in enumerate(values):
            out[f"{finger}_q_{i}"] = value
    return out


def skeleton_path_for_trial(skeleton_dir: Path, trial_id: str) -> Path:
    match = re.search(r"(\d+)$", trial_id)
    if not match:
        raise ValueError(f"cannot parse numeric id from trial_id={trial_id!r}")
    return skeleton_dir / f"{int(match.group(1)):05d}.xlsx"


def process_finger(
    spec: FingerSpec,
    input_path: Path,
    output_path: Path,
    skeleton_dir: Path,
    strict: bool,
) -> dict:
    with input_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {input_path}")
        fieldnames = reader.fieldnames
        rows = list(reader)

    required = ["trial_id", "q_0", "q_1", "q_2", "q_3"]
    missing = [c for c in required if c not in fieldnames]
    if missing:
        raise ValueError(f"{input_path} missing columns: {missing}")

    replaced = 0
    failed = 0
    failures = []
    cache: dict[Path, list[float]] = {}

    for row in rows:
        xlsx_path = skeleton_path_for_trial(skeleton_dir, row["trial_id"])
        try:
            if xlsx_path not in cache:
                cache[xlsx_path] = read_xlsx_first_row(xlsx_path)
            q_values = compute_finger_q(cache[xlsx_path], spec)
        except Exception as exc:
            failed += 1
            if len(failures) < 20:
                failures.append(f"{row['trial_id']}: {exc}")
            if strict:
                raise
            continue

        for i, value in enumerate(q_values):
            row[f"q_{i}"] = f"{value:.10f}"
        replaced += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return {
        "finger": spec.name,
        "input_rows": len(rows),
        "replaced_rows": replaced,
        "failed_rows": failed,
        "unique_skeleton_files_read": len(cache),
        "input": str(input_path),
        "output": str(output_path),
        "skeleton_dir": str(skeleton_dir),
        "failures": failures,
    }


def process_combined_q16(
    fingers: list[str],
    input_template: str,
    output_path: Path,
    skeleton_root: Path,
    strict: bool,
) -> dict:
    q16_cols = [f"{finger}_q_{i}" for finger in FINGER_SPECS for i in range(4)]
    metadata_cols = ["source_finger", "session_id", "trial_id", "active_finger", "point_id"]
    motor_cols = [f"m_{i}" for i in range(5)]
    fieldnames = metadata_cols + q16_cols + motor_cols

    out_rows = []
    failed = 0
    failures = []
    cache: dict[Path, list[float]] = {}

    for source_finger in fingers:
        input_path = Path(input_template.format(finger=source_finger))
        skeleton_dir = skeleton_dir_for_finger(skeleton_root, source_finger)
        with input_path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError(f"CSV has no header: {input_path}")
            rows = list(reader)

        for row in rows:
            out = {"source_finger": source_finger}
            for col in ["session_id", "trial_id", "active_finger", "point_id"]:
                out[col] = row.get(col, "")
            for col in motor_cols:
                out[col] = row.get(col, "")

            xlsx_path = skeleton_path_for_trial(skeleton_dir, row["trial_id"])
            try:
                if xlsx_path not in cache:
                    cache[xlsx_path] = read_xlsx_first_row(xlsx_path)
                q16 = compute_non_thumb_q16(cache[xlsx_path])
            except Exception as exc:
                failed += 1
                if len(failures) < 20:
                    failures.append(f"{source_finger}:{row['trial_id']}: {exc}")
                if strict:
                    raise
                q16 = {}

            for col in q16_cols:
                value = q16.get(col)
                out[col] = "" if value is None else f"{value:.10f}"
            out_rows.append(out)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    return {
        "rows": len(out_rows),
        "failed_rows": failed,
        "unique_skeleton_files_read": len(cache),
        "output": str(output_path),
        "failures": failures,
    }


def print_q_order():
    print("Output q order for each non-thumb finger:")
    print("  q_0: proximal +X out of metacarpal XZ plane, positive toward metacarpal +Y")
    print("  q_1: proximal +X projected on metacarpal XZ, from metacarpal +X toward -Z")
    print("  q_2: intermediate +X projected on proximal XZ, from proximal +X toward -Z")
    print("  q_3: distal +X projected on intermediate XZ, from intermediate +X toward -Z")
    print("Combined q16 columns:")
    print("  index_q_0..index_q_3, middle_q_0..middle_q_3, ring_q_0..ring_q_3, little_q_0..little_q_3")


def main():
    args = parse_args()
    fingers = parse_fingers(args.fingers)
    skeleton_root = Path(args.skeleton_root)

    print_q_order()
    for finger in fingers:
        spec = FINGER_SPECS[finger]
        input_path = Path(args.input_template.format(finger=finger))
        output_path = Path(args.output_template.format(finger=finger))
        skeleton_dir = skeleton_dir_for_finger(skeleton_root, finger)
        result = process_finger(spec, input_path, output_path, skeleton_dir, args.strict)

        print()
        print(f"[{finger}] input rows: {result['input_rows']}")
        print(f"[{finger}] replaced rows: {result['replaced_rows']}")
        print(f"[{finger}] failed rows: {result['failed_rows']}")
        print(f"[{finger}] unique skeleton files read: {result['unique_skeleton_files_read']}")
        print(f"[{finger}] wrote: {result['output']}")
        if result["failures"]:
            print(f"[{finger}] first failures:")
            for item in result["failures"]:
                print(f"  {item}")

    if args.combined_output:
        combined = process_combined_q16(
            fingers=fingers,
            input_template=args.input_template,
            output_path=Path(args.combined_output),
            skeleton_root=skeleton_root,
            strict=args.strict,
        )
        print()
        print("[q16] rows:", combined["rows"])
        print("[q16] failed rows:", combined["failed_rows"])
        print("[q16] unique skeleton files read:", combined["unique_skeleton_files_read"])
        print("[q16] wrote:", combined["output"])
        if combined["failures"]:
            print("[q16] first failures:")
            for item in combined["failures"]:
                print(f"  {item}")


if __name__ == "__main__":
    main()
