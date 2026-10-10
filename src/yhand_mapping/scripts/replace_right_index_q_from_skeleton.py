#!/usr/bin/env python3
"""Replace right-index q_0..q_3 using Glove Skeleton xlsx node 5..9 geometry.

Output q order:
  q_0 = Node6 local +X out of Node5 local XZ plane, positive toward Node5 +Y
  q_1 = Node6 local +X projected to Node5 local XZ plane, angle from Node5 +X,
        positive toward Node5 -Z
  q_2 = Node7 local +X projected to Node6 local XZ plane, angle from Node6 +X,
        positive toward Node6 -Z
  q_3 = Node8 local +X projected to Node7 local XZ plane, angle from Node7 +X,
        positive toward Node7 -Z
"""

import argparse
import csv
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZipFile


DEFAULT_SKELETON_DIR = (
    "/home/ubuntu/new_hand_data/20260428_215534_right_collection/"
    "right_hand/index/result/Glove Skeleton Data"
)

NODE_NAMES = {
    5: "Index Metacarpal",
    6: "Index Proximal",
    7: "Index Intermediate",
    8: "Index Distal",
    9: "Index Tip",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute right-index q_0..q_3 from skeleton node 5..9 and replace CSV q columns."
    )
    parser.add_argument("--input-csv", default="preprocessed_right_index_clean.csv")
    parser.add_argument("--output-csv", default="preprocessed_right_index_clean_skeleton_q.csv")
    parser.add_argument("--skeleton-dir", default=DEFAULT_SKELETON_DIR)
    parser.add_argument("--strict", action="store_true", help="fail if any skeleton xlsx is missing or invalid")
    return parser.parse_args()


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


def quat_to_rotvec(q):
    x, y, z, w = quat_normalize(q)
    if w < 0:
        x, y, z, w = -x, -y, -z, -w
    s = math.sqrt(x * x + y * y + z * z)
    if s < 1e-12:
        return [0.0, 0.0, 0.0]
    angle = 2.0 * math.atan2(s, w)
    return [angle * x / s, angle * y / s, angle * z / s]


def fold_angle_to_pm90(angle_deg: float) -> float:
    return ((angle_deg + 90.0) % 180.0) - 90.0


def compute_spread_from_node6_x_out_of_node5_xz(q5, q6) -> float:
    # local X = bone direction, local Y = flexion axis / Node5 XZ-plane normal,
    # local Z = spread axis. The spread angle is the signed amount that Node6
    # local +X deviates out of Node5 local XZ plane.
    #
    # R56 = inverse(R5) * R6
    # A = R56 * [1, 0, 0] = Node6 local +X in Node5 coordinates
    # spread = asin(A_y / ||A||), positive toward Node5 local +Y.
    q56 = quat_multiply(quat_inverse(q5), quat_normalize(q6))
    node6_x_in_node5 = quat_rotate(q56, [1.0, 0.0, 0.0])
    n = norm(node6_x_in_node5)
    if n < 1e-12:
        raise ValueError("zero-length Node6 local +X vector")
    ay = max(-1.0, min(1.0, node6_x_in_node5[1] / n))
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


def compute_index_q(data: list[float]) -> list[float]:
    n5 = node(data, 5)
    n6 = node(data, 6)
    n7 = node(data, 7)
    n8 = node(data, 8)

    q5 = n5[3:7]
    q6 = n6[3:7]
    q7 = n7[3:7]
    q8 = n8[3:7]

    spread = compute_spread_from_node6_x_out_of_node5_xz(q5, q6)
    mcp = compute_child_x_projected_xz_angle(q5, q6, "Node6", "Node5")
    pip = compute_child_x_projected_xz_angle(q6, q7, "Node7", "Node6")
    dip = compute_child_x_projected_xz_angle(q7, q8, "Node8", "Node7")
    return [spread, mcp, pip, dip]


def skeleton_path_for_trial(skeleton_dir: Path, trial_id: str) -> Path:
    match = re.search(r"(\d+)$", trial_id)
    if not match:
        raise ValueError(f"cannot parse numeric id from trial_id={trial_id!r}")
    return skeleton_dir / f"{int(match.group(1)):05d}.xlsx"


def main():
    args = parse_args()
    input_path = Path(args.input_csv)
    output_path = Path(args.output_csv)
    skeleton_dir = Path(args.skeleton_dir)

    with input_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {input_path}")
        fieldnames = reader.fieldnames
        rows = list(reader)

    required = ["trial_id", "q_0", "q_1", "q_2", "q_3"]
    missing = [c for c in required if c not in fieldnames]
    if missing:
        raise ValueError(f"input CSV missing columns: {missing}")

    replaced = 0
    failed = 0
    failures = []
    cache: dict[Path, list[float]] = {}

    for row in rows:
        xlsx_path = skeleton_path_for_trial(skeleton_dir, row["trial_id"])
        try:
            if xlsx_path not in cache:
                cache[xlsx_path] = read_xlsx_first_row(xlsx_path)
            q_values = compute_index_q(cache[xlsx_path])
        except Exception as exc:
            failed += 1
            if len(failures) < 20:
                failures.append(f"{row['trial_id']}: {exc}")
            if args.strict:
                raise
            continue

        for i, value in enumerate(q_values):
            row[f"q_{i}"] = f"{value:.10f}"
        replaced += 1

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"input rows: {len(rows)}")
    print(f"replaced rows: {replaced}")
    print(f"failed rows: {failed}")
    print(f"unique skeleton files read: {len(cache)}")
    print(f"wrote: {output_path}")
    if failures:
        print("first failures:")
        for item in failures:
            print(f"  {item}")


if __name__ == "__main__":
    main()
