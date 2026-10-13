#!/usr/bin/env python3
"""Count unique point pairs with close glove angles but different motor feedback.

Input is points.csv exported by plot_collection_combo_distribution.py. Values
are collection-window means. Angle distance is max(abs(q_a - q_b)), not a norm
that mixes the four joints. Motor distance uses the same max-component rule.
"""

import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from plot_collection_combo_distribution import combo_axes


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(points_csv, output, angle_threshold, motor_threshold, hand, finger, examples):
    with points_csv.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("Empty points CSV")
    motor_ids = sorted(int(key.split("_")[1]) for key in rows[0]
                       if key.startswith("motor_") and key.endswith("_pos"))
    if not motor_ids:
        raise ValueError("No motor position columns")
    ids = np.array([int(row["point_id"]) for row in rows])
    if len(np.unique(ids)) != len(ids):
        raise ValueError("Duplicate point IDs")
    order = np.argsort(ids)
    rows = [rows[i] for i in order]
    ids = ids[order]
    q = np.array([[float(row[f"glove_q{i}"]) for i in range(4)] for row in rows])
    m = np.array([[float(row[f"motor_{mid}_pos"]) for mid in motor_ids] for row in rows])
    if not np.isfinite(q).all() or not np.isfinite(m).all():
        raise ValueError("Input contains missing/non-finite angles or motor positions")
    combos = np.array([row["combo_id"] for row in rows])
    reached = np.array([row["reached"] == "1" and row["timeout"] == "0" for row in rows])
    all_pairs = cKDTree(q).query_pairs(angle_threshold, p=np.inf, output_type="ndarray")
    # query_pairs returns each unordered pair once. Sort by ID for reproducible export.
    all_pairs.sort(axis=1)
    a0, b0 = all_pairs.T
    delta0 = np.abs(m[a0] - m[b0])
    max_delta0 = delta0.max(axis=1)
    keep = max_delta0 > motor_threshold
    pairs = all_pairs[keep]
    a, b = pairs.T
    delta = delta0[keep]
    max_delta = max_delta0[keep]
    angle_delta = np.abs(q[a] - q[b])
    angle_max = angle_delta.max(axis=1)
    same = combos[a] == combos[b]
    both_reached = reached[a] & reached[b]
    summary = {
        "input_csv": str(points_csv.resolve()), "hand": hand, "finger": finger,
        "angle_rule": "all four abs(glove_q_i_a - glove_q_i_b) <= angle_threshold",
        "angle_threshold": angle_threshold, "motor_threshold": motor_threshold,
        "motor_rule": "at least one abs(motor_feedback_pos_a - motor_feedback_pos_b) > motor_threshold",
        "motor_ids": motor_ids, "point_count": len(rows),
        "total_possible_pairs": len(rows) * (len(rows) - 1) // 2,
        "angle_close_pair_count": len(all_pairs), "matching_pair_count": len(pairs),
        "involved_point_count": len(np.unique(pairs)),
        "same_combo_pair_count": int(same.sum()), "cross_combo_pair_count": int((~same).sum()),
        "both_reached_pair_count": int(both_reached.sum()),
        "both_reached_same_combo_pair_count": int((both_reached & same).sum()),
        "both_reached_cross_combo_pair_count": int((both_reached & ~same).sum()),
        "both_reached_angle_close_pair_count": int((reached[a0] & reached[b0]).sum()),
        "both_reached_point_count": int(reached.sum()),
        "both_reached_involved_point_count": len(np.unique(pairs[both_reached])),
        "max_motor_difference": float(max_delta.max()) if len(pairs) else None,
        "motor_difference_quantiles": dict(zip(["p05", "median", "p95"],
                                              np.percentile(max_delta, [5, 50, 95]).tolist())) if len(pairs) else {},
        "notes": ["Every unordered pair is counted once; pair counts are not point counts.",
                  "Per-motor counts overlap because one pair may exceed the threshold on multiple motors.",
                  "Values are result-window averages, not individual process frames.",
                  "Glove values come from glove_data, not skeleton data or motor commands."]}
    per_motor = []
    for col, mid in enumerate(motor_ids):
        mask = delta[:, col] > motor_threshold
        per_motor.append({"motor_id": mid, "matching_pair_count": int(mask.sum()),
                          "same_combo_pair_count": int((mask & same).sum()),
                          "cross_combo_pair_count": int((mask & ~same).sum()),
                          "both_reached_pair_count": int((mask & both_reached).sum()),
                          "max_position_difference": float(delta[:, col].max()) if len(delta) else None})
    combo_rows = []
    same_combo_axes_exceed = np.zeros(len(pairs), dtype=bool)
    same_combo_axes_max = np.zeros(len(pairs))
    for combo in sorted(set(combos)):
        close_mask = (combos[a0] == combo) & (combos[b0] == combo)
        mask = (combos[a] == combo) & (combos[b] == combo)
        active_ids = combo_axes(combo, hand, finger)
        active_cols = [motor_ids.index(mid) for mid in active_ids]
        axis_mask = (delta[:, active_cols] > motor_threshold).any(axis=1)
        same_combo_axes_exceed |= mask & axis_mask
        same_combo_axes_max[mask] = delta[mask][:, active_cols].max(axis=1)
        combo_rows.append({"combo_id": combo, "point_count": int((combos == combo).sum()),
                           "angle_close_pair_count": int(close_mask.sum()),
                           "matching_pair_count": int(mask.sum()),
                           "involved_point_count": len(np.unique(pairs[mask])),
                           "both_reached_pair_count": int((mask & both_reached).sum()),
                           "x_motor": active_ids[0], "y_motor": active_ids[1],
                           "xy_motor_matching_pair_count": int((mask & axis_mask).sum()),
                           "xy_motor_both_reached_pair_count": int((mask & axis_mask & both_reached).sum())})
    combo_pair_rows = []
    names = sorted(set(combos))
    for ci, first in enumerate(names):
        for second in names[ci:]:
            mask = ((combos[a] == first) & (combos[b] == second))
            close_mask = ((combos[a0] == first) & (combos[b0] == second))
            if first != second:
                mask |= ((combos[a] == second) & (combos[b] == first))
                close_mask |= ((combos[a0] == second) & (combos[b0] == first))
            combo_pair_rows.append({"combo_a": first, "combo_b": second,
                                    "angle_close_pair_count": int(close_mask.sum()),
                                    "matching_pair_count": int(mask.sum()),
                                    "both_reached_pair_count": int((mask & both_reached).sum())})
    summary["per_motor"] = per_motor
    summary["same_combo_breakdown"] = combo_rows
    summary["same_combo_xy_motor_pair_count"] = int(same_combo_axes_exceed.sum())
    summary["same_combo_xy_motor_both_reached_pair_count"] = int((same_combo_axes_exceed & both_reached).sum())
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "per_motor.csv", per_motor)
    write_csv(output / "same_combo_summary.csv", combo_rows)
    write_csv(output / "combo_pair_summary.csv", combo_pair_rows)

    # Keep all matches in a compressed table and detailed examples in a normal CSV.
    pair_header = ["point_id_a", "point_id_b", "combo_a", "combo_b", "both_reached",
                   "max_glove_angle_diff", "max_motor_position_diff"]
    pair_header += [f"glove_q{i}_diff" for i in range(4)]
    pair_header += [f"motor_{mid}_pos_diff" for mid in motor_ids]
    with gzip.open(output / "matching_pairs.csv.gz", "wt", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(pair_header)
        for i, (pa, pb) in enumerate(pairs):
            writer.writerow([ids[pa], ids[pb], combos[pa], combos[pb], int(both_reached[i]),
                             angle_max[i], max_delta[i], *angle_delta[i], *delta[i]])
    # Include extremes and closest-angle pairs, plus same-combo reached examples.
    selections = {}
    categories = {"largest_motor_difference": np.ones(len(pairs), dtype=bool),
                  "same_combo_both_reached": same & both_reached,
                  "same_combo_xy_both_reached": same_combo_axes_exceed & both_reached,
                  "closest_glove_angles": np.ones(len(pairs), dtype=bool)}
    for category, mask in categories.items():
        indices = np.flatnonzero(mask)
        if category == "closest_glove_angles":
            ranking = np.lexsort((-max_delta[indices], angle_max[indices]))
        else:
            ranking_values = same_combo_axes_max if category == "same_combo_xy_both_reached" else max_delta
            ranking = np.argsort(-ranking_values[indices], kind="stable")
        for i in indices[ranking[:examples]]:
            selections.setdefault(int(i), []).append(category)
    example_rows = []
    for i, categories in selections.items():
        pa, pb = pairs[i]
        record = dict(zip(pair_header, [int(ids[pa]), int(ids[pb]), combos[pa], combos[pb], int(both_reached[i]),
                                       angle_max[i], max_delta[i], *angle_delta[i], *delta[i]]))
        record["example_categories"] = "|".join(categories)
        for label, index in (("a", pa), ("b", pb)):
            record[f"reached_{label}"] = rows[index]["reached"]
            record[f"timeout_{label}"] = rows[index]["timeout"]
            for qi in range(4):
                record[f"glove_q{qi}_{label}"] = q[index, qi]
            for col, mid in enumerate(motor_ids):
                record[f"motor_{mid}_pos_{label}"] = m[index, col]
        example_rows.append(record)
    if example_rows:
        write_csv(output / "examples.csv", example_rows)
    degrees = np.bincount(pairs.ravel(), minlength=len(rows))
    reached_degrees = np.bincount(pairs[both_reached].ravel(), minlength=len(rows))
    same_degrees = np.bincount(pairs[same].ravel(), minlength=len(rows))
    point_rows = [{"point_id": int(ids[i]), "combo_id": combos[i], "matching_partner_count": int(degrees[i]),
                   "same_combo_partner_count": int(same_degrees[i]),
                   "both_reached_partner_count": int(reached_degrees[i])} for i in range(len(rows))]
    write_csv(output / "per_point.csv", point_rows)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("points_csv", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--angle-threshold", type=float, default=3.0)
    parser.add_argument("--motor-threshold", type=float, default=20.0)
    parser.add_argument("--hand", choices=("left", "right"), default="left")
    parser.add_argument("--finger", choices=("thumb", "index", "middle", "ring", "little"), default="index")
    parser.add_argument("--examples", type=int, default=100, help="Examples per ranking category")
    args = parser.parse_args()
    if not np.isfinite([args.angle_threshold, args.motor_threshold]).all() or args.angle_threshold < 0 or args.motor_threshold < 0 or args.examples < 1:
        parser.error("Thresholds must be finite and nonnegative; --examples must be positive")
    output = args.output_dir or args.points_csv.parent / "glove_motor_ambiguity"
    summary = analyze(args.points_csv, output, args.angle_threshold, args.motor_threshold,
                      args.hand, args.finger, args.examples)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("notes", "per_motor")}, ensure_ascii=False, indent=2))
    print(f"Output: {output.resolve()}")


if __name__ == "__main__":
    main()
