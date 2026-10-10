#!/usr/bin/env python3
"""Compatibility adapter for runtime right-index q solving from raw skeleton."""

from replace_right_non_thumb_q_from_skeleton import compute_non_thumb_q16


def compute_right_index_q(raw_skeleton_chunk):
    """Return dict keys q_0..q_3 from a 176D one-hand raw-skeleton chunk."""
    q16 = compute_non_thumb_q16(raw_skeleton_chunk)
    required_keys = ("index_q_0", "index_q_1", "index_q_2", "index_q_3")
    missing = [k for k in required_keys if k not in q16]
    if missing:
        raise KeyError(f"Missing index q keys from compute_non_thumb_q16: {missing}")
    return {
        "q_0": float(q16["index_q_0"]),
        "q_1": float(q16["index_q_1"]),
        "q_2": float(q16["index_q_2"]),
        "q_3": float(q16["index_q_3"]),
    }
