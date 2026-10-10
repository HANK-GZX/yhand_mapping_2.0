#!/usr/bin/env python3
import argparse
import json
import time
from datetime import datetime
from pathlib import Path


def _parse_args():
    p = argparse.ArgumentParser(description="Data collection live health monitor")
    p.add_argument("--status-file", required=True, help="Path to runtime status json")
    p.add_argument("--label", default="data-collection-health")
    p.add_argument("--refresh-sec", type=float, default=0.5)
    return p.parse_args()


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _fmt_pct(x):
    try:
        return f"{100.0 * float(x):6.2f}%"
    except Exception:
        return "   n/a"


def _render(payload: dict, label: str):
    print("\033[2J\033[H", end="")
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{label}] {now}")
    if not isinstance(payload, dict):
        print("waiting for status payload...")
        return

    hand = payload.get("hand", "")
    phase = payload.get("phase", "")
    reason = payload.get("reason", "")
    cur = payload.get("current", {}) if isinstance(payload.get("current"), dict) else {}
    g = payload.get("global", {}) if isinstance(payload.get("global"), dict) else {}
    z = payload.get("zero_calibration", {}) if isinstance(payload.get("zero_calibration"), dict) else {}

    print(f"hand={hand} phase={phase} reason={reason}")
    print(
        "current: "
        f"finger={cur.get('finger', '')} combo={cur.get('combo', '')} stage={cur.get('stage', '')}"
    )
    print(
        "global: "
        f"point={g.get('point', 0)} timeout={g.get('timeout', 0)} timeout_ratio={_fmt_pct(g.get('timeout_ratio', 0.0))}"
    )
    print(
        "zero_calibration: "
        f"state={z.get('state', '')} fail_streak={z.get('fail_streak', 0)}"
    )

    fingers = payload.get("fingers", {}) if isinstance(payload.get("fingers"), dict) else {}
    if not fingers:
        print("fingers: (none)")
        return

    print("fingers:")
    for finger in sorted(fingers.keys()):
        s = fingers.get(finger, {}) if isinstance(fingers.get(finger), dict) else {}
        print(
            f"  - {finger:6s} "
            f"point={int(s.get('point_id', 0)):5d}/{int(s.get('nominal_total', 0)):5d} "
            f"progress={_fmt_pct(s.get('progress_ratio', 0.0))} "
            f"unreached={int(s.get('unreached_count', 0)):4d} "
            f"pending_combo={int(s.get('combo_pending_count', 0)):3d} "
            f"none_max={int(s.get('combo_target_none_max', 0)):3d} "
            f"reopen={int(s.get('reopen_round', 0))}/{int(s.get('reopen_no_progress_round', 0))} "
            f"done={bool(s.get('done', False))}"
        )


def main():
    args = _parse_args()
    status_file = Path(args.status_file).expanduser()
    refresh = max(0.1, float(args.refresh_sec))

    last_mtime = None
    cached = None
    while True:
        if status_file.exists():
            try:
                mtime = status_file.stat().st_mtime
            except Exception:
                mtime = None
            if mtime is not None and mtime != last_mtime:
                payload = _read_json(status_file)
                if payload is not None:
                    cached = payload
                    last_mtime = mtime

        _render(cached, args.label)
        time.sleep(refresh)


if __name__ == "__main__":
    main()
