import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from common import Standardizer, ensure_required_columns, load_table, save_json
from train_thumb_delta_mult_head import ThumbDeltaMultiHeadNet, build_column_names


def parse_args():
    p = argparse.ArgumentParser(description="Run thumb delta multihead inference.")
    p.add_argument("--ckpt", type=str, required=True, help="Path to best_thumb_delta_multihead.pt")
    p.add_argument("--out-csv", type=str, required=True)

    p.add_argument("--data", type=str, default="", help="Input csv/parquet path. Uses consecutive rows as q_prev/m_prev -> q_t.")
    p.add_argument("--q-values", type=float, nargs="+", default=None, help="Direct q_t values, length 4.")
    p.add_argument("--q-prev-values", type=float, nargs="+", default=None, help="Direct q_{t-1} values, length 4.")
    p.add_argument("--m-prev-values", type=float, nargs="+", default=None, help="Direct m_{t-1} values, length 6.")

    p.add_argument("--session-col", type=str, default="session_id")
    p.add_argument("--active-finger-col", type=str, default="active_finger")
    p.add_argument("--point-col", type=str, default="point_id")
    p.add_argument("--allow-non-consecutive", action="store_true")
    p.add_argument("--include-input-cols", action="store_true")
    p.add_argument("--device", type=str, default="")
    return p.parse_args()


def build_direct_arrays(q_values, q_prev_values, m_prev_values, q_dim: int, m_dim: int):
    if q_values is None or q_prev_values is None or m_prev_values is None:
        raise ValueError("Direct mode requires --q-values, --q-prev-values and --m-prev-values.")
    if len(q_values) != q_dim:
        raise ValueError(f"--q-values length must be {q_dim}, got {len(q_values)}")
    if len(q_prev_values) != q_dim:
        raise ValueError(f"--q-prev-values length must be {q_dim}, got {len(q_prev_values)}")
    if len(m_prev_values) != m_dim:
        raise ValueError(f"--m-prev-values length must be {m_dim}, got {len(m_prev_values)}")

    q_now = np.array([q_values], dtype=np.float32)
    q_prev = np.array([q_prev_values], dtype=np.float32)
    m_prev = np.array([m_prev_values], dtype=np.float32)
    x = np.concatenate([q_now, q_prev, m_prev, q_now - q_prev], axis=1)

    out_df = pd.DataFrame(
        {
            "session_id": ["direct_input"],
            "trial_id": ["direct_input_0"],
            "active_finger": ["thumb"],
            "point_id": [0],
        }
    )
    return x, m_prev, out_df


def build_sequence_arrays(df, args, q_cols, m_cols):
    group_cols = [c for c in [args.session_col, args.active_finger_col] if c in df.columns]
    sort_cols = group_cols + [args.point_col] if args.point_col in df.columns else group_cols
    if not sort_cols:
        raise ValueError("Cannot build sequence pairs: no session/active_finger/point_id columns found.")

    x_all = []
    m_prev_all = []
    out_rows = []
    sorted_df = df.sort_values(sort_cols).reset_index(drop=True)

    grouped = sorted_df.groupby(group_cols, sort=False) if group_cols else [(None, sorted_df)]
    for _, gdf in grouped:
        if len(gdf) < 2:
            continue
        q = gdf[q_cols].to_numpy(dtype=np.float32)
        m = gdf[m_cols].to_numpy(dtype=np.float32)
        prev = gdf.iloc[:-1].reset_index(drop=True)
        now = gdf.iloc[1:].reset_index(drop=True)

        keep = np.ones(len(now), dtype=bool)
        if not args.allow_non_consecutive and args.point_col in gdf.columns:
            p_prev = prev[args.point_col].to_numpy()
            p_now = now[args.point_col].to_numpy()
            keep = p_now == p_prev + 1

        q_prev = q[:-1][keep]
        q_now = q[1:][keep]
        m_prev = m[:-1][keep]
        now = now.iloc[np.where(keep)[0]].reset_index(drop=True)
        prev = prev.iloc[np.where(keep)[0]].reset_index(drop=True)

        x = np.concatenate([q_now, q_prev, m_prev, q_now - q_prev], axis=1)
        x_all.append(x)
        m_prev_all.append(m_prev)

        for i in range(len(now)):
            row = {}
            if args.include_input_cols:
                row.update(now.iloc[i].to_dict())
            else:
                for c in ["session_id", "trial_id", "active_finger", "point_id"]:
                    if c in now.columns:
                        row[c] = now.loc[i, c]
            if args.point_col in prev.columns:
                row["prev_point_id"] = prev.loc[i, args.point_col]
            for j, c in enumerate(q_cols):
                row[f"prev_{c}"] = q_prev[i, j]
            for j, c in enumerate(m_cols):
                row[f"prev_{c}"] = m_prev[i, j]
            out_rows.append(row)

    if not x_all:
        raise ValueError("No valid sequence pairs were built from input data.")

    return np.concatenate(x_all, axis=0), np.concatenate(m_prev_all, axis=0), pd.DataFrame(out_rows)


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = dict(ckpt["config"])
    cfg.setdefault("q_dim", 4)
    cfg.setdefault("m_dim", 6)
    cfg.setdefault("q_prefix", "q_")
    cfg.setdefault("m_prefix", "m_")
    cfg.setdefault("hidden_dim", 128)
    cfg.setdefault("head_hidden_dim", 64)
    cfg.setdefault("dropout", 0.0)

    q_cols, m_cols = build_column_names(int(cfg["q_dim"]), int(cfg["m_dim"]), cfg["q_prefix"], cfg["m_prefix"])

    direct_mode = args.q_values is not None or args.q_prev_values is not None or args.m_prev_values is not None
    data_mode = bool(args.data)
    if direct_mode == data_mode:
        raise ValueError("Use exactly one input mode: --data or direct mode (--q-values/--q-prev-values/--m-prev-values).")

    if direct_mode:
        x, m_prev, out_df = build_direct_arrays(args.q_values, args.q_prev_values, args.m_prev_values, len(q_cols), len(m_cols))
    else:
        df = load_table(args.data).copy().reset_index(drop=True)
        ensure_required_columns(df, q_cols, "q")
        ensure_required_columns(df, m_cols, "m")
        x, m_prev, out_df = build_sequence_arrays(df, args, q_cols, m_cols)

    x_scaler = Standardizer.from_dict(ckpt["x_scaler"])
    y_scaler = Standardizer.from_dict(ckpt["y_scaler"])
    x_norm = x_scaler.transform(x)

    ns = SimpleNamespace(**cfg)
    model = ThumbDeltaMultiHeadNet(
        input_dim=ns.q_dim * 3 + ns.m_dim,
        output_dim=ns.m_dim,
        hidden_dim=ns.hidden_dim,
        head_hidden_dim=ns.head_hidden_dim,
        dropout=ns.dropout,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    preds = []
    batch_size = 4096
    with torch.no_grad():
        for start in range(0, len(x_norm), batch_size):
            xb = torch.tensor(x_norm[start:start + batch_size], dtype=torch.float32, device=device)
            pred_delta_norm = model(xb).cpu().numpy()
            preds.append(y_scaler.inverse(pred_delta_norm))
    pred_delta = np.concatenate(preds, axis=0)
    pred_m = m_prev + pred_delta

    for i, col in enumerate(m_cols):
        out_df[f"pred_delta_{col}"] = pred_delta[:, i]
    for i, col in enumerate(m_cols):
        out_df[f"pred_{col}"] = pred_m[:, i]

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)

    save_json(
        str(out_csv.with_suffix(".summary.json")),
        {
            "ckpt": str(args.ckpt),
            "data": str(args.data) if args.data else None,
            "out_csv": str(out_csv),
            "num_rows": int(len(out_df)),
            "q_cols": q_cols,
            "m_cols": m_cols,
            "input_layout": ["q_t", "q_t_minus_1", "m_t_minus_1", "dq_t"],
            "pred_delta_cols": [f"pred_delta_{c}" for c in m_cols],
            "pred_m_cols": [f"pred_{c}" for c in m_cols],
            "device": device,
        },
    )

    print(f"wrote predictions: {out_csv}")
    print(f"rows: {len(out_df)}")
    print(f"predicted columns: {[f'pred_{c}' for c in m_cols]}")


if __name__ == "__main__":
    main()
