import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from common import (
    MLP,
    Standardizer,
    build_active_onehot,
    concat_if_needed,
    ensure_required_columns,
    load_table,
    save_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inverse model inference from q to m.")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to best_inverse.pt")
    parser.add_argument("--data", type=str, required=True, help="Input csv/parquet path")
    parser.add_argument("--out-csv", type=str, required=True, help="Output CSV path")
    parser.add_argument(
        "--include-input-cols",
        action="store_true",
        help="Include original input columns in the output CSV",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Device override, e.g. cpu or cuda. Default: auto",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(args.ckpt)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(ckpt_path, map_location=device)
    config = ckpt["config"]
    q_cols = ckpt["q_cols"]
    m_cols = ckpt["m_cols"]
    use_active_finger = bool(ckpt.get("use_active_finger", False))
    active_finger_col = ckpt.get("active_finger_col")

    q_scaler = Standardizer.from_dict(ckpt["q_scaler"])
    m_scaler = Standardizer.from_dict(ckpt["m_scaler"])

    model = MLP(
        in_dim=int(config["q_dim"]) + (5 if use_active_finger else 0),
        out_dim=int(config["m_dim"]),
        hidden_dims=tuple(config["hidden_dims"]),
        dropout=float(config["dropout"]),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    df = load_table(args.data).copy().reset_index(drop=True)
    ensure_required_columns(df, q_cols, "q")

    q = df[q_cols].to_numpy(dtype=np.float32)
    q_norm = q_scaler.transform(q)

    active = build_active_onehot(df, active_finger_col if use_active_finger else None)

    with torch.no_grad():
        q_tensor = torch.tensor(q_norm, dtype=torch.float32, device=device)
        active_tensor = torch.tensor(active, dtype=torch.float32, device=device)
        x = concat_if_needed(q_tensor, active_tensor)
        pred_m_norm = model(x).cpu().numpy()

    pred_m = m_scaler.inverse(pred_m_norm)

    if args.include_input_cols:
        out_df = df.copy()
    else:
        keep_cols = [c for c in ["session_id", "trial_id", "active_finger", "point_id"] if c in df.columns]
        out_df = df[keep_cols].copy() if keep_cols else pd.DataFrame(index=df.index)

    for i, col in enumerate(m_cols):
        out_df[f"pred_{col}"] = pred_m[:, i]

    out_df.to_csv(out_csv, index=False)

    summary = {
        "ckpt": str(ckpt_path),
        "data": str(args.data),
        "out_csv": str(out_csv),
        "num_rows": int(len(out_df)),
        "q_cols": q_cols,
        "pred_m_cols": [f"pred_{col}" for col in m_cols],
        "use_active_finger": use_active_finger,
        "device": device,
    }
    save_json(str(out_csv.with_suffix(".summary.json")), summary)

    print(f"wrote predictions: {out_csv}")
    print(f"rows: {len(out_df)}")
    print(f"predicted columns: {[f'pred_{col}' for col in m_cols]}")


if __name__ == "__main__":
    main()
