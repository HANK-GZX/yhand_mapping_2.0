import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from common import (
    MLP,
    Standardizer,
    StaticMappingDataset,
    apply_split_info,
    build_active_onehot,
    close_wandb,
    concat_if_needed,
    ensure_required_columns,
    init_wandb,
    load_table,
    log_wandb_metrics,
    make_split_info,
    regression_metrics_raw,
    resolve_existing_column,
    save_json,
    set_seed,
)


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--data", type=str, required=True, help="csv/parquet path")
    p.add_argument("--out-dir", type=str, required=True)

    p.add_argument("--q-dim", type=int, default=4)
    p.add_argument("--m-dim", type=int, default=6)
    p.add_argument("--q-prefix", type=str, default="q_")
    p.add_argument("--m-prefix", type=str, default="m_")

    p.add_argument("--session-col", type=str, default="session_id")
    p.add_argument("--trial-col", type=str, default="trial_id")
    p.add_argument("--split-col", type=str, default="", help="if empty, auto use session_col or trial_col")
    p.add_argument("--active-finger-col", type=str, default="active_finger")

    p.add_argument("--use-active-finger", action="store_true")

    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--test-ratio", type=float, default=0.15)

    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--hidden-dims", type=int, nargs="+", default=[256, 256, 128])
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--wandb", action="store_true", help="enable Weights & Biases logging")
    p.add_argument("--wandb-project", type=str, default="mapping")
    p.add_argument("--wandb-run-name", type=str, default="")
    p.add_argument("--wandb-mode", type=str, default="", choices=["", "online", "offline", "disabled"])

    return p.parse_args()


def build_column_names(q_dim: int, m_dim: int, q_prefix: str, m_prefix: str) -> Tuple[List[str], List[str]]:
    q_cols = [f"{q_prefix}{i}" for i in range(q_dim)]
    m_cols = [f"{m_prefix}{i}" for i in range(m_dim)]
    return q_cols, m_cols


def make_loaders(
    train_df,
    val_df,
    test_df,
    q_cols_delta: List[str],
    m_cols_delta: List[str],
    use_active_finger: bool,
    active_finger_col: str,
    batch_size: int,
):
    q_train = train_df[q_cols_delta].to_numpy(dtype=np.float32)
    q_val = val_df[q_cols_delta].to_numpy(dtype=np.float32)
    q_test = test_df[q_cols_delta].to_numpy(dtype=np.float32)

    m_train = train_df[m_cols_delta].to_numpy(dtype=np.float32)
    m_val = val_df[m_cols_delta].to_numpy(dtype=np.float32)
    m_test = test_df[m_cols_delta].to_numpy(dtype=np.float32)

    q_scaler = Standardizer.fit(q_train)
    m_scaler = Standardizer.fit(m_train)

    q_train_n = q_scaler.transform(q_train)
    q_val_n = q_scaler.transform(q_val)
    q_test_n = q_scaler.transform(q_test)

    m_train_n = m_scaler.transform(m_train)
    m_val_n = m_scaler.transform(m_val)
    m_test_n = m_scaler.transform(m_test)

    active_train = build_active_onehot(train_df, active_finger_col if use_active_finger else None)
    active_val = build_active_onehot(val_df, active_finger_col if use_active_finger else None)
    active_test = build_active_onehot(test_df, active_finger_col if use_active_finger else None)

    train_ds = StaticMappingDataset(q_train_n, m_train_n, active_train)
    val_ds = StaticMappingDataset(q_val_n, m_val_n, active_val)
    test_ds = StaticMappingDataset(q_test_n, m_test_n, active_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    return train_loader, val_loader, test_loader, q_scaler, m_scaler


def evaluate_forward(model, loader, q_scaler, device):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            m = batch["m"].to(device)
            active = batch["active"].to(device)
            x = concat_if_needed(m, active)
            pred_q = model(x)
            preds.append(pred_q.cpu())
            targets.append(batch["q"])

    pred = torch.cat(preds, dim=0)
    target = torch.cat(targets, dim=0)
    return regression_metrics_raw(pred, target, q_scaler)


def main():
    args = parse_args()
    set_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    df = load_table(args.data).copy().reset_index(drop=True)
    df["row_id"] = np.arange(len(df))

    q_cols, m_cols = build_column_names(args.q_dim, args.m_dim, args.q_prefix, args.m_prefix)
    ensure_required_columns(df, q_cols, "q")
    ensure_required_columns(df, m_cols, "m")

    session_col = resolve_existing_column(df, [args.session_col, args.trial_col], "session_col")
    split_col = resolve_existing_column(df, [args.split_col if args.split_col else None, args.session_col, args.trial_col], "split_col")

    active_finger_col = args.active_finger_col if args.active_finger_col in df.columns else None
    q_cols_delta = q_cols
    m_cols_delta = m_cols

    split_info = make_split_info(
        df=df,
        split_col=split_col,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    train_df, val_df, test_df = apply_split_info(df, split_info)

    train_loader, val_loader, test_loader, q_scaler, m_scaler = make_loaders(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        q_cols_delta=q_cols_delta,
        m_cols_delta=m_cols_delta,
        use_active_finger=args.use_active_finger,
        active_finger_col=active_finger_col,
        batch_size=args.batch_size,
    )

    in_dim = args.m_dim + (5 if args.use_active_finger else 0)
    model = MLP(
        in_dim=in_dim,
        out_dim=args.q_dim,
        hidden_dims=tuple(args.hidden_dims),
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.HuberLoss(delta=1.0)

    best_val = float("inf")
    bad_epochs = 0

    history = []
    wandb_run = init_wandb(args, out_dir, "forward_thumb")

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        n_train = 0

        for batch in train_loader:
            q = batch["q"].to(device)
            m = batch["m"].to(device)
            active = batch["active"].to(device)

            x = concat_if_needed(m, active)
            pred_q = model(x)
            loss = criterion(pred_q, q)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bs = q.size(0)
            train_loss += loss.item() * bs
            n_train += bs

        train_loss /= max(1, n_train)

        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                q = batch["q"].to(device)
                m = batch["m"].to(device)
                active = batch["active"].to(device)

                x = concat_if_needed(m, active)
                pred_q = model(x)
                loss = criterion(pred_q, q)

                bs = q.size(0)
                val_loss += loss.item() * bs
                n_val += bs

        val_loss /= max(1, n_val)

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        log_wandb_metrics(
            wandb_run,
            {
                "loss/train": float(train_loss),
                "loss/val": float(val_loss),
                "lr": float(optimizer.param_groups[0]["lr"]),
            },
            epoch,
        )
        print(f"[Forward Thumb] epoch={epoch:03d} train={train_loss:.6f} val={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            bad_epochs = 0
            ckpt = {
                "model_state": model.state_dict(),
                "config": vars(args),
                "q_cols": q_cols,
                "m_cols": m_cols,
                "q_cols_delta": q_cols_delta,
                "m_cols_delta": m_cols_delta,
                "preprocess_mode": "raw",
                "session_col": session_col,
                "split_info": split_info,
                "q_scaler": q_scaler.to_dict(),
                "m_scaler": m_scaler.to_dict(),
                "use_active_finger": args.use_active_finger,
                "active_finger_col": active_finger_col,
            }
            torch.save(ckpt, out_dir / "best_forward.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stop at epoch {epoch}")
                break

    best_ckpt = torch.load(out_dir / "best_forward.pt", map_location=device)
    model.load_state_dict(best_ckpt["model_state"])

    train_metrics = evaluate_forward(model, train_loader, q_scaler, device)
    val_metrics = evaluate_forward(model, val_loader, q_scaler, device)
    test_metrics = evaluate_forward(model, test_loader, q_scaler, device)

    result = {
        "train": {
            "mae_mean": train_metrics["mae_mean"],
            "rmse_mean": train_metrics["rmse_mean"],
            "mae_per_dim": train_metrics["mae_per_dim"].tolist(),
            "rmse_per_dim": train_metrics["rmse_per_dim"].tolist(),
        },
        "val": {
            "mae_mean": val_metrics["mae_mean"],
            "rmse_mean": val_metrics["rmse_mean"],
            "mae_per_dim": val_metrics["mae_per_dim"].tolist(),
            "rmse_per_dim": val_metrics["rmse_per_dim"].tolist(),
        },
        "test": {
            "mae_mean": test_metrics["mae_mean"],
            "rmse_mean": test_metrics["rmse_mean"],
            "mae_per_dim": test_metrics["mae_per_dim"].tolist(),
            "rmse_per_dim": test_metrics["rmse_per_dim"].tolist(),
        },
    }

    save_json(str(out_dir / "forward_metrics.json"), result)
    save_json(str(out_dir / "forward_history.json"), {"history": history})
    log_wandb_metrics(
        wandb_run,
        {
            "metrics/train_mae_mean": float(train_metrics["mae_mean"]),
            "metrics/val_mae_mean": float(val_metrics["mae_mean"]),
            "metrics/test_mae_mean": float(test_metrics["mae_mean"]),
            "metrics/train_rmse_mean": float(train_metrics["rmse_mean"]),
            "metrics/val_rmse_mean": float(val_metrics["rmse_mean"]),
            "metrics/test_rmse_mean": float(test_metrics["rmse_mean"]),
        },
        len(history),
    )
    close_wandb(wandb_run)

    print("Forward thumb test MAE mean:", test_metrics["mae_mean"])
    print("Forward thumb test RMSE mean:", test_metrics["rmse_mean"])


if __name__ == "__main__":
    main()
