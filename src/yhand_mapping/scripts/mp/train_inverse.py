import argparse
from pathlib import Path
from typing import List, Tuple

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

    p.add_argument("--q-dim", type=int, default=20)
    p.add_argument("--m-dim", type=int, default=24)
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

    p.add_argument("--forward-ckpt", type=str, default="", help="best_forward.pt path. If set, enable cycle loss.")
    p.add_argument("--lambda-motor", type=float, default=1.0)
    p.add_argument("--lambda-cycle", type=float, default=0.5)
    p.add_argument("--wandb", action="store_true", help="enable Weights & Biases logging")
    p.add_argument("--wandb-project", type=str, default="mapping")
    p.add_argument("--wandb-run-name", type=str, default="")
    p.add_argument("--wandb-mode", type=str, default="", choices=["", "online", "offline", "disabled"])

    return p.parse_args()


def build_column_names(q_dim: int, m_dim: int, q_prefix: str, m_prefix: str) -> Tuple[List[str], List[str]]:
    q_cols = [f"{q_prefix}{i}" for i in range(q_dim)]
    m_cols = [f"{m_prefix}{i}" for i in range(m_dim)]
    return q_cols, m_cols


def make_loaders_from_scalers(
    train_df,
    val_df,
    test_df,
    q_cols_delta: List[str],
    m_cols_delta: List[str],
    q_scaler: Standardizer,
    m_scaler: Standardizer,
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

    return train_loader, val_loader, test_loader


def evaluate_inverse(model, loader, m_scaler, device):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            q = batch["q"].to(device)
            active = batch["active"].to(device)
            x = concat_if_needed(q, active)
            pred_m = model(x)
            preds.append(pred_m.cpu())
            targets.append(batch["m"])
    pred = torch.cat(preds, dim=0)
    target = torch.cat(targets, dim=0)
    return regression_metrics_raw(pred, target, m_scaler)


def evaluate_cycle(inverse_model, forward_model, loader, q_scaler, fwd_use_active, device):
    inverse_model.eval()
    forward_model.eval()

    preds_q, targets_q = [], []
    with torch.no_grad():
        for batch in loader:
            q = batch["q"].to(device)
            active = batch["active"].to(device)

            inv_in = concat_if_needed(q, active)
            pred_m = inverse_model(inv_in)

            fwd_in = concat_if_needed(pred_m, active if fwd_use_active else None)
            pred_q = forward_model(fwd_in)

            preds_q.append(pred_q.cpu())
            targets_q.append(batch["q"])

    pred_q = torch.cat(preds_q, dim=0)
    target_q = torch.cat(targets_q, dim=0)
    return regression_metrics_raw(pred_q, target_q, q_scaler)


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

    forward_model = None
    q_scaler = None
    m_scaler = None
    split_info = None
    fwd_use_active = False

    # If forward checkpoint exists, reuse its split + scalers + forward model.
    if args.forward_ckpt:
        fwd_ckpt = torch.load(args.forward_ckpt, map_location=device)
        q_scaler = Standardizer.from_dict(fwd_ckpt["q_scaler"])
        m_scaler = Standardizer.from_dict(fwd_ckpt["m_scaler"])
        split_info = fwd_ckpt["split_info"]
        fwd_use_active = bool(fwd_ckpt.get("use_active_finger", False))

        fwd_in_dim = args.m_dim + (5 if fwd_use_active else 0)
        forward_model = MLP(
            in_dim=fwd_in_dim,
            out_dim=args.q_dim,
            hidden_dims=tuple(fwd_ckpt["config"]["hidden_dims"]),
            dropout=float(fwd_ckpt["config"]["dropout"]),
        ).to(device)
        forward_model.load_state_dict(fwd_ckpt["model_state"])
        forward_model.eval()
        for p in forward_model.parameters():
            p.requires_grad = False

    if split_info is None:
        split_info = make_split_info(
            df=df,
            split_col=split_col,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )

    train_df, val_df, test_df = apply_split_info(df, split_info)

    if q_scaler is None:
        q_scaler = Standardizer.fit(train_df[q_cols_delta].to_numpy(dtype=np.float32))
    if m_scaler is None:
        m_scaler = Standardizer.fit(train_df[m_cols_delta].to_numpy(dtype=np.float32))

    train_loader, val_loader, test_loader = make_loaders_from_scalers(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        q_cols_delta=q_cols_delta,
        m_cols_delta=m_cols_delta,
        q_scaler=q_scaler,
        m_scaler=m_scaler,
        use_active_finger=args.use_active_finger,
        active_finger_col=active_finger_col,
        batch_size=args.batch_size,
    )

    inv_in_dim = args.q_dim + (5 if args.use_active_finger else 0)
    inverse_model = MLP(
        in_dim=inv_in_dim,
        out_dim=args.m_dim,
        hidden_dims=tuple(args.hidden_dims),
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(inverse_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    huber = nn.HuberLoss(delta=1.0)

    best_val = float("inf")
    bad_epochs = 0
    history = []
    wandb_run = init_wandb(args, out_dir, "inverse")

    for epoch in range(args.epochs):
        inverse_model.train()
        train_loss = 0.0
        n_train = 0

        for batch in train_loader:
            q = batch["q"].to(device)
            m = batch["m"].to(device)
            active = batch["active"].to(device)

            inv_in = concat_if_needed(q, active)
            pred_m = inverse_model(inv_in)

            loss_motor = huber(pred_m, m)
            loss = args.lambda_motor * loss_motor

            if forward_model is not None and args.lambda_cycle > 0.0:
                fwd_in = concat_if_needed(pred_m, active if fwd_use_active else None)
                pred_q = forward_model(fwd_in)
                loss_cycle = huber(pred_q, q)
                loss = loss + args.lambda_cycle * loss_cycle

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bs = q.size(0)
            train_loss += loss.item() * bs
            n_train += bs

        train_loss /= max(1, n_train)

        inverse_model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                q = batch["q"].to(device)
                m = batch["m"].to(device)
                active = batch["active"].to(device)

                inv_in = concat_if_needed(q, active)
                pred_m = inverse_model(inv_in)

                loss_motor = huber(pred_m, m)
                loss = args.lambda_motor * loss_motor

                if forward_model is not None and args.lambda_cycle > 0.0:
                    fwd_in = concat_if_needed(pred_m, active if fwd_use_active else None)
                    pred_q = forward_model(fwd_in)
                    loss_cycle = huber(pred_q, q)
                    loss = loss + args.lambda_cycle * loss_cycle

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
        print(f"[Inverse] epoch={epoch:03d} train={train_loss:.6f} val={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            bad_epochs = 0
            ckpt = {
                "model_state": inverse_model.state_dict(),
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
                "forward_ckpt": args.forward_ckpt,
            }
            torch.save(ckpt, out_dir / "best_inverse.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stop at epoch {epoch}")
                break

    # final eval
    best_ckpt = torch.load(out_dir / "best_inverse.pt", map_location=device)
    inverse_model.load_state_dict(best_ckpt["model_state"])

    train_motor = evaluate_inverse(inverse_model, train_loader, m_scaler, device)
    val_motor = evaluate_inverse(inverse_model, val_loader, m_scaler, device)
    test_motor = evaluate_inverse(inverse_model, test_loader, m_scaler, device)

    result = {
        "motor": {
            "train": {
                "mae_mean": train_motor["mae_mean"],
                "rmse_mean": train_motor["rmse_mean"],
                "mae_per_dim": train_motor["mae_per_dim"].tolist(),
                "rmse_per_dim": train_motor["rmse_per_dim"].tolist(),
            },
            "val": {
                "mae_mean": val_motor["mae_mean"],
                "rmse_mean": val_motor["rmse_mean"],
                "mae_per_dim": val_motor["mae_per_dim"].tolist(),
                "rmse_per_dim": val_motor["rmse_per_dim"].tolist(),
            },
            "test": {
                "mae_mean": test_motor["mae_mean"],
                "rmse_mean": test_motor["rmse_mean"],
                "mae_per_dim": test_motor["mae_per_dim"].tolist(),
                "rmse_per_dim": test_motor["rmse_per_dim"].tolist(),
            },
        }
    }

    if forward_model is not None and args.lambda_cycle > 0.0:
        train_cycle = evaluate_cycle(inverse_model, forward_model, train_loader, q_scaler, fwd_use_active, device)
        val_cycle = evaluate_cycle(inverse_model, forward_model, val_loader, q_scaler, fwd_use_active, device)
        test_cycle = evaluate_cycle(inverse_model, forward_model, test_loader, q_scaler, fwd_use_active, device)

        result["cycle_joint"] = {
            "train": {
                "mae_mean": train_cycle["mae_mean"],
                "rmse_mean": train_cycle["rmse_mean"],
                "mae_per_dim": train_cycle["mae_per_dim"].tolist(),
                "rmse_per_dim": train_cycle["rmse_per_dim"].tolist(),
            },
            "val": {
                "mae_mean": val_cycle["mae_mean"],
                "rmse_mean": val_cycle["rmse_mean"],
                "mae_per_dim": val_cycle["mae_per_dim"].tolist(),
                "rmse_per_dim": val_cycle["rmse_per_dim"].tolist(),
            },
            "test": {
                "mae_mean": test_cycle["mae_mean"],
                "rmse_mean": test_cycle["rmse_mean"],
                "mae_per_dim": test_cycle["mae_per_dim"].tolist(),
                "rmse_per_dim": test_cycle["rmse_per_dim"].tolist(),
            },
        }

        print("Inverse cycle joint test MAE mean:", test_cycle["mae_mean"])
        print("Inverse cycle joint test RMSE mean:", test_cycle["rmse_mean"])

    save_json(str(out_dir / "inverse_metrics.json"), result)
    save_json(str(out_dir / "inverse_history.json"), {"history": history})
    final_log = {
        "metrics/motor_train_mae_mean": float(train_motor["mae_mean"]),
        "metrics/motor_val_mae_mean": float(val_motor["mae_mean"]),
        "metrics/motor_test_mae_mean": float(test_motor["mae_mean"]),
        "metrics/motor_train_rmse_mean": float(train_motor["rmse_mean"]),
        "metrics/motor_val_rmse_mean": float(val_motor["rmse_mean"]),
        "metrics/motor_test_rmse_mean": float(test_motor["rmse_mean"]),
    }
    if forward_model is not None and args.lambda_cycle > 0.0:
        final_log.update(
            {
                "metrics/cycle_train_mae_mean": float(train_cycle["mae_mean"]),
                "metrics/cycle_val_mae_mean": float(val_cycle["mae_mean"]),
                "metrics/cycle_test_mae_mean": float(test_cycle["mae_mean"]),
                "metrics/cycle_train_rmse_mean": float(train_cycle["rmse_mean"]),
                "metrics/cycle_val_rmse_mean": float(val_cycle["rmse_mean"]),
                "metrics/cycle_test_rmse_mean": float(test_cycle["rmse_mean"]),
            }
        )
    log_wandb_metrics(wandb_run, final_log, len(history))
    close_wandb(wandb_run)

    print("Inverse motor test MAE mean:", test_motor["mae_mean"])
    print("Inverse motor test RMSE mean:", test_motor["rmse_mean"])


if __name__ == "__main__":
    main()
