#!/usr/bin/env bash
set -eo pipefail

# Desktop-friendly launcher: opens a terminal window and runs
# start_motor_position_replay_o.sh so users can double-click an icon.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_SCRIPT="$SCRIPT_DIR/start_motor_position_replay_o.sh"

HAND="${1:-right}"
AUTO_MODE="${2:-1}"
AUTO_INTERVAL="${3:-1.0}"
SNAPSHOT_PATH="${4:-}"

if [[ ! -x "$BASE_SCRIPT" ]]; then
  echo "错误: 未找到可执行脚本: $BASE_SCRIPT"
  exit 1
fi

if command -v gnome-terminal >/dev/null 2>&1; then
  CMD=("$BASE_SCRIPT" "$HAND" "$AUTO_MODE" "$AUTO_INTERVAL")
  if [[ -n "$SNAPSHOT_PATH" ]]; then
    CMD+=("$SNAPSHOT_PATH")
  fi

  printf -v CMD_STR '%q ' "${CMD[@]}"
  exec gnome-terminal \
    --title "YHand Motor Replay O" \
    -- \
    bash -lc "${CMD_STR}; rc=\$?; if [[ \$rc -ne 0 ]]; then echo; echo \"启动失败，退出码=\$rc\"; read -r -p \"按回车关闭...\"; fi"
fi

exec "$BASE_SCRIPT" "$HAND" "$AUTO_MODE" "$AUTO_INTERVAL" "$SNAPSHOT_PATH"
