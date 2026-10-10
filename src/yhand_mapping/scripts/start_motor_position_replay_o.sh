#!/usr/bin/env bash
set -eo pipefail

# One-click launcher for motor position replay auto-loop mode.
# Usage:
#   ./start_motor_position_replay_o.sh [hand] [mode] [interval] [snapshot_path]
# Example:
#   ./start_motor_position_replay_o.sh right 1 1.0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_ROOT="${YHAND_WS_ROOT:-}"
if [[ -z "$WS_ROOT" ]]; then
  search_dir="$SCRIPT_DIR"
  while [[ "$search_dir" != "/" ]]; do
    if [[ -d "$search_dir/src/yhand_mapping" ]]; then
      WS_ROOT="$search_dir"
      break
    fi
    search_dir="$(dirname "$search_dir")"
  done
fi

if [[ -z "$WS_ROOT" || ! -d "$WS_ROOT/src/yhand_mapping" ]]; then
  echo "错误: 未找到工作区，请设置 YHAND_WS_ROOT。"
  exit 1
fi

HAND="${1:-right}"            # left | right
AUTO_MODE="${2:-1}"           # 1 | 2
AUTO_INTERVAL="${3:-1.0}"     # seconds
SNAPSHOT_PATH="${4:-}"        # optional absolute/relative path

if [[ "$HAND" != "left" && "$HAND" != "right" ]]; then
  echo "错误: hand 仅支持 left/right，当前: $HAND"
  exit 1
fi

if [[ "$AUTO_MODE" != "1" && "$AUTO_MODE" != "2" ]]; then
  echo "错误: mode 仅支持 1/2，当前: $AUTO_MODE"
  exit 1
fi

# Source ROS distro setup (prefer current ROS_DISTRO if present).
if [[ -n "${ROS_DISTRO:-}" && -f "/opt/ros/${ROS_DISTRO}/setup.bash" ]]; then
  set +u
  # shellcheck disable=SC1090
  source "/opt/ros/${ROS_DISTRO}/setup.bash"
  set -u
else
  for distro in humble jazzy iron rolling foxy galactic; do
    if [[ -f "/opt/ros/${distro}/setup.bash" ]]; then
      set +u
      # shellcheck disable=SC1090
      source "/opt/ros/${distro}/setup.bash"
      set -u
      break
    fi
  done
fi

if [[ ! -f "$WS_ROOT/install/setup.bash" ]]; then
  echo "错误: 未找到工作区环境文件: $WS_ROOT/install/setup.bash"
  echo "请先在工作区编译一次：colcon build"
  exit 1
fi

# shellcheck disable=SC1091
set +u
source "$WS_ROOT/install/setup.bash"
set -u
cd "$WS_ROOT"

CMD=(
  ros2 launch yhand_mapping motor_position_replay_launch.py
  "target_hand:=$HAND"
  "auto_loop_mode:=$AUTO_MODE"
  "auto_loop_interval:=$AUTO_INTERVAL"
)

if [[ -n "$SNAPSHOT_PATH" ]]; then
  CMD+=("snapshot_path:=$SNAPSHOT_PATH")
fi

echo "[start_motor_position_replay_o] workspace=$WS_ROOT"
echo "[start_motor_position_replay_o] hand=$HAND, mode=$AUTO_MODE, interval=${AUTO_INTERVAL}s"
if [[ -n "$SNAPSHOT_PATH" ]]; then
  echo "[start_motor_position_replay_o] snapshot_path=$SNAPSHOT_PATH"
fi
echo "[start_motor_position_replay_o] launching..."

exec "${CMD[@]}"
