#!/usr/bin/env bash
set -eo pipefail

# Desktop-friendly pretighten helper:
# 1) Prompt user to wear clamp and press 1
# 2) Run yhand_pretighten_launch.py for 5 seconds
# 3) Prompt power-cycle and clamp removal

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

if [[ ! -d "$WS_ROOT/src/yhand_mapping" ]]; then
  echo "错误: 未找到工作区目录: $WS_ROOT/src/yhand_mapping"
  read -r -p "按回车退出..."
  exit 1
fi

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
  read -r -p "按回车退出..."
  exit 1
fi

set +u
# shellcheck disable=SC1091
source "$WS_ROOT/install/setup.bash"
set -u
cd "$WS_ROOT"

echo "请先佩戴夹具，佩戴好后请按1。"
while true; do
  read -r -p "输入 1 开始预紧: " user_input
  if [[ "$user_input" == "1" ]]; then
    break
  fi
  echo "输入无效，请按 1。"
done

echo "开始预紧：运行 yhand_pretighten_launch.py（5秒）..."
echo "提示：预紧过程中请保持夹具稳定。"

set +e
timeout --signal=INT 5s ros2 launch yhand_mapping yhand_pretighten_launch.py
launch_rc=$?
set -e

if [[ $launch_rc -ne 0 && $launch_rc -ne 124 ]]; then
  echo "预紧启动失败，退出码: $launch_rc"
  read -r -p "按回车退出..."
  exit "$launch_rc"
fi

echo "预紧完毕，请关闭电源、摘掉夹具并重新供电，重新上电后再执行第二个对指demo展示的功能。"
read -r -p "按回车退出..."
