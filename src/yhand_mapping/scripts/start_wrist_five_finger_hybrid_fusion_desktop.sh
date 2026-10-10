#!/usr/bin/env bash
set -eo pipefail

# Desktop-friendly launcher for wrist_five_finger_hybrid_fusion_right_launch.py
# Prompts:
# 1) target hand: left / right / both
# 2) whether to start wrist mocap node: yes / no

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

LAUNCH_FILE="wrist_five_finger_hybrid_fusion_right_launch.py"

if [[ ! -d "$WS_ROOT/src/yhand_mapping" ]]; then
  echo "错误: 未找到工作区目录: $WS_ROOT/src/yhand_mapping"
  read -r -p "按回车退出..."
  exit 1
fi

prompt_hand_terminal() {
  while true; do
    echo "请选择手部:"
    echo "  1) 左手"
    echo "  2) 右手"
    echo "  3) 双手"
    read -r -p "输入 1/2/3: " user_input
    case "$user_input" in
      1) printf '%s\n' "left"; return 0 ;;
      2) printf '%s\n' "right"; return 0 ;;
      3) printf '%s\n' "both"; return 0 ;;
    esac
    echo "输入无效，请重新输入。"
  done
}

prompt_wrist_terminal() {
  while true; do
    echo "是否开启手腕?"
    echo "  1) 是"
    echo "  2) 否"
    read -r -p "输入 1/2: " user_input
    case "$user_input" in
      1) printf '%s\n' "true"; return 0 ;;
      2) printf '%s\n' "false"; return 0 ;;
    esac
    echo "输入无效，请重新输入。"
  done
}

TARGET_HAND=""
START_MOCAP_NODE=""

if command -v zenity >/dev/null 2>&1; then
  TARGET_HAND="$(
    zenity --list \
      --title="YHand 启动设置" \
      --text="请选择左右手" \
      --radiolist \
      --column="" \
      --column="选项" \
      TRUE "左手" \
      FALSE "右手" \
      FALSE "双手" \
      --height=260 \
      --width=320
  )" || exit 0

  case "$TARGET_HAND" in
    "左手") TARGET_HAND="left" ;;
    "右手") TARGET_HAND="right" ;;
    "双手") TARGET_HAND="both" ;;
    *)
      echo "错误: 无法识别手部选项: $TARGET_HAND"
      exit 1
      ;;
  esac

  START_MOCAP_NODE="$(
    zenity --list \
      --title="YHand 启动设置" \
      --text="是否开启手腕?" \
      --radiolist \
      --column="" \
      --column="选项" \
      TRUE "是" \
      FALSE "否" \
      --height=220 \
      --width=320
  )" || exit 0

  case "$START_MOCAP_NODE" in
    "是") START_MOCAP_NODE="true" ;;
    "否") START_MOCAP_NODE="false" ;;
    *)
      echo "错误: 无法识别手腕选项: $START_MOCAP_NODE"
      exit 1
      ;;
  esac
else
  TARGET_HAND="$(prompt_hand_terminal)"
  START_MOCAP_NODE="$(prompt_wrist_terminal)"
fi

RUN_CMD=$(cat <<EOF
set -eo pipefail
cd $(printf '%q' "$WS_ROOT")
if [[ -n "\${ROS_DISTRO:-}" && -f "/opt/ros/\${ROS_DISTRO}/setup.bash" ]]; then
  set +u
  source "/opt/ros/\${ROS_DISTRO}/setup.bash"
  set -u
else
  for distro in jazzy humble iron rolling foxy galactic; do
    if [[ -f "/opt/ros/\${distro}/setup.bash" ]]; then
      set +u
      source "/opt/ros/\${distro}/setup.bash"
      set -u
      break
    fi
  done
fi

if [[ ! -f "install/setup.bash" ]]; then
  echo "错误: 未找到工作区环境文件: $WS_ROOT/install/setup.bash"
  echo "请先在工作区编译一次：colcon build"
  read -r -p "按回车关闭..."
  exit 1
fi

set +u
source "install/setup.bash"
set -u

echo "启动参数:"
echo "  target_hand: $TARGET_HAND"
echo "  start_mocap_node: $START_MOCAP_NODE"
echo

ros2 launch yhand_mapping $LAUNCH_FILE target_hand:=$TARGET_HAND start_mocap_node:=$START_MOCAP_NODE
rc=\$?
if [[ \$rc -ne 0 ]]; then
  echo
  echo "启动失败，退出码=\$rc"
  read -r -p "按回车关闭..."
fi
exit \$rc
EOF
)

if command -v gnome-terminal >/dev/null 2>&1; then
  exec gnome-terminal \
    --title "YHand Wrist Five Finger Hybrid Fusion" \
    -- \
    bash -lc "$RUN_CMD"
fi

exec bash -lc "$RUN_CMD"
