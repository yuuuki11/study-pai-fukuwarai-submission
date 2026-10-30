#!/usr/bin/env bash
set -e
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export ROS_LOG_DIR="$(mktemp -d /tmp/fukuwarai-teleop.XXXXXX)"
source /opt/ros/humble/setup.bash
exec python3 "$PROJECT_ROOT/scripts/teleop_keyboard.py"
