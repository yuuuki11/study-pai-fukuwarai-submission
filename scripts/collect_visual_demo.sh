#!/usr/bin/env bash
set -e
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export ROS_LOG_DIR="$(mktemp -d /tmp/fukuwarai-visual.XXXXXX)"
source /opt/ros/humble/setup.bash
# Share the arm lock with the simulator-state collector.
exec 9>/tmp/fukuwarai-expert.lock
flock -n 9 || { echo 'Another arm collector is already running.'; exit 1; }
exec python3 "$PROJECT_ROOT/scripts/collect_visual_demo.py" "$@"
