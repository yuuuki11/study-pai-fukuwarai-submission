#!/usr/bin/env bash
set -e
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# Keep Gazebo and terminal clients under the desktop user so Fast DDS shared
# memory and GUI files are not created under different owners.
if [ "$(id -u)" = 0 ] && id ubuntu >/dev/null 2>&1; then
  exec runuser -u ubuntu -- bash "$PROJECT_ROOT/scripts/start_fukuwarai.sh"
fi
export DISPLAY="${DISPLAY:-:0}"
export QT_IM_MODULE=xim
export QT_X11_NO_MITSHM=1
export LIBGL_ALWAYS_SOFTWARE=1
export ROS_LOG_DIR="$(mktemp -d /tmp/fukuwarai-ros-log.XXXXXX)"
source /opt/ros/humble/setup.bash
if pgrep -f '(^| )/usr/bin/ign gazebo( |$)' >/dev/null; then
  echo 'Gazebo is already running. To restart this scene: python3 /workspace/scripts/stop_fukuwarai.py'
  exit 1
fi
# An isolated session bus prevents the GUI from hanging before creating a
# window when the desktop input-method/session bus is unavailable.
exec dbus-run-session -- ros2 launch "$PROJECT_ROOT/launch/fukuwarai.launch.py"
