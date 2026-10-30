"""Launch the mouth-only Fukuwarai world and the installed CRANE+ robot."""
import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from crane_plus_description.robot_description_loader import RobotDescriptionLoader
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node


def generate_launch_description():
    root = Path(__file__).resolve().parents[1]
    description_package = get_package_share_directory('crane_plus_description')
    loader = RobotDescriptionLoader()
    loader.use_gazebo = 'true'
    loader.use_ignition = 'true'
    loader.gz_control_config_package = 'crane_plus_control'
    loader.gz_control_config_file_path = 'config/crane_plus_controllers.yaml'
    description = loader.load()
    env = {
        'IGN_GAZEBO_SYSTEM_PLUGIN_PATH': os.environ.get('LD_LIBRARY_PATH',''),
        'IGN_GAZEBO_RESOURCE_PATH': str(root/'models') + ':' + os.path.dirname(description_package) + ':' +
                                  os.environ.get('IGN_GAZEBO_RESOURCE_PATH',''),
    }
    return LaunchDescription([
        ExecuteProcess(cmd=['ign','gazebo','-r',str(root/'worlds/fukuwarai.sdf'),
                            '--gui-config',str(root/'config/fukuwarai.gui.config')],
                       additional_env=env,output='screen'),
        Node(package='robot_state_publisher',executable='robot_state_publisher',
             parameters=[{'robot_description':description,'use_sim_time':True}]),
        Node(package='ros_gz_sim',executable='create',
             arguments=['-world','default','-topic','/robot_description',
                        '-name','crane_plus','-z','1.015']),
        *[Node(package='controller_manager',executable='spawner',
               arguments=[name,'--controller-manager-timeout','120']) for name in
          ['joint_state_broadcaster','crane_plus_arm_controller','crane_plus_gripper_controller']],
        Node(package='ros_gz_bridge',executable='parameter_bridge',
             arguments=['/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock']),
        Node(package='ros_gz_image',executable='image_bridge',
             arguments=['/fukuwarai/overhead/image'], parameters=[{'qos':'sensor_data'}]),
    ])
