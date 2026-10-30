#!/usr/bin/env python3
"""Keyboard joint control for the Gazebo CRANE+. No command is sent on startup."""
import select
import sys
import termios
import time
import tty
import xml.etree.ElementTree as ET

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

ARM = ['crane_plus_joint1','crane_plus_joint2','crane_plus_joint4','crane_plus_joint3']
HAND = 'crane_plus_joint_hand'
HELP = '''
CRANE+ keyboard control (simulation)
  1/2/3/4 : select joint directly
  Q / E   : previous / next joint
  W / S   : selected joint + / - 0.025 rad
  O / C   : open / close gripper
  Space   : hold the current measured arm position
  H       : show this help
  Esc / Ctrl+C : quit
Keep this terminal focused while pressing keys. Commands are sent only on input.
Joint limits are enforced; collision avoidance is NOT performed by this node.
'''


class KeyboardNode(Node):
    def __init__(self):
        super().__init__('fukuwarai_keyboard')
        self.measured = {}
        self.limits = {}
        self.targets = None
        self.selected = 0
        self.arm_pub = self.create_publisher(JointTrajectory,'/crane_plus_arm_controller/joint_trajectory',10)
        self.hand_pub = self.create_publisher(JointTrajectory,'/crane_plus_gripper_controller/joint_trajectory',10)
        self.create_subscription(JointState,'/joint_states',self.on_state,10)
        qos = QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(String,'/robot_description',self.on_description,qos)

    def on_state(self,msg):
        self.measured.update(zip(msg.name,msg.position))

    def on_description(self,msg):
        for joint in ET.fromstring(msg.data).findall('joint'):
            limit=joint.find('limit')
            if limit is not None and 'lower' in limit.attrib and 'upper' in limit.attrib:
                self.limits[joint.attrib['name']] = (float(limit.attrib['lower']),float(limit.attrib['upper']))

    def ready(self):
        return all(j in self.measured and j in self.limits for j in ARM+[HAND])

    def send(self,publisher,names,positions):
        if not publisher.get_subscription_count():
            print('No controller subscriber; command skipped.',flush=True)
            return
        msg=JointTrajectory()
        msg.joint_names=names
        point=JointTrajectoryPoint()
        point.positions=[min(max(p,self.limits[n][0]),self.limits[n][1]) for n,p in zip(names,positions)]
        point.time_from_start.nanosec=300_000_000
        msg.points=[point]
        publisher.publish(msg)

    def key(self,key):
        key=key.lower()
        if key in ('1','2','3','4'):
            self.selected=int(key)-1
        elif key in ('q','e'):
            self.selected=(self.selected+(1 if key=='e' else -1))%len(ARM)
        elif key in ('w','s'):
            i=self.selected
            desired=self.targets[i]+(0.025 if key=='w' else -0.025)
            actual=self.measured[ARM[i]]
            lo,hi=self.limits[ARM[i]]
            self.targets[i]=max(lo,actual-0.15,min(hi,actual+0.15,desired))
            self.send(self.arm_pub,ARM,self.targets)
        elif key in ('o','c'):
            self.send(self.hand_pub,[HAND],[-0.5236 if key=='o' else 0.1745])
            print('Gripper '+('OPEN' if key=='o' else 'CLOSE'),flush=True)
        elif key==' ':
            self.targets=[self.measured[n] for n in ARM]
            self.send(self.arm_pub,ARM,self.targets)
        elif key=='h':
            print(HELP,flush=True)
        if key and key in '1234qews':
            print(f'{self.selected+1}: {ARM[self.selected]}  target={self.targets[self.selected]:+.3f} rad',flush=True)


def main():
    if not sys.stdin.isatty():
        raise SystemExit('Run this command in an interactive terminal.')
    rclpy.init()
    node=KeyboardNode()
    fd=sys.stdin.fileno()
    original=termios.tcgetattr(fd)
    try:
        print('Waiting for joint states and URDF limits (no motion)...',flush=True)
        deadline=time.monotonic()+10
        while rclpy.ok() and not node.ready():
            rclpy.spin_once(node,timeout_sec=0.1)
            if time.monotonic()>deadline:
                missing_states=[j for j in ARM+[HAND] if j not in node.measured]
                missing_limits=[j for j in ARM+[HAND] if j not in node.limits]
                raise RuntimeError(f'Missing joint states: {missing_states}; missing URDF limits: {missing_limits}. '
                                   'Start the Fukuwarai world as the same Ubuntu user.')
        node.targets=[node.measured[n] for n in ARM]
        print(HELP,flush=True)
        print('Ready. Starting pose preserved. Selected: '+ARM[0],flush=True)
        tty.setcbreak(fd)
        while rclpy.ok():
            rclpy.spin_once(node,timeout_sec=0.02)
            if select.select([sys.stdin],[],[],0)[0]:
                key=sys.stdin.read(1)
                if key in ('','\x1b','\x03'):
                    break
                node.key(key)
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(fd,termios.TCSADRAIN,original)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__=='__main__':
    main()
