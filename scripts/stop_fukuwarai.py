#!/usr/bin/env python3
"""Stop this repository's simulator and its ROS child processes only."""
import os
from pathlib import Path
import signal
import subprocess
import time

root=Path(__file__).resolve().parents[1]
rows=subprocess.check_output(['ps','-eo','pid=,ppid=,args='],text=True).splitlines()
processes={}
for row in rows:
    pid,ppid,command=row.strip().split(None,2)
    processes[int(pid)]=(int(ppid),command)
selected={pid for pid,(_,command) in processes.items()
          if 'ros2 launch '+str(root/'launch/fukuwarai.launch.py') in command or
          ('/usr/bin/ign gazebo' in command and str(root/'worlds/fukuwarai.sdf') in command)}
while True:
    expanded=selected|{pid for pid,(ppid,_) in processes.items() if ppid in selected}
    if expanded==selected:
        break
    selected=expanded
for sig,seconds in ((signal.SIGINT,6),(signal.SIGTERM,3),(signal.SIGKILL,1)):
    alive=[]
    for pid in selected:
        try:
            os.kill(pid,sig)
            alive.append(pid)
        except ProcessLookupError:
            pass
    if not alive:
        break
    deadline=time.monotonic()+seconds
    while time.monotonic()<deadline:
        if not any(Path(f'/proc/{pid}').exists() for pid in alive):
            break
        time.sleep(0.1)
print('Stopped Fukuwarai scene.' if selected else 'Fukuwarai scene is not running.')
