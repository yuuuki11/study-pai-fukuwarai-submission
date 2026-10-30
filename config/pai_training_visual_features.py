"""LeRobot v2.1 features for the visual Fukuwarai dataset in pai_training.

Point the course's configs/train_config.yaml ``config_path`` at this file when
setting up a future image-and-state policy run. Training is not started here.
"""

features = {
    "observation.images.overhead": {
        "dtype": "video",
        "shape": (3, 480, 640),
        "names": ["channels", "height", "width"],
    },
    "observation.state": {
        "dtype": "float32",
        "shape": (5,),
        "names": ["joint1", "joint2", "joint3", "joint4", "gripper"],
    },
    # pai_training's CRANE+ ACT config expects this field. It duplicates the
    # observed joint state and contains no simulator object coordinates.
    "observation.environment_state": {
        "dtype": "float32",
        "shape": (5,),
        "names": ["joint1", "joint2", "joint3", "joint4", "gripper"],
    },
    "action": {
        "dtype": "float32",
        "shape": (5,),
        "names": ["joint1", "joint2", "joint3", "joint4", "gripper"],
    },
}

robot_type = "crane_plus"
repo_id = "crane_plus_fukuwarai"
task_name = "Push the mouth piece to the face target and release it for a one second hold."
