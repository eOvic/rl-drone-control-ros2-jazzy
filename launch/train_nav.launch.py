#
# THIS IS THE CORRECTED LAUNCH FILE.
# IT BYPASSES THE FAULTY ROS 2 SCRIPT.
#
from launch import LaunchDescription
from launch.actions import ExecuteProcess
import sys
import os

def generate_launch_description():
    # Get the path to the current python interpreter
    python_executable = sys.executable
    
    # Get the path to your workspace
    workspace_path = os.path.expanduser('~/ws_ros2')

    return LaunchDescription([
        ExecuteProcess(
            # This command is like running "python -m px4_rl_project.train"
            # in your terminal. It uses the correct Conda python.
            cmd=[python_executable, '-m', 'px4_rl_project.train'],
            output='screen',
            emulate_tty=True,
            # We need to tell the process where to find the workspace
            additional_env={'PYTHONPATH': f"{workspace_path}/install/px4_rl_project/lib/python3.12/site-packages:" + (os.environ.get('PYTHONPATH') or '')}
        )
    ])