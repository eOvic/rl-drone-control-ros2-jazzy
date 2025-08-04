# RL-Powered Drone Altitude Control with ROS 2 and PX4

This project demonstrates how to train a drone to perform a stable altitude-hold maneuver using Reinforcement Learning. The system integrates the power of the PX4 flight stack, the Gazebo physics simulator, and the ROS 2 robotics framework with a Deep Q-Network (DQN) agent trained using Stable Baselines3.

## Demonstration

A key challenge in this project was overcoming the agent's initial tendency to "give up." The GIF below shows the final trained agent successfully controlling its thrust to maintain the target altitude of 20 meters, even after starting from a slightly lower position.

[![Demonstration of Drone Altitude Control](https://github.com/chahine/rl-drone-control-ros2/blob/main/media/demo.png?raw=true)](https://github.com/chahine/rl-drone-control-ros2/blob/main/media/demo.mp4)

## Project Overview

The goal of this project is to replace a traditional PID controller for a single objective (altitude hold) with a Reinforcement Learning agent that learns the optimal control policy from scratch. This serves as a foundational project for more complex autonomous behaviors.

The agent's objective is simple:
- **Goal:** Maintain the drone at a target altitude of 20.0 meters.
- **Actions:** The agent can choose to `increase thrust`, `decrease thrust`, or `restart` the episode.
- **Observations:** It perceives the environment through its vertical position (z), vertical velocity (z), and the last applied thrust command.

## Key Features

- **End-to-End RL Pipeline:** A complete, working pipeline from simulation to a trained AI agent.
- **Custom ROS 2 & Gymnasium Environment:** A custom `DroneEnv` class that bridges the gap between the ROS 2 ecosystem and the standard Gymnasium (formerly OpenAI Gym) API for RL training.
- **Reward Shaping:** The reward function was carefully designed to provide continuous feedback, rewarding the agent for getting closer to the target altitude.
- **Constraint Handling:** The environment enforces physical constraints, such as maximum velocity and operational altitude, terminating the episode and penalizing the agent for violations.
- **Stable Baselines3 Integration:** Leverages the robust and popular Stable Baselines3 library for training a DQN agent.

## Installations
### Create a virtual environment
```commandline
# create
conda create venv

# activate
conda activate venv 
```

### Install PX4
```commandline
cd ~
git clone https://github.com/PX4/PX4-Autopilot.git --recursive
cd PX4-Autopilot/
git checkout v1.15.2
git submodule update --init --recursive
make px4_sitl
```
### Install ROS 2
To get the best installation possible check the ROS2 Jazzy website installation guide : 
https://docs.ros.org/en/jazzy/Installation.html

### Setup Micro XRCE-DDS Agent & Client
```commandline
cd ~
git clone https://github.com/eProsima/Micro-XRCE-DDS-Agent.git
cd Micro-XRCE-DDS-Agent
mkdir build
cd build
cmake ..
make
sudo make install
sudo ldconfig /usr/local/lib/
```
### Build ROS 2 Workspace
We will need px4_msgs and px4_ros_com:

```commandline
mkdir -p ~/ws_ros2/src/
cd ~/ws_ros2/src/
git clone https://github.com/PX4/px4_msgs.git
git clone https://github.com/PX4/px4_ros_com.git
git clone https://github.com/eOvic/rl-drone-control-ros2-jazzy.git
cd ..
source /opt/ros/humble/setup.bash
colcon build
source install/setup.bash
```
### Install MAVSDK (installation must be in Conda env)
```commandline
pip install mavsdk
pip install aioconsole
pip install pygame
pip install numpy
```
Navigate to the package folder and run this command : 
```
pip install -r requirements.txt
```
### Additional Configs
- Put below lines in your bashrc:
```commandline
source /opt/ros/humble/setup.bash
export GZ_SIM_RESOURCE_PATH=~/.gz/models
```

## Tech Stack

- **AI / Reinforcement Learning:**
  - Stable Baselines3 (DQN)
  - Gymnasium (OpenAI Gym)
  - Python 3
- **Robotics & Simulation:**
  - ROS 2 (Jazzy)
  - PX4 Autopilot (Software-in-the-Loop)
  - Gazebo Simulator

## Setup and Installation

### Prerequisites

- Ubuntu 24.04 with a complete ROS 2 Jazzy installation.
- PX4 Autopilot source code cloned and built, with the SITL + Gazebo simulation environment fully set up version 1.15.2 since it's the most stable
- Python packages: `stable-baselines3`, `gymnasium`, `rclpy`

### Installation Steps

1.  **Build the ROS 2 Package:**
    Navigate to the root of your workspace and build the package using `colcon`.
    ```bash
    cd ~/ws_ros2/
    colcon build --packages-select px4_rl_project
    ```

2.  **Source the Workspace:**
    In every new terminal you use, you must source the workspace to make the package available.
    ```bash
    source ~/ws_ros2/install/setup.bash
    ```

## Usage

Running the project requires three terminals.

### Terminal 1: Launch the PX4-Gazebo Simulation

First, launch the PX4 SITL simulation. This command will start the PX4 flight controller and open a Gazebo window with the drone model.

```bash
# Navigate to your PX4 directory
cd ~/PX4-Autopilot/

# You must choose a world that has a ground plane.
PX4_GZ_WORLD=lawn make px4_sitl gz_x500_depth
```

Wait for the simulation to load completely and for the PX4 terminal to be ready for takeoff.
### Terminal 2: Micro DDS
To ensure the connection between Gazebo, ROS and PX4 we use MicroXRCEAgent 
Open another Terminal and write this command : 
```
 MicroXRCEAgent udp4 -p 8888 
``` 
IMPORTANT NOTE : You must run QGroundControl in the backgroud :
https://docs.qgroundcontrol.com/master/en/qgc-user-guide/getting_started/download_and_install.html
### Terminal 3: Run the RL Training Agent

Once the simulation is running, open a new terminal, source your ROS 2 workspace, and run the training script.

```bash
conda activate venv

# Source your workspace
source ~/ws_ros2/install/setup.bash

# Run the training script
python3 ~/ws_ros2/src/px4_rl_project/px4_rl_project/train_agent.py
```

You will see log output from both the drone environment (takeoff sequence, position updates) and the Stable Baselines3 trainer (rollout rewards, episode lengths, etc.).

## The Learning Journey: From Failure to Flight

A fascinating part of this project was debugging the agent's behavior.

-   **Initial Problem:** The agent quickly learned that "restarting" the episode was an easy way to escape situations where it was performing poorly. It got stuck in a loop of immediately choosing the `RESTART` action.
-   **Solution - Reward Shaping:** The `RESTART` action was given a large negative penalty (-10.0). This immediately taught the agent that restarting was an undesirable outcome.

## Future Improvements

-   **Complex Tasks:** Extend the agent's capabilities to perform path following or obstacle avoidance.
-   **Continuous Action Space:** Move from a discrete action space (DQN) to a continuous one using an algorithm like PPO or SAC for finer thrust control.
-   **Sim-to-Real Transfer:** Explore techniques to transfer the policy learned in simulation to a real drone and using PX4 ensures that.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
