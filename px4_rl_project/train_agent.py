# File: ~/ws_ros2/src/px4_rl_project/px4_rl_project/train_agent.py

import rclpy
from px4_rl_project.environment.px4_nav_env import DroneEnv
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

def main(args=None):
    rclpy.init(args=args)
    
    # Create environment
    env = DroneEnv()
    
    # Wrap environment
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    
    # Create checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=5000, 
        save_path="./dqn_drone_checkpoints/", 
        name_prefix="drone_model"
    )
    
    # Create DQN model for discrete action space
    model = DQN(
        "MlpPolicy", 
        env, 
        verbose=1, 
        tensorboard_log="./dqn_drone_tensorboard/",
        learning_rate=0.0005,
        buffer_size=50000,
        learning_starts=100,
        batch_size=32,
        target_update_interval=1000,
        exploration_fraction=0.3,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        train_freq=4,
        gradient_steps=1
    )
    
    print("Starting DQN training for altitude control...")
    print("Target: Keep drone at 20.0m altitude (reward zone: 19.7-20.3m)")
    print("Actions: 0=decrease thrust, 1=increase thrust, 2=restart")
    print("Constraints: Thrust[0.4-0.78], Vel_z[-3,3], Pos_z[10,30]")
    
    try:
        model.learn(
            total_timesteps=100000, 
            callback=checkpoint_callback, 
            log_interval=10
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    finally:
        print("Saving final model...")
        model.save("dqn_drone_final")
        print("Closing environment and shutting down ROS 2...")
        env.close()

if __name__ == '__main__':
    main()