import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

class PPOAgent:
    def __init__(self, env, log_dir='./logs/', models_dir='./models/'):
        self.env = env
        self.log_dir = log_dir
        self.models_dir = models_dir
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)

        self.model = PPO('MlpPolicy', self.env, verbose=1,
                         tensorboard_log=self.log_dir,
                         device='auto') # Use 'cuda' if you have a GPU

    def train(self, total_timesteps=100000):
        # Save a checkpoint every 1000 steps
        checkpoint_callback = CheckpointCallback(save_freq=1000, save_path=self.models_dir,
                                                 name_prefix='ppo_nav')
        
        self.model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = PPO.load(path, env=self.env)

    def predict(self, obs):
        return self.model.predict(obs)