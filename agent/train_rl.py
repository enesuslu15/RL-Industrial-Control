import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from environment.tank_env import TankEnv

def main():
    print("Initializing environment...")
    # NOTE: Make sure the mock_opcua_server.py is running before starting this script.
    env = TankEnv()
    
    # Check if the environment follows Gymnasium API
    print("Checking environment...")
    check_env(env, warn=True)
    
    print("Initializing PPO Model...")
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./data/tensorboard/")
    
    print("Starting Training...")
    # 2000 timesteps is a very short test run.
    model.learn(total_timesteps=2000, progress_bar=True)
    
    # Save the model
    model.save("data/ppo_tank_model")
    print("Model saved to data/ppo_tank_model.zip")
    
    # Test the trained model
    print("Testing the trained model...")
    obs, info = env.reset()
    for i in range(10):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i+1} | Temp: {info['temperature']:.2f} | Power: {info['heater_power']:.2f}% | Reward: {reward:.2f}")
        if terminated or truncated:
            break
            
    env.close()

if __name__ == "__main__":
    main()
