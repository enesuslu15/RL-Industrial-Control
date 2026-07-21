import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib.pyplot as plt
import numpy as np
import time

from stable_baselines3 import PPO
from environment.tank_env import TankEnv
from utils.pid_controller import PIDController

def run_pid_baseline(env, steps=100):
    print("Running PID Baseline...")
    pid = PIDController(kp=5.0, ki=0.1, kd=1.0, min_out=0.0, max_out=100.0)
    
    # PID requires setpoint, let's read from env
    setpoint = env.target_temperature
    
    obs, info = env.reset()
    
    temperatures = []
    powers = []
    
    for i in range(steps):
        current_temp = info.get('temperature', 20.0)
        
        # We simulate a 0.5s dt, as in env
        action_power = pid.compute(setpoint, current_temp, dt=0.5)
        
        # Env step takes array
        obs, reward, terminated, truncated, info = env.step(np.array([action_power]))
        
        temperatures.append(info['temperature'])
        powers.append(info['heater_power'])
        
        if terminated or truncated:
            break
            
    return temperatures, powers

def run_rl_agent(env, model_path, steps=100):
    print("Running RL Agent...")
    try:
        model = PPO.load(model_path)
    except Exception as e:
        print(f"Could not load RL model: {e}")
        return [], []
        
    obs, info = env.reset()
    
    temperatures = []
    powers = []
    
    for i in range(steps):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        temperatures.append(info['temperature'])
        powers.append(info['heater_power'])
        
        if terminated or truncated:
            break
            
    return temperatures, powers

def plot_comparison():
    env = TankEnv()
    
    # Run tests
    test_steps = 60 # 60 steps = ~30 seconds in sim time
    
    pid_temps, pid_powers = run_pid_baseline(env, steps=test_steps)
    rl_temps, rl_powers = run_rl_agent(env, "data/ppo_tank_model", steps=test_steps)
    
    env.close()

    # Plot Results
    time_axis = np.arange(len(pid_temps)) * 0.5  # 0.5s per step
    
    plt.figure(figsize=(12, 6))
    
    # Temperature Plot
    plt.subplot(2, 1, 1)
    plt.plot(time_axis, pid_temps, label='PID Temperature', linestyle='--')
    if rl_temps:
        plt.plot(time_axis[:len(rl_temps)], rl_temps, label='RL Temperature', linewidth=2)
    plt.axhline(y=50.0, color='r', linestyle=':', label='Target Temperature')
    plt.ylabel("Temperature (°C)")
    plt.title("PID vs Reinforcement Learning Control")
    plt.legend()
    plt.grid(True)
    
    # Power Plot
    plt.subplot(2, 1, 2)
    plt.plot(time_axis, pid_powers, label='PID Power', linestyle='--')
    if rl_powers:
        plt.plot(time_axis[:len(rl_powers)], rl_powers, label='RL Power', linewidth=2)
    plt.ylabel("Heater Power (%)")
    plt.xlabel("Time (s)")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("data/comparison_plot.png")
    print("Plot saved to data/comparison_plot.png")
    plt.show()

if __name__ == "__main__":
    plot_comparison()
