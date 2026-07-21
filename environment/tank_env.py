import gymnasium as gym
from gymnasium import spaces
import numpy as np
import asyncio
import logging

from communication.opcua_client import OPCUAClient

logger = logging.getLogger("TankEnv")

class TankEnv(gym.Env):
    """
    Custom Environment that follows gym interface.
    This environment connects to the TIA Portal (or Mock) OPC UA Server.
    """
    metadata = {'render.modes': ['console']}

    def __init__(self, opcua_url="opc.tcp://192.168.0.1:4840"):
        super(TankEnv, self).__init__()
        
        # Action space: HeaterPower from 0.0 to 100.0
        # For PPO, normalized action space [-1.0, 1.0] is often better, 
        # but we can start with raw values for simplicity or scale them.
        self.action_space = spaces.Box(low=0.0, high=100.0, shape=(1,), dtype=np.float32)
        
        # Observation space: Temperature (e.g. 0 to 100 degrees)
        self.observation_space = spaces.Box(low=0.0, high=100.0, shape=(1,), dtype=np.float32)
        
        self.target_temperature = 50.0  # Goal temperature
        
        # OPC UA Setup
        self.opcua_url = opcua_url
        self.client = OPCUAClient(self.opcua_url)
        self.loop = asyncio.get_event_loop()
        
        # Connect synchronously during init (or can be done lazy)
        try:
            self.loop.run_until_complete(self.client.connect())
            
            # The namespace and ID might vary based on TIA Portal or Mock Server
            # These are the ones we defined in mock_opcua_server.py
            self.loop.run_until_complete(self.client.register_node("Temperature", "ns=4;i=3"))
            self.loop.run_until_complete(self.client.register_node("HeaterPower", "ns=4;i=2"))
        except Exception as e:
            logger.error(f"Failed to initialize OPC UA connection: {e}")

        self.current_step = 0
        self.max_steps = 200

    def step(self, action):
        from asyncua import ua
        
        heater_power = float(action[0])
        # Clamp action just in case
        heater_power = max(0.0, min(100.0, heater_power))
        
        # Send Action to PLC
        self.loop.run_until_complete(self.client.write_value("HeaterPower", heater_power, ua.VariantType.Float))
        
        # Wait a bit for PLC to process (simulating process time / cycle time)
        # In a real environment, you might wait for a specific PLC tick or run fixed time.
        self.loop.run_until_complete(asyncio.sleep(0.5))
        
        # Read New State
        temperature = self.loop.run_until_complete(self.client.read_value("Temperature"))
        
        # Calculate Reward
        # Negative reward for distance to target. Also small penalty for high power to save energy.
        temp_error = abs(self.target_temperature - temperature)
        reward = -temp_error - (heater_power * 0.01)
        
        # Done condition
        self.current_step += 1
        terminated = False
        truncated = self.current_step >= self.max_steps
        
        if temp_error < 0.5:
            # We reached the target nicely
            reward += 10.0
            
        info = {"temperature": temperature, "heater_power": heater_power}
        observation = np.array([temperature], dtype=np.float32)
        
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        from asyncua import ua
        
        self.current_step = 0
        # Reset the environment by setting Heater to 0 and Temperature to ambient (e.g., 20.0)
        # Note: Writing Temperature directly to PLC might not be possible if it's an analog input,
        # but in simulation we can reset the state variable.
        self.loop.run_until_complete(self.client.write_value("HeaterPower", 0.0, ua.VariantType.Float))
        self.loop.run_until_complete(self.client.write_value("Temperature", 20.0, ua.VariantType.Float))
        
        self.loop.run_until_complete(asyncio.sleep(0.5))
        temperature = self.loop.run_until_complete(self.client.read_value("Temperature"))
        
        observation = np.array([temperature], dtype=np.float32)
        info = {}
        return observation, info

    def render(self):
        pass

    def close(self):
        self.loop.run_until_complete(self.client.disconnect())
