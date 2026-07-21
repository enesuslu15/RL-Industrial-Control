# RL Industrial Control (Smart Tank Reactor)

This repository demonstrates the integration of **Reinforcement Learning (PPO)** with industrial automation systems (Siemens S7-1200 PLC) using Python and **OPC UA**. The project aims to replace classic PID controllers with an AI agent that learns thermodynamic processes dynamically to optimize energy usage and settling time.

## Project Architecture

1. **TIA Portal (Environment):** A thermodynamic tank simulation written in SCL (Structured Control Language) running cyclically (100ms) inside a Siemens S7-1200 PLC.
2. **OPC UA Server:** The PLC exposes `Temperature` and `HeaterPower` nodes via its built-in OPC UA Server.
3. **Python (Agent):** A custom OpenAI Gymnasium environment (`tank_env.py`) that reads/writes data to the PLC as state/action.
4. **PyTorch (PPO):** Stable-Baselines3 is used to train the agent to reach a target temperature while minimizing energy expenditure.

## Directory Structure

* `agent/`: Contains the Reinforcement Learning training scripts (`train_rl.py`).
* `communication/`: Contains the OPC UA client module.
* `configs/`: Hyperparameters and configuration files.
* `data/`: TensorBoard logs, saved models, and plots.
* `environment/`: Custom Gymnasium environment for the tank simulation.
* `plc/`: TIA Portal SCL source code (`TankSimulation.scl`).
* `tests/`: Mock OPC UA server for testing without physical hardware.
* `utils/`: Baseline PID controller and plotting scripts.

## Installation

1. Clone this repository:
```bash
git clone https://github.com/<enesuslu15>/RL-Industrial-Control.git
cd RL-Industrial-Control
```

2. Create a virtual environment and install the required dependencies:
```bash
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

## TIA Portal Setup
1. Create a new S7-1200 project (Firmware >= V4.4).
2. Activate the **OPC UA Server** in the PLC hardware configuration.
3. Create a Global Data Block (`OPC_Data`) with `HeaterPower` and `Temperature` (Real).
4. Create a Function Block using the SCL code in `plc/TankSimulation.scl` and call it inside a Cyclic Interrupt OB (e.g., `OB30`, 100ms).
5. Map the variables to the OPC UA Server Interface.

## Usage

**1. Configure the IP Address:**
Update the OPC UA URL and Node IDs in `communication/opcua_client.py` and `environment/tank_env.py` to match your PLC's network settings.

**2. Train the RL Agent:**
```bash
python agent/train_rl.py
```

**3. Test and Plot (PID vs RL):**
```bash
python utils/plot_results.py
```

## License
MIT License
