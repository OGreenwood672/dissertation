# Multi-Agent Reinforcement Learning with Emergent Communication in Assembly Line Environments

This repository contains the codebase for a Cambridge Computer Science undergraduate dissertation project. The research compares discrete quantized and continuous communication channels in multi-agent reinforcement learning (MARL) using a custom Rust-based warehouse assembly line simulation and Python-based learning agents.

## Structure

- `rust_world/` — Rust simulation environment and agent core (with PyO3 bindings)
- `python_controller/` — Python code for training, policies, logging, and evaluation
- `configs/` — Experiment and environment configuration files
- `docs/` — Documentation, figures, and system architecture diagrams
- `results/` — Logs, checkpoints, and analysis outputs
- `notebooks/` — Jupyter notebooks (for exploration and post-experiment analysis)
- `tests/` — Integration and unit tests

## Key Features

- Hybrid system: Rust for fast simulation, Python for flexible RL training
- Support for discrete (VQ-VAE, HQ-VAE) and continuous agent communication protocols
- Modular design for agents, environment maps, and policy/training loops
- Automated logging, checkpointing, and evaluation

## Getting Started

To set up the Python environment on a Linux-based system, follow these steps:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

To setup the rust environment code:

```bash
maturin develop
```

To run the controller:

```bash
python -m controller.main
```

## License

This repository is provided for academic and research purposes only.
