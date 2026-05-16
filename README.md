# WhisperWise: Emergent Communication in Multi-Agent Reinforcement Learning

A research codebase for studying **emergent discrete communication** in cooperative multi-agent reinforcement learning (MARL) under partial observability. The project was built for a Cambridge Computer Science Part II dissertation and implements a high-throughput Rust environment, a Python/PyTorch MAPPO training stack, and interchangeable communication protocols including AIMotherTongue and WhisperWise.

## Project scope

The repository investigates whether agents can learn useful, interpretable communication protocols that improve coordination on complex cooperative tasks. The dissertation studies a configurable assembly-line-style Dec-POMDP in which agents have asymmetric information and must exchange compressed messages to discover stations, coordinate movement, and complete recipes.

The codebase contributes:

- A vectorised Rust simulation with Python bindings for fast rollout collection.
- A MAPPO implementation with shared actors, a centralised critic, GAE, rollout buffering, checkpointing, and logging.
- Pluggable communication protocols, including no-communication, discrete communication, AIMotherTongue variants, and WhisperWise.
- Offline language pretraining and imitation learning pipelines for discrete communication experiments.
- Analysis tooling for codebook usage, semantic grounding, memory probing, and protocol behaviour.

## Repository layout

```text
configs/                  Configuration files for environment, MAPPO, communication, autoencoders, and training runs
controller/               Python training, model, protocol, and infrastructure code
controller/marlcomms/     Communication protocol implementations
controller/marlcore/      Rollout buffer, config loading, logging, checkpointing
controller/marlmodels/    Actor, critic, VAE, encoder, and supporting neural modules
controller/marlrunners/   Training loops for RL, imitation learning, and language training
environment-rs/           Rust simulation backend with vectorised worlds and PyO3 bindings
frontend/                 Lightweight real-time visualiser
notebooks/                Evaluation and analysis notebooks
tests/                    Python unit tests
README.md                 Project overview and usage guide
```

The dissertation reports roughly 13,722 lines of code across the main repository, with substantial code in the Python controller, Rust environment, notebooks, and tests.

## Research contributions

### 1. High-throughput environment

The environment is implemented in Rust and exposed to Python through PyO3 bindings. It supports many parallel worlds, spatial hashing for efficient nearest-entity lookup, and a lightweight visualisation frontend over WebSocket streaming.

### 2. MARL training stack

Training uses MAPPO with centralised training and decentralised execution. The actor uses local observations plus received communication, while the critic uses global state during training to reduce non-stationarity and credit assignment issues.

### 3. Communication protocols

The repository is designed around a protocol interface so communication strategies can be swapped without rewriting the training algorithm. The dissertation evaluates at least four protocol families: no communication, fixed/discrete communication, AIMotherTongue-style communication with frozen discrete bottlenecks, and WhisperWise, which applies reflection strategies directly to on-policy discrete communication.

### 4. Autoencoder-backed language learning

For AIMotherTongue experiments, the code supports VQ-VAE, SQ-VAE, and HQ-VAE-style discrete bottlenecks. These are pretrained offline and then frozen before RL training so the agents learn to use a discrete language rather than discovering one purely on-policy.

## Requirements

Exact package versions should come from the repository's `pyproject.toml`, Rust manifests, and lockfiles. At a high level, the dissertation indicates the following stack:

- Python with PyTorch, NumPy, Pandas, Matplotlib, Jupyter, and Pydantic.
- Rust for the simulation backend, with Rayon for parallelism.
- PyO3 and setuptools-rust for Python/Rust interoperability.
- TypeScript plus p5.js for the visualiser frontend.
- Optional GPU acceleration for training, depending on experiment scale.

## Installation

Create a virtual environment and install the project in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\\Scripts\\activate
pip install -U pip
pip install -e .
```

This editable install should also build the Rust extension if the local Rust toolchain is available, as the dissertation states the Python/Rust build is configured through `pyproject.toml` and setuptools-rust.

## Configuration

The project is driven by configuration files. The dissertation describes separate configuration for:

- Communication protocol settings.
- AIM / autoencoder training.
- Imitation learning.
- MAPPO hyperparameters.
- Training loop settings.
- Simulation environment settings.

Example parameters reported in the dissertation include a vocabulary size of 32, message embedding size of 8, actor hidden size 512, LSTM hidden size 256, discount factor 0.99, GAE 0.95, PPO clipping 0.2, and support for hundreds of parallel worlds.

Before running experiments, review the files in `configs/` and confirm:

1. Environment dimensions, agent visibility radii, and world initialisation.
2. Communication type and any VAE checkpoint paths.
3. Training steps, rollout length, batch sizes, and random seeds.
4. Checkpoint and logging destinations.

## Running experiments

### Train a language model

AIM-style protocols require an offline discrete bottleneck before RL training:

```bash
python -m controller.marl.main --mode train-language
```

### Run imitation learning

The repository supports supervised warm-starting from an expert policy to reduce RL training time:

```bash
python -m controller.marl.main --mode imitate
```

### Run reinforcement learning

Once configs are set, train the MARL agents with:

```bash
python -m controller.marl.main --mode train
```

The dissertation appendix uses `train-language`, `imitate`, and `train`, while the body describes a unified script with modes for language pretraining, imitation learning, and reinforcement learning.

## Evaluation and analysis

The repository includes notebooks and tooling for analysing both task performance and language behaviour. Reported analyses include:

- Training curves across multiple runs.
- Discrete codebook usage and active vocabulary concentration.
- PCA-style projections of codebooks or latent spaces.
- Conditional distributions such as target given code and action given code.
- Memory probing and spatial/cognitive mapping analysis.

These analyses matter because the dissertation evaluates not only reward, but also whether messages become semantically grounded, compressed, and context-aware.

## Adding a new communication protocol

New protocols should be added under `controller/marlcomms/`. The dissertation states that protocol implementations inherit from an abstract communication base class, and the actor dynamically instantiates the selected strategy from configuration.

A practical workflow is:

1. Create a new protocol module in `controller/marlcomms/`.
2. Implement the required interface methods from the base communication class.
3. Register the protocol in the factory or config-driven instantiation path.
4. Add any protocol-specific config entries.
5. Reuse existing evaluation notebooks to compare against baselines.

## Engineering and reproducibility

The project emphasises reproducibility through configuration management, logging, and checkpointing. The dissertation states that configurations are schema-validated with Pydantic, run metrics are flushed to disk, and checkpoints save models, optimisers, and run config for later resumption or comparison.

It also reports strong automated testing for the core infrastructure: 97% function coverage for the Rust environment and 86% line coverage for the Python core infrastructure, while noting that some MARL convergence behaviour is better evaluated experimentally than with unit tests alone.

## Dissertation findings

The dissertation concludes that agents can develop compressed, context-aware communication in this environment, but that AIMotherTongue's semantic bottleneck can hinder learning. It reports that the novel WhisperWise protocol outperformed AIMotherTongue by 40% on the basic task and scaled to more demanding tasks with fewer training steps.

That makes the repository useful in two ways: as a reproduction artifact for the dissertation and as a reusable experimental framework for testing new emergent-communication ideas in cooperative MARL.
