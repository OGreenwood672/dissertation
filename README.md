# Emergent Communication in Multi-Agent Reinforcement Learning with Assembly Line Environments

Built for a Cambridge Computer Science undergraduate dissertation project.

This open-source program provides a framework for evaluation of emergent protocols in Multi-Agent Systems.

## Getting Started

Before starting, start the virtual environment and build the simulation environment.

`python -m venv .venv`

`.venv\Scripts\activate`

`pip install -e .`

A set of eight configuration files are provided for customisability, and are as follows:

Model Architecture
actor.yaml and critic.yaml define the neural network architectures used by the agents. These control layer structure, hidden sizes, and any recurrent components.

Training Parameters
training.yaml specifies global training settings such as the number of epochs, dataset size, and checkpoint frequency.
mappo.yaml contains MAPPO specific hyperparameters including learning rates, clipping values, and optimisation details.

Communication
comms.yaml determines the active communication protocol and its hyperparameters.
For AIM-based protocols, aim_training.yaml controls the pretraining process of the communication model.

Environment
simulation.yaml defines the environment, including agent visibility, station behaviour, and other simulation-specific rules.

Imitation Learning
imitation.yaml configures the pretraining process using an expert policy, allowing the actor and critic to initialise from a strong baseline before MARL training.

The AI Mother Tongue communication protocol (aim) requires pretraining a language.
To train the language run:

`python -m controller.marl.main --mode train-language`

Once the configuration files have been customised, training starts via the command:

`python -m controller.marl.main --mode train`

MARL famously takes thousands, sometime millions, of training steps for complete convergence on an optimal policy.
To speed this up, initial imitation learning from an optimal agent was developed to pretrain the actor and critic methods.
To use, simply run the following command before training.

`python -m controller.marl.main --mode imitate`

## Customising Protocols

New communication protocols can be added to the folder `controller/marl/comms/`.
The `comms.py` file contains an abstract class which for the new protocol to inherit from.

## Analysis

A large set of evaluation tools are provided to analyse the communication between agents.
This includes analysis of:

- Discrete codebook usage
- Discrete codebook PCA
- Comparing training from multiple runs
- Memory Probing
- Conditional probability of target or action given code
