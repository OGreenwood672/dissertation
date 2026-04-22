# Emergent Communication in Multi-Agent Reinforcement Learning with Assembly Line Environments

Built for a Cambridge Computer Science undergraduate dissertation project.

This open-source program provides a framework for evaluation of emergent protocols in Multi-Agent Systems.

## Getting Started

Before starting, start the virtual environment and build the simulation environment.

`python -m venv .venv`
`.venv\Scripts\activate`
`pip install -e .`

A set of eight configuration files are provided for customisability:

- actor.yaml
  Allows customising of the actor architecture.
- aim_training.yaml
  Allows editing of VAE training parameters.
- comms.yaml
  Determines the current communication protocol their hyperparameters.
- critic.yaml
  Allows customising of the critic architecture.
- imitation.yaml
  Allows editing of imitation learning training parameters.
- mappo.yaml
  Allows editing of MAPPO training parameters.
- simulation.yaml
  Allows editing of simulation parameters. This includes visibility of agents and agent and station behaviours.
- training.yaml
  Machine Learning parameters such as number of epochs, number of training states, how often to save, etc....

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
