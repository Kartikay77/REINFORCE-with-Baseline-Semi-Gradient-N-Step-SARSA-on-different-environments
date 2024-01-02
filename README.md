# Project: Semi Gradient N Step SARSA and REINFORCE with Baseline

This project involves the independent implementation of two reinforcement learning (RL) algorithms, Episodic Semi-Gradient N Step SARSA and REINFORCE with Baseline, carried out by Aparajith Raghuvir and Karthikay Gupta, respectively. The following sections provide an overview of each algorithm and the specific domains they were applied to.

## Overview

Each RL algorithm and its associated hyperparameter tuning have been implemented completely independently and in parallel by the contributors. If you require any code clarifications, please reach out to the respective person involved in writing the code.

### Semi Gradient N Step SARSA

#### Implemented Domains
- 687 Gridworld
- Mountain Car Domain

#### Implementation Details
- Gridworld: Implemented using the standard NumPy framework, providing sufficient efficiency for quick training and iteration.
- Mountain Car: Implemented using the Ivy framework, which can be installed with `pip install ivy`. Ivy allows for efficient computation graph tracing and optimizations, facilitating rapid iteration and training of the Mountain Car environment in a GPU environment. While not strictly necessary, it expedited the code development process and enabled conversion between frameworks such as Torch, TensorFlow, and JAX for autotuning purposes.

#### Additional Experiment (Unreported)
An experiment was conducted with the CartPole domain, but it was deemed unsuccessful and therefore not reported. The experiment involved running CartPole with Monte Carlo return instead of estimated Q values, resulting in excessive variance.


### Contact for Code Clarifications

- Aparajith Raghuvir: Email [araghuvir@umass.edu](mailto:araghuvir@umass.edu)
- Karthikay Gupta: Email [kartikaygupt@umass.edu](mailto:kartikaygupt@umass.edu)

### Post Script:

Additionally, it's worth noting that the initial experiment conducted by Kartikay aimed to explore the feasibility of utilizing an RL agent to operate an autotuner. In this context, an autotuner refers to a tool designed to execute a grid search across machine learning frameworks, compilers, compression techniques, and hardware. The goal is to determine the optimal combination that yields the fastest and most memory-efficient model performance.

However, Kartikay transitioned to the current project due to the complexity involved in implementing the autotuner within the given time constraints. This decision allowed for a more focused and achievable exploration of RL algorithms.
