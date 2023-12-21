# BreakoutAI
An Deep Q Reinforcement Learning model in python TensorFlow that learns how to play Breakout
The script leverages TensorFlow, NumPy, and the OpenAI Gym environment.

## Overview

The model uses a DQN with experience replay and a separate target Q-network to learn optimal strategies for playing Breakout. The approach includes epsilon-greedy policy for exploration vs exploitation, experience replay for efficient learning, and a target network to stabilize learning.

### Key Features

- **Exploration vs Exploitation**: Epsilon-greedy policy with decay.
- **Experience Replay**: Enhances learning by using a memory replay mechanism.
- **Separate Target Q-Network**: Stabilizes the learning process.
- **Custom Neural Network Architecture**: Designed specifically for processing Breakout game states.
- **Action Selection Optimization**: Reduces the action space to improve decision-making.

## Model Architecture

The model consists of a main Q-network and a target Q-network, each with the following layers:
- **Input Layer**: Accepts the state representation from the Breakout game.
- **Hidden Layers**: Two hidden layers, each with 400 neurons and ReLU activation.
- **Output Layer**: Produces Q-values for each possible action.

### Training Process

- **Experience Collection**: The model initially collects experiences by taking random actions in the game environment.
- **Batch Learning**: The model samples from the collected experiences to learn and update the Q-networks.
- **Target Update**: The weights of the target Q-network are periodically updated to match the main Q-network.

## Prerequisites

- Python 3.6+
- TensorFlow
- NumPy
- OpenAI Gym

## Note 

This code was written 5 years ago, some dependencies would be outdated. 
