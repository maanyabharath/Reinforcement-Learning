# Reinforcement-Learning

Multi-Armed Bandit (MAB) model to dynamically select optimal mutation and crossover rates

Genetic Algorithm: Evolves a population of individuals to approximate a target polynomial function by optimizing their genes.
Multi-Armed Bandit: Uses softmax-based action selection to probabilistically choose mutation and crossover rates, adapting to the GA's performance feedback.

Represents a solution with genes initialized to random values.
Calculates fitness based on how well the polynomial (genes) matches the target function.

Bandit Arm (Arm)
Represents a combination of mutation and crossover rates.
Tracks cumulative performance to guide selection.

K-Armed Bandit (KArmBandit)
Selects arms using softmax for balanced exploration-exploitation.
Updates arm values based on rewards from the GA.

Usage
Initialize GA and MAB, run optimization, and identify the best mutation and crossover rates.
