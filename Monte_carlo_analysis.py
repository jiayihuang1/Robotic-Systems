import numpy as np
import gym
import matplotlib.pyplot as plt
from bandits.bandit import Bandit
from bandits.bandit import BanditEnvironment
from bandits.epsilon_greedy_agent import EpsilonGreedyAgent


def Monte_carlo(agent, number_of_steps, number_of_repeats):
    means = np.zeros(number_of_repeats)
    stds = np.zeros(number_of_repeats)
    for r in range(0, number_of_repeats):
        reward_history = np.zeros(number_of_steps)
        action_history = np.zeros(number_of_steps)
        for s in range(0,number_of_steps):
            action_history[s], reward_history[s] = agent.step()
        means[r] = np.mean(reward_history)
        stds[r] = np.std(reward_history)
    print(f'mean mean = {np.mean(means)}, mean sigma = {np.mean(stds)}')

if __name__ == '__main__':
    # Making a the multi-arm bandit based on table 1
    environment = BanditEnvironment(4)
    environment.set_bandit(0, Bandit(4, 1))
    environment.set_bandit(1, Bandit(4.1, 1))
    environment.set_bandit(2, Bandit(3.9, 1))
    environment.set_bandit(3, Bandit(4.2, 1))

    # Set number of steps
    number_of_steps = 1000
    # Set number of repeats
    number_of_repeats = 10
    # Set epsilon
    epsilon = 0.01
    print(f'epsilon = {epsilon}')
    # Create agent
    agent = EpsilonGreedyAgent(environment, epsilon)

    # Monte Carlo analysis
    Monte_carlo(agent, number_of_steps, number_of_repeats)

