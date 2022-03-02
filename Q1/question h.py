import numpy as np
from bandits.bandit import Bandit
from bandits.bandit import BanditEnvironment
from bandits.upper_confidence_bound_agent import UpperConfidenceBoundAgent
import matplotlib.pyplot as plt
from bandits.performance_measures import compute_percentage_of_optimal_actions_selected
from bandits.performance_measures import compute_regret

if __name__ == '__main__':
    # Making the multi-arm bandit based on table 1
    environment = BanditEnvironment(4)
    environment.set_bandit(0, Bandit(4, 1))
    environment.set_bandit(1, Bandit(4.1, 1))
    environment.set_bandit(2, Bandit(3.9, 1))
    environment.set_bandit(3, Bandit(4.2, 1))
    # set steps and degree of UCB
    degree = [0.05, 0.5, 1, 2, 5]
    number_of_steps = 100000
    reward_history = np.zeros((len(degree), number_of_steps))
    action_history = np.zeros((len(degree), number_of_steps))
    label = ['0.05', '0.5', '1', '2', '5']

    for d in range(0, len(degree)):
        # set multi-arm agent
        agent = UpperConfidenceBoundAgent(environment, degree[d])
        for p in range(0, number_of_steps):
            action_history[d, p], reward_history[d, p] = agent.step()
        # plot optimal percentage, cumulative regret and pull history
        plt.figure(1)
        percentage_correct_actions = compute_percentage_of_optimal_actions_selected(environment, action_history[d])
        plt.plot(percentage_correct_actions, label=label[d])
        plt.figure(2)
        regret = compute_regret(environment, reward_history[d])
        plt.plot(regret, label=label[d])
        plt.figure(3, figsize=(12, 5))
        plt.plot(np.linspace(500, 1001, 500), action_history[d, 500:1000], label=label[d])


    plt.figure(1)
    plt.legend()
    plt.xlabel('Sample number')
    plt.ylabel('Percentage optimal action')
    plt.title('optimal percentage of UCB strategy')
    plt.figure(2)
    plt.legend()
    plt.xlabel('Sample number')
    plt.ylabel('Regret')
    plt.title('cumulative regret of UCB strategy')
    plt.figure(3, figsize=(9, 3))
    plt.xlabel('Sample number')
    plt.ylabel('action')
    plt.title('History of arm pulled')
    plt.legend()
    plt.show()
