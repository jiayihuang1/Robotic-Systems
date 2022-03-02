import numpy as np
from bandits.bandit import Bandit
from bandits.bandit import BanditEnvironment
from bandits.epsilon_greedy_agent import EpsilonGreedyAgent
from bandits.damped_epsilon_greedy_agent import DampedEpsilonGreedyAgent
import matplotlib.pyplot as plt
from bandits.performance_measures import compute_percentage_of_optimal_actions_selected
from bandits.performance_measures import compute_regret

if __name__ == '__main__':
    # Making a the multi-arm bandit based on table 1
    environment = BanditEnvironment(4)
    environment.set_bandit(0, Bandit(4, 1))
    environment.set_bandit(1, Bandit(4.1, 1))
    environment.set_bandit(2, Bandit(3.9, 1))
    environment.set_bandit(3, Bandit(4.2, 1))
    # set steps and epsilon
    epsilon = [0.02, 0.05, 0.1, 0.2]
    number_of_steps = 100000
    reward_history = np.zeros((len(epsilon), number_of_steps))
    action_history = np.zeros((len(epsilon), number_of_steps))
    reward_history_d = np.zeros((len(epsilon), number_of_steps))#d for damped
    action_history_d = np.zeros((len(epsilon), number_of_steps))
    label = ['ϵ=0.02', 'ϵ=0.05', 'ϵ=0.1', 'ϵ=0.2']

    for e in range(0, len(epsilon)):
        #set multi-arm agent
        agent = EpsilonGreedyAgent(environment, epsilon[e])
        agent_d = DampedEpsilonGreedyAgent(environment, epsilon[e])
        for p in range(0, number_of_steps):
            action_history[e, p], reward_history[e, p] = agent.step()
            action_history_d[e, p], reward_history_d[e, p] = agent_d.step()

        # plot optimal percentage and cumulative regret
        plt.figure(1)
        percentage_correct_actions = compute_percentage_of_optimal_actions_selected(environment, action_history[e])
        plt.plot(percentage_correct_actions, label=label[e])
        plt.figure(2)
        percentage_correct_actions_d = compute_percentage_of_optimal_actions_selected(environment, action_history_d[e])
        plt.plot(percentage_correct_actions_d, label=label[e])
        plt.figure(3)
        regret = compute_regret(environment, reward_history[e])
        plt.plot(regret, label=label[e])
        plt.figure(4)
        regret_d = compute_regret(environment, reward_history_d[e])
        plt.plot(regret_d, label=label[e])

    plt.figure(1)
    plt.legend()
    plt.xlabel('Sample number')
    plt.ylabel('Percentage optimal action')
    plt.title('optimal percentage of ϵ-greedy strategy')
    plt.figure(2)
    plt.legend()
    plt.xlabel('Sample number')
    plt.ylabel('Percentage optimal action')
    plt.title('optimal percentage of damped ϵ-greedy strategy')
    plt.figure(3)
    plt.legend()
    plt.xlabel('Sample number')
    plt.ylabel('Regret')
    plt.title('cumulative regret of ϵ-greedy strategy')
    plt.figure(4)
    plt.legend()
    plt.xlabel('Sample number')
    plt.ylabel('Regret')
    plt.title('cumulative regret of damped ϵ-greedy strategy')
    plt.show()
