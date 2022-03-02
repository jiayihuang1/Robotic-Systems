import numpy as np
from bandits.bandit import Bandit
from bandits.bandit import BanditEnvironment
from bandits.epsilon_greedy_agent import EpsilonGreedyAgent
import matplotlib.pyplot as plt

if __name__ == '__main__':

    environment = BanditEnvironment(4)
    environment.set_bandit(0, Bandit(4, 1))
    environment.set_bandit(1, Bandit(4.1, 1))
    environment.set_bandit(2, Bandit(3.9, 1))
    environment.set_bandit(3, Bandit(4.2, 1))
    epsilon = 0.01#change here
    agent = EpsilonGreedyAgent(environment, epsilon)

    number_of_steps = 1000
    num_simulation = 500
    reward_history = np.zeros(number_of_steps)
    action_history = np.zeros(number_of_steps)
    p_optimal = np.zeros(num_simulation)
    pos = np.zeros((num_simulation, 4))
    for i in range(num_simulation):
        for p in range(0, number_of_steps):
            action_history[p], reward_history[p] = agent.step()

        occurrence3 = np.count_nonzero(action_history == 3)
        occurrence2 = np.count_nonzero(action_history == 2)
        occurrence1 = np.count_nonzero(action_history == 1)
        occurrence0 = np.count_nonzero(action_history == 0)
        occurrences = [occurrence0, occurrence1, occurrence2, occurrence3]
        number_of_pull = len(action_history)
        p_optimal[i] = np.divide(occurrence3, len(action_history))
        pos[i, :] = np.divide(occurrences, number_of_pull)

    p_expect = 1 - epsilon * (1 - 1 / environment.number_of_bandits())
    plt.figure(1)
    a = np.linspace(round(min(p_optimal), 3), round(max(p_optimal), 3), 80)
    plt.hist(p_optimal, bins=a)
    plt.axvline(x=p_expect, color='red')
    plt.axvline(x=p_optimal.mean(), color='green')
    plt.xlabel('Count')
    plt.ylabel('Possibility')
    plt.title('distribution of P(optimal)')
    plt.figure(2)
    plt.violinplot(pos, showmeans=True, showextrema=True, showmedians=True)
    plt.xlabel('Charging Location')
    plt.ylabel('Possibility of pick up')
    plt.xticks(np.arange(1, environment.number_of_bandits() + 1))
    plt.title('violin plot of possibility')
    plt.ion()
    plt.show()
