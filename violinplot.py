import numpy as np
import gym
import matplotlib.pyplot as plt
from bandit import Bandit
from bandit import BanditEnvironment

def run_bandits(environment, number_of_steps):
    # Making and array to store all the rewards
    # Each column contains all the rewards of each arm
    # This is because of how plt.violinplot works. It takes each column as a separate category
    rewards = np.zeros((number_of_steps, environment.number_of_bandits()))
    for b in range(0, environment.number_of_bandits()):
        for s in range(0, number_of_steps):
            obs, reward, done, info = environment.step(b)
            rewards[s,b] = reward
        print(f'bandit={b}, mean={np.mean(rewards[b,:])}, sigma={np.std(rewards[b,:])}')
    return rewards
        
if __name__ == '__main__':
    # Making a the multi-arm bandit based on table 1
    environment = BanditEnvironment(4)
    environment.set_bandit(0, Bandit(4, 1))
    environment.set_bandit(1, Bandit(4.1, 2))
    environment.set_bandit(2, Bandit(3.9, 1))
    environment.set_bandit(3, Bandit(4.2, 2))

    # Generate violin plot
    rewards = run_bandits(environment, 1000)
    plt.figure()
    plt.violinplot(rewards, showmeans=True, showextrema=True, showmedians=True);
    plt.xlabel('Charging Locaction')
    plt.ylabel('Charging Rates')
    plt.xticks(np.arange(1, environment.number_of_bandits() + 1));
    plt.ion()
    plt.show()
