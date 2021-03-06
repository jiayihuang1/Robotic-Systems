#!/usr/bin/env python3

'''
Created on 21 Mar 2022

@author: ucacsjj
'''

import math
import numpy as np
import matplotlib.pyplot as plt

from airport.scenarios import *
from airport.airport_driving_environment import AirportDrivingEnvironment
from airport.airport_map_drawer import AirportMapDrawer
from airport.driving_policy_drawer import DrivingPolicyDrawer
from airport.driving_actions import DrivingActionType
from airport.driving_q_grid import DrivingQGrid

from generalized_policy_iteration.value_function_drawer import ValueFunctionDrawer

from td_learning.sarsa_learner import SarsaLearner

if __name__ == '__main__':
    
    # Create test environment
    airport_map, drawer_height = corridor_scenario()
    airport = AirportDrivingEnvironment(airport_map)
    airport.set_nominal_direction_probability(0.8)
    
    q_grid = DrivingQGrid('Sarsa Learner', airport_map)
    q_grid.show()
    
    learner = SarsaLearner(airport)    
    learner.initialize(q_grid)
    learner.set_gamma(1)
    learner.set_alpha(1e-3)
    learner.set_epsilon(1)
            
    # Bind the drawer with the solver
    policy_drawer = DrivingPolicyDrawer(q_grid.policy(), drawer_height)
    learner.set_policy_drawer(policy_drawer)
    
    value_function_drawer = ValueFunctionDrawer(q_grid.value_function(), drawer_height)
    learner.set_value_function_drawer(value_function_drawer)
    
    # Run the learning algorithm.
    ep_R = [0 for a in range(100000)]
    for i in range(100000):
        learner.set_epsilon(1/math.sqrt(i+1))
        learner.learn_online_policy()
        ep_R[i] = learner.extract_episode_return()
        q_grid.show()

    x = np.arange(1, 100001)
    plt.figure()
    plt.plot(x, ep_R)
    plt.xlabel('Number of episodes')
    plt.ylabel('Average return')
    plt.title('Average return over total number of episodes')
    plt.show()
    policy_drawer.wait_for_key_press()



