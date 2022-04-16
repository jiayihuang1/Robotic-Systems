'''
Created on 19 Mar 2022

@author: ucacsjj
'''

import random
import numpy as np
from airport.driving_actions import DrivingActionType

from .td_learner_base import TDLearnerBase

class QLearner(TDLearnerBase):
    '''
    classdocs
    '''

    def __init__(self, environment):
        TDLearnerBase.__init__(self, environment)
        
        self._q = None

    def initialize(self, q):
        self._q = q
        
        if self._q.policy() is not None:
            self._q.policy().set_epsilon(self._epsilon)

    def _learn_online_from_episode(self):
        
        # Initialize a random state
        S = self._environment.pick_random_start()
        assert(S is not None)
        self._environment.reset(S)
        
        # Main loop
        done = False
        total_q = 0
        n_steps = 0
        
        while done is False:
                        
            # Sample the action
            A = self._q.policy().sample_action(S[0], S[1])
           
            # Step the environment
            S_prime, R, done, info = self._environment.step(A)

            # Q3b : Replace with code to implement Q-learning
            if done is False:
                predict = self._q.value(S[0], S[1], A)

                # store all action values for this state
                all_action_values = self._q.values_of_actions(S_prime[0], S_prime[1])

                # search through the actions to see which actions give the max action value
                max_actions = (np.where(all_action_values == np.max(all_action_values)))[0]

                # if there is more than one action that gives the max action value, select one randomly
                max_action = max_actions[random.choice(range(max_actions.size))]

                # store the max action value
                max_action_value = all_action_values[max_action]

                target = R + self.gamma() * max_action_value
                new_q = predict + self.alpha() * (target - predict)
                total_q = total_q + new_q
                n_steps = n_steps + 1
                self.avg_q = total_q / n_steps
            
                self._q.set_value(S[0], S[1], A, new_q)
           
                # Store the state
                S = S_prime

    def extract_episode_return(self):
        return self.avg_q


