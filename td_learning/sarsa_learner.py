'''
Created on 19 Mar 2022

@author: ucacsjj
'''

import random

from .td_learner_base import TDLearnerBase
# from generalized_policy_iteration.q_grid import QGrid

class SarsaLearner(TDLearnerBase):
    '''
    classdocs
    '''

    def __init__(self, environment):
        TDLearnerBase.__init__(self, environment)

    def initialize(self, q):
        self._q = q

    def _learn_online_from_episode(self):
        
        # Initialize a random state
        S = self._environment.pick_random_start()
        assert(S is not None)
        self._environment.reset(S)
                   
        # Pick the first action
        A = self._q.policy().sample_action(S[0], S[1])

        # Main loop
        done = False
        total_q = 0
        n_steps = 0
           
        while done is False:
    
            S_prime, R, done, info = self._environment.step(A)
    
            # Q3a: Replace with code to implement SARSA
            # if termination state is not reached
            if done is False:

                A_prime = self._q.policy().sample_action(S_prime[0], S_prime[1])
                predict = self._q.value(S[0], S[1], A)
                target = R + self.gamma() * self._q.value(S_prime[0], S_prime[1], A_prime)
                new_q = predict + self.alpha()*(target - predict)
                total_q = total_q + new_q
                n_steps = n_steps + 1
                self.avg_q = total_q / n_steps
            
                self._q.set_value(S[0], S[1], A, new_q)
   
                # Store the state
                S = S_prime
                A = A_prime

    def extract_episode_return(self):
        return self.avg_q