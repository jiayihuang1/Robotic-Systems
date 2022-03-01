'''
Created on 29 Jan 2022

@author: ucacsjj
'''

from .dynamic_programming_base import DynamicProgrammingBase
import numpy as np

# This class ipmlements the value iteration algorithm

class ValueIterator(DynamicProgrammingBase):

    def __init__(self, environment):
        DynamicProgrammingBase.__init__(self, environment)
        
        # The maximum number of times the value iteration
        # algorithm is carried out is carried out.
        self._max_optimal_value_function_iterations = 2000
        self._iterations = 0
   
    # Method to change the maximum number of iterations
    def set_max_optimal_value_function_iterations(self, max_optimal_value_function_iterations):
        self._max_optimal_value_function_iterations = max_optimal_value_function_iterations

    #    
    def solve_policy(self):

        # Initialize the drawers
        if self._policy_drawer is not None:
            self._policy_drawer.update()
            
        if self._value_drawer is not None:
            self._value_drawer.update()
        
        self._compute_optimal_value_function()
 
        self._extract_policy()
        
        # Draw one last time to clear any transients which might
        # draw changes
        if self._policy_drawer is not None:
            self._policy_drawer.update()
            
        if self._value_drawer is not None:
            self._value_drawer.update()
        
        return self._v, self._pi

    # Q3h: Finish implementation of the value iterator
    
    def _compute_optimal_value_function(self):

        # Get the environment and map
        environment = self._environment
        map = environment.map()

        # Execute the loop at least once

        iteration = 0

        while True:

            delta = 0

            # Sweep systematically over all the states
            for x in range(map.width()):
                for y in range(map.height()):

                    # We skip obstructions and terminals. If a cell is obstructed,
                    # there's no action the robot can take to access it, so it doesn't
                    # count. If the cell is terminal, it executes the terminal action
                    # state. The value of the value of the terminal cell is the reward.
                    # The reward itself was set up as part of the initial conditions for the
                    # value function.
                    if map.cell(x, y).is_obstruction() or map.cell(x, y).is_terminal():
                        continue

                    # Unfortunately the need to use coordinates is a bit inefficient, due
                    # to legacy code
                    cell = (x, y)

                    # Get the previous value function
                    old_v = self._v.value(x, y)

                    new_v = np.zeros(10)
                    for a in range(10):
                        # Compute p(s',r|s,a)
                        s_prime, r, p = environment.next_state_and_reward_distribution(cell, a)

                        # Sum over the rewards
                        for t in range(len(p)):
                            sc = s_prime[t].coords()
                            new_v[a] = new_v[a] + p[t] * (r[t] + self._gamma * self._v.value(sc[0], sc[1]))


                    # max_v p(s',r|s,a)
                    self._v.set_value(x, y, np.max(new_v))

                    # Update the maximum deviation
                    delta = max(delta, abs(old_v - self._v.value(x,y)))

            # Increment the policy evaluation counter
            iteration += 1

            print(f'Finished policy evaluation iteration {iteration}')

            # Terminate the loop if either the change was very small, or we exceeded
            # the maximum number of iterations.
            if (delta < self._theta) or (iteration >= self._max_optimal_value_function_iterations):
                # Record the total iterations for each evaluation step
                self._iterations = iteration
                break

    def _extract_policy(self):
        # Get the environment and map
        environment = self._environment
        map = environment.map()
        for x in range(map.width()):
            for y in range(map.height()):
                # We skip obstructions and terminals. If a cell is obstructed,
                # there's no action the robot can take to access it, so it doesn't
                # count. If the cell is terminal, it executes the terminal action
                # state. The value of the value of the terminal cell is the reward.
                # The reward itself was set up as part of the initial conditions for the
                # value function.
                if map.cell(x, y).is_obstruction() or map.cell(x, y).is_terminal():
                    continue

                # Unfortunately the need to use coordinates is a bit inefficient, due
                # to legacy code
                cell = (x, y)

                new_v = np.zeros(10)
                # For all actions
                for a in range(10):
                    # Compute p(s',r|s,a)
                    s_prime, r, p = environment.next_state_and_reward_distribution(cell, a)

                    # Sum over the rewards
                    for t in range(len(p)):
                        sc = s_prime[t].coords()
                        new_v[a] = new_v[a] + p[t] * (r[t] + self._gamma * self._v.value(sc[0], sc[1]))

                # argmax_a p(s',r|s,a)
                self._pi.set_action(x, y, np.argmax(new_v))
        pass

    def number_of_iterations(self):
        return self._iterations