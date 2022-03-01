#!/usr/bin/env python3

'''
Created on 3 Feb 2022

@author: ucacsjj
'''

from airport.scenarios import *
from airport.airport_driving_environment import AirportDrivingEnvironment
from generalized_policy_iteration.policy_iterator import PolicyIterator
from generalized_policy_iteration.value_function_drawer import ValueFunctionDrawer
from airport.driving_policy_drawer import DrivingPolicyDrawer
from generalized_policy_iteration.value_iterator import ValueIterator


if __name__ == '__main__':
    
    # Get the map for the scenario
    #airport_map, drawer_height = three_row_scenario()
    airport_map, drawer_height = full_scenario()
    
    # Set up the environment for the robot driving around
    airport_environment = AirportDrivingEnvironment(airport_map)
    
    # Configure the process model
    airport_environment.set_nominal_direction_probability(0.8)
    
    # Create the policy iterator
    policy_solver = PolicyIterator(airport_environment)
    
    # Set up initial state
    policy_solver.initialize()
        
    # Bind the drawer with the solver
    policy_drawer = DrivingPolicyDrawer(policy_solver.policy(), drawer_height)
    policy_solver.set_policy_drawer(policy_drawer)
    
    value_function_drawer = ValueFunctionDrawer(policy_solver.value_function(), drawer_height)
    policy_solver.set_value_function_drawer(value_function_drawer)
        
    # Compute the solution
    v, pi = policy_solver.solve_policy()
    
    # Save screen shot; this is in the current directory
    policy_drawer.save_screenshot("policy_iteration_results.jpg")
    
    # Wait for a key press
    value_function_drawer.wait_for_key_press()
    
    # Q3i: Add code to evaluate value iteration down here.
    # Create the policy iterator
    policy_solver_2 = ValueIterator(airport_environment)

    # Set up initial state
    policy_solver_2.initialize()

    # Bind the drawer with the solver
    policy_drawer_2 = DrivingPolicyDrawer(policy_solver_2.policy(), drawer_height)
    policy_solver_2.set_policy_drawer(policy_drawer_2)

    value_function_drawer_2 = ValueFunctionDrawer(policy_solver_2.value_function(), drawer_height)
    policy_solver_2.set_value_function_drawer(value_function_drawer_2)

    # Compute the solution
    v_2, pi_2 = policy_solver_2.solve_policy()

    # Print number of iterations for policy iteration and value iteration
    print('Policy iteration:')
    print(f'The total number of iterations required for a policy to converge = {policy_solver.total_iterations()}')
    print(f'The total iterations for each evaluation step = {policy_solver.total_iterations_for_each_evaluation()}')
    print('\nValue iteration:')
    print(f'The total number of iterations required for a policy to converge = {policy_solver_2.number_of_iterations()}')

    # Wait for a key press
    value_function_drawer_2.wait_for_key_press()

    
    
