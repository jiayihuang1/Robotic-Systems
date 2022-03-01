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

    #Discount factor
    gamma = 1
    policy_solver.set_gamma(gamma)

    # Set up initial state
    policy_solver.initialize()
        
    # Bind the drawer with the solver
    policy_drawer = DrivingPolicyDrawer(policy_solver.policy(), drawer_height)
    policy_solver.set_policy_drawer(policy_drawer)
    
    value_function_drawer = ValueFunctionDrawer(policy_solver.value_function(), drawer_height)
    policy_solver.set_value_function_drawer(value_function_drawer)
        
    # Compute the solution
    v, pi = policy_solver.solve_policy()

    # Print number of iterations
    total_iterations = policy_solver.total_iterations()
    total_iterations_for_each_evaluation = policy_solver.total_iterations_for_each_evaluation()
    print(f'The total number of iterations required for a policy to converge = {total_iterations}')
    print(f'The total iterations for each evaluation step = {total_iterations_for_each_evaluation}')
    
    # Save screen shot; this is in the current directory
    policy_drawer.save_screenshot("policy_iteration_results.jpg")
    
    # Wait for a key press
    value_function_drawer.wait_for_key_press()
    value_function_drawer.wait_for_key_press()
    value_function_drawer.wait_for_key_press()
