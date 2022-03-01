#!/usr/bin/env python3

"""
Created on 26 Jan 2022

@author: ucacsjj
"""

# This script is needed for Q2i-l

from airport.scenarios import *
from airport.airport_environment import AirportBatteryChargingEnvironment
from airport.actions import ActionType
from airport.charging_policy import ChargingPolicy
from airport.charging_policy_drawer import ChargingPolicyDrawer

if __name__ == '__main__':
    # Get the map
    airport_map = full_scenario()
    
    charging_policy = ChargingPolicy(airport_map, set_random=False)
    
    # Create the environment
    airport_environment = AirportBatteryChargingEnvironment(airport_map)

    # Q2j, k:
    # Implement your algorithm here to use the airport_environment
    # to work out the optimal. Modify the heuristic of the planner and run again.

    for x in range(airport_map.width()):
        for y in range(airport_map.height()):
            if not airport_map.is_obstructed(x, y):
                best_reward = -float('inf')

                for count, charging_station in enumerate(airport_map.all_charging_stations()):
                    action = (ActionType.TELEPORT_ROBOT_TO_NEW_POSITION, (x, y))
                    observation, reward, done, info = airport_environment.step(action)

                    action = (ActionType.DRIVE_ROBOT_TO_NEW_POSITION, charging_station.coords())
                    observation, reward, done, info = airport_environment.step(action)

                    mean = charging_station.params()[0]

                    total_reward = mean + reward
                    if total_reward > best_reward:
                        charging_policy.set_action(x, y, count)
                        best_reward = total_reward

    # Plot the resulting policy
    charging_policy_drawer = ChargingPolicyDrawer(charging_policy, 200)
    charging_policy_drawer.update()
    #charging_policy_drawer.wait_for_key_press()
    
    try:
        input("Press enter in the command window to continue.....")
    except SyntaxError:
        pass
   