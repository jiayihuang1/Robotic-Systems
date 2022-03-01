"""
Created on 2 Jan 2022

@author: ucacsjj
"""

import math

from .dijkstra_planner import DijkstraPlanner

# This class implements the A* search algorithm

class AStarPlanner(DijkstraPlanner):
    
    def __init__(self, occupancyGrid):
        DijkstraPlanner.__init__(self, occupancyGrid)

    # Q2h:
    # Complete implementation of A*.
