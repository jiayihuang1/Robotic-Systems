"""
Created on 2 Jan 2022

@author: ucacsjj
"""

from math import sqrt
from queue import PriorityQueue

from .planner_base import PlannerBase

class DijkstraPlanner(PlannerBase):

    # This implements Dijkstra. The priority queue is the path length
    # to the current position.
    
    def __init__(self, occupancyGrid):
        PlannerBase.__init__(self, occupancyGrid)
        self.priorityQueue = PriorityQueue()


    # Q2d:
    # Modify this class to finish implementing Dijkstra
        
