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
    def push_cell_onto_queue(self, cell):

        cell_coords = cell.coords()
        goal_coords = self.goal.coords()

        # heuristics
        # euclidean distance between current cell and goal cell
        dX = cell_coords[0] - goal_coords[0]
        dY = cell_coords[1] - goal_coords[1]
        # predicted_cost = math.sqrt(dX * dX + dY * dY)

        # manhattan distance
        predicted_cost = math.sqrt(dX * dX) + math.sqrt(dY * dY)

        if cell.parent is not None:
            parent_coords = cell.parent.coords()
            cost = self._environment_map.compute_transition_cost(parent_coords, cell_coords)
            cell.path_cost = cell.parent.path_cost + cost

        # predicted total path cost
        self.priorityQueue.put((cell.path_cost + predicted_cost, cell))

