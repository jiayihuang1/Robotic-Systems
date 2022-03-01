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

    # Insert into queue according to path cost with the least path cost at the start of the queue
    def push_cell_onto_queue(self, cell):

        cell_coords = cell.coords()

        if cell.parent is not None:
            parent_coords = cell.parent.coords()
            cost = self._environment_map.compute_transition_cost(parent_coords, cell_coords)
            cell.path_cost = cell.parent.path_cost + cost
        else:
            cell.path_cost = 0

        self.priorityQueue.put((cell.path_cost, cell))

    # Check if the queue size is zero
    def is_queue_empty(self):
        return self.priorityQueue.empty()

    # Pull from front of the list
    def pop_cell_from_queue(self):
        t = self.priorityQueue.get()
        return t[1]

    # If a cell is visited more than once, replace parent cell with path with the least cost
    def resolve_duplicate(self, cell, parent_cell):
        cell_coords = cell.coords()
        parent_coords = parent_cell.coords()
        additive_cost = self._environment_map.compute_transition_cost(parent_coords, cell_coords)
        predicted_path_cost = parent_cell.path_cost + additive_cost
        if predicted_path_cost < cell.path_cost:
            cell.path_cost = predicted_path_cost
            cell.set_parent(parent_cell)
        pass
