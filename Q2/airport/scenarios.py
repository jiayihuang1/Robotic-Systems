"""
Created on 25 Jan 2022

@author: ucacsjj
"""

# from .airport_map import MapCell
from .airport_map import MapCellType
from .airport_map import AirportMap


# Helper function
def _set_block_to_single_type(airport_map, cell_type, start_coords, end_coords):
    for x in range(start_coords[0], end_coords[0] + 1):
        for y in range(start_coords[1], end_coords[1] + 1):
            airport_map.set_cell_type(x, y, cell_type)


# This scenario can be used to test the different traversability costs
def test_traversability_costs_scenario():
    airport_map = AirportMap("Test Traversabilty Map", 15, 15)

    for x in range(0, 14):
        airport_map.set_wall(x, 7)

    airport_map.add_secret_door(7, 7)

    return airport_map


def test_nearest_charging_station_scenario():
    airport_map = AirportMap("Test Nearest Charging Station", 45, 15)
    airport_map.add_charging_station(0, 7, 1, 1)
    airport_map.add_charging_station(44, 7, 1, 1)

    return airport_map


def mini_scenario():
    # Create the map
    airport_map = AirportMap("Mini Scenario", 15, 15)

    # Create the wall on either side and the customs area
    for x in range(0, 15):
        airport_map.set_wall(x, 7)

    for x in range(5, 7):
        airport_map.set_customs_area(x, 7)

    airport_map.add_charging_station(4, 4, 1, 1)

    airport_map.add_secret_door(14, 7)

    airport_map.add_toilet(4, 1)

    return airport_map


def medium_test_scenario():
    airport_map = AirportMap("Full Scenario", 30, 20)

    # The wall separating the two areas, including the customs area
    # and the secret door
    _set_block_to_single_type(airport_map, MapCellType.WALL, (0, 8), (30, 10))
    _set_block_to_single_type(airport_map, MapCellType.CUSTOMS_AREA, (10, 8), (20, 10))
    _set_block_to_single_type(airport_map, MapCellType.SECRET_DOOR, (29, 8), (29, 10))

    # The reclaim areas; need to be avoided
    airport_map.add_reclaim_area(13, 15, 1, 1)
    airport_map.add_reclaim_area(15, 15, 1, 1)
    airport_map.add_reclaim_area(17, 15, 1, 1)
    airport_map.add_reclaim_area(19, 15, 1, 1)

    # The toilets. These generate rubbish to be collected
    airport_map.add_toilet(1, 11)
    airport_map.add_toilet(1, 7)
    airport_map.add_toilet(19, 0)
    airport_map.add_toilet(28, 11)

    # These charge the robot back up again
    airport_map.add_charging_station(0, 19, 15, 1)
    airport_map.add_charging_station(29, 19, 15, 1)
    airport_map.add_charging_station(18, 0, 30, 1)
    airport_map.add_charging_station(29, 0, 40, 1)

    return airport_map


def full_scenario():
    airport_map = AirportMap("Full Scenario", 60, 40)

    # The wall separating the two areas, including the customs area
    # and the secret door
    _set_block_to_single_type(airport_map, MapCellType.WALL, (0, 18), (59, 20))
    _set_block_to_single_type(airport_map, MapCellType.CUSTOMS_AREA, (25, 18), (35, 20))
    _set_block_to_single_type(airport_map, MapCellType.SECRET_DOOR, (59, 18), (59, 20))

    # The reclaim areas
    airport_map.add_rubbish_bin(2, 33)
    _set_block_to_single_type(airport_map, MapCellType.BAGGAGE_CLAIM, (5, 30), (8, 36))
    airport_map.add_rubbish_bin(11, 33)
    _set_block_to_single_type(airport_map, MapCellType.BAGGAGE_CLAIM, (15, 28), (18, 39))
    airport_map.add_rubbish_bin(22, 38)
    _set_block_to_single_type(airport_map, MapCellType.BAGGAGE_CLAIM, (25, 28), (28, 39))
    airport_map.add_rubbish_bin(31, 38)
    _set_block_to_single_type(airport_map, MapCellType.BAGGAGE_CLAIM, (35, 28), (38, 39))
    airport_map.add_rubbish_bin(41, 38)
    _set_block_to_single_type(airport_map, MapCellType.BAGGAGE_CLAIM, (45, 28), (48, 39))
    airport_map.add_rubbish_bin(51, 33)
    _set_block_to_single_type(airport_map, MapCellType.BAGGAGE_CLAIM, (55, 30), (58, 36))

    # The bins in the reclaim areas

    # Add the horizontal chairs with bins at either end
    for i in range(5):
        y_coord = 2 + i * 3
        _set_block_to_single_type(airport_map, MapCellType.CHAIR, (5, y_coord), (18, y_coord))
        airport_map.add_rubbish_bin(4, y_coord)
        airport_map.add_rubbish_bin(19, y_coord)

    # Add the vertical chairs with bins at either end
    for i in range(5):
        x_coord = 42 + i * 3
        _set_block_to_single_type(airport_map, MapCellType.CHAIR, (x_coord, 2), (x_coord, 14))
        airport_map.add_rubbish_bin(x_coord, 1)
        airport_map.add_rubbish_bin(x_coord, 15)

    # The toilets. These generate rubbish to be collected
    airport_map.add_toilet(0, 21)
    airport_map.add_toilet(0, 17)
    airport_map.add_toilet(38, 0)
    airport_map.add_toilet(58, 21)

    # These charge the robot back up again
    airport_map.add_charging_station(1, 38, 15, 1)
    airport_map.add_charging_station(58, 38, 15, 1)
    airport_map.add_charging_station(36, 0, 30, 1)
    airport_map.add_charging_station(59, 0, 40, 1)

    return airport_map
