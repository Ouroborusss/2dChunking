"""Spatial grid utilities for partitioning coordinates into chunks."""

import math

import numpy as np

from chunking.config import BOUNDS


class Point:
    __slots__ = ("x", "y")

    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)


def remap(value, in_min, in_max, out_min, out_max):
    """Map a value from one numeric range to another."""
    return out_min + (float(value - in_min) / float(in_max - in_min) * (out_max - out_min))


def create_chunk_grid(level):
    """Create a 2D grid of empty point buckets."""
    grid_size = 2**level
    grid = np.empty((grid_size, grid_size), dtype=object)
    for i in range(grid_size):
        for j in range(grid_size):
            grid[i, j] = []
    return grid


def find_chunk_index(coord, level, bounds=BOUNDS):
    """Return the grid indices for a coordinate."""
    x, y = float(coord[0]), float(coord[1])
    low, high = float(bounds[0]), float(bounds[1])

    if not (low <= x <= high and low <= y <= high):
        raise ValueError(f"Coordinates ({x}, {y}) are out of bounds. Must be between {low} and {high}")

    grid_max = 2**level
    converted_x = remap(x, low, high, 0, grid_max)
    converted_y = remap(y, low, high, 0, grid_max)

    if x == low:
        converted_x = 0
    if y == low:
        converted_y = 0
    if x == high:
        converted_x = grid_max - 1
    if y == high:
        converted_y = grid_max - 1

    return int(converted_x), int(converted_y), Point(x, y)


def place_coordinate(grid, coord, level):
    """Place a coordinate into its corresponding chunk."""
    chunk_x, chunk_y, point = find_chunk_index(coord, level)
    grid[chunk_x, chunk_y].append(point)
    return chunk_x, chunk_y, point
