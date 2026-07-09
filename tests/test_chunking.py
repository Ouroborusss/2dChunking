import unittest

import numpy as np

from chunking.grid import create_chunk_grid, find_chunk_index, place_coordinate
from chunking.search import nearest_neighbor


class TestChunking(unittest.TestCase):
    def test_find_chunk_index_within_bounds(self):
        chunk_x, chunk_y, point = find_chunk_index([0, 0], level=4, bounds=(-100, 100))
        self.assertEqual(point.x, 0.0)
        self.assertEqual(point.y, 0.0)
        self.assertGreaterEqual(chunk_x, 0)
        self.assertGreaterEqual(chunk_y, 0)

    def test_place_and_find_nearest_neighbor(self):
        grid = create_chunk_grid(level=2)
        coords = [(-10, -10), (10, 10), (0.5, 0.5)]
        for coord in coords:
            place_coordinate(grid, coord, level=2)

        nearest_point, distance = nearest_neighbor(grid, [0, 0], level=2)
        self.assertAlmostEqual(nearest_point.x, 0.5)
        self.assertAlmostEqual(nearest_point.y, 0.5)
        self.assertLess(distance, 1.0)


if __name__ == "__main__":
  unittest.main()
