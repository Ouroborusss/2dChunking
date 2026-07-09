"""Nearest-neighbor search using adjacent spatial chunks."""

import math

from chunking.grid import find_chunk_index


def _collect_ring_chunks(grid, center_x, center_y, radius):
    """Collect non-empty chunks along the perimeter of a square search radius."""
    chunks = []
    grid_size = grid.shape[0]

    for offset in range(-radius, radius + 1):
        top_y = center_y + radius
        bottom_y = center_y - radius
        left_x = center_x - radius
        right_x = center_x + radius

        for x, y in (
            (center_x + offset, top_y),
            (center_x + offset, bottom_y),
            (right_x, center_y + offset),
            (left_x, center_y + offset),
        ):
            if 0 <= x < grid_size and 0 <= y < grid_size and grid[x, y]:
                chunks.append(grid[x, y])

    return chunks


def search_adjacent_chunks(grid, center_x, center_y, radius=1):
    """Find nearby non-empty chunks, expanding the radius until neighbors are found."""
    chunks = [grid[center_x, center_y]] if grid[center_x, center_y] else []
    chunks.extend(_collect_ring_chunks(grid, center_x, center_y, radius))

    if len(chunks) <= 1 and radius < grid.shape[0]:
        return search_adjacent_chunks(grid, center_x, center_y, radius + 1)

    return chunks


def nearest_neighbor(grid, coord, level):
    """Find the nearest point to a coordinate using chunked spatial search."""
    center_x, center_y, _ = find_chunk_index(coord, level)
    nearby_chunks = search_adjacent_chunks(grid, center_x, center_y)

    points = [point for chunk in nearby_chunks for point in chunk]
    if not points:
        raise ValueError("No nearest neighbor found")

    target_x, target_y = coord[0], coord[1]
    nearest_point = None
    lowest_distance = None

    for point in points:
        distance = math.hypot(point.x - target_x, point.y - target_y)
        if lowest_distance is None or distance < lowest_distance:
            lowest_distance = distance
            nearest_point = point

    return nearest_point, lowest_distance
