"""2D chunking nearest-neighbor demo."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import time

import tqdm

from chunking import config
from chunking.grid import create_chunk_grid, find_chunk_index, place_coordinate
from chunking.search import nearest_neighbor
from coordinates import generate_coordinates, load_coordinates_from_csv


def run_demo():
    generate_coordinates(num_points=config.NUM_POINTS, bounds=config.BOUNDS)

    coords = load_coordinates_from_csv(config.COORD_CSV_FILEPATH)
    if coords is None:
        print("No coordinates were loaded")
        return

    level = config.CHUNKING_LEVEL
    chunk_size = 2**level
    print(f"Loaded {len(coords)} coordinate points")
    print(f"{chunk_size} x {chunk_size}: {chunk_size * chunk_size} chunks calculated")

    grid = create_chunk_grid(level)
    print(f"Placing {len(coords)} coordinate points in chunks")
    for coord in tqdm.tqdm(coords):
        place_coordinate(grid, coord, level)

    search_coord = [0, 0]
    chunk_x, chunk_y, _ = find_chunk_index(search_coord, level)
    print(
        f"\nSearch Coordinate: ({search_coord[0]}, {search_coord[1]}), "
        f"Chunks: {chunk_x} {chunk_y} ({len(grid[chunk_x, chunk_y])} points)"
    )

    chunking_start = time.time()
    nearest_point, chunk_distance = nearest_neighbor(grid, search_coord, level)
    chunking_time = time.time() - chunking_start
    print(
        f"\nNearest neighbor to {search_coord}: "
        f"{nearest_point.x}, {nearest_point.y}, Distance: {chunk_distance} via chunking"
    )

    brute_force_start = time.time()
    closest_coord = None
    brute_distance = None
    for coord in coords:
        distance = ((coord[0] - search_coord[0]) ** 2 + (coord[1] - search_coord[1]) ** 2) ** 0.5
        if brute_distance is None or distance < brute_distance:
            brute_distance = distance
            closest_coord = coord
    brute_force_time = time.time() - brute_force_start

    print(
        f"Nearest neighbor to {search_coord}: "
        f"{closest_coord[0]}, {closest_coord[1]}, Distance: {brute_distance} via brute force"
    )

    speedup = (1 - (chunking_time / brute_force_time)) * 100 if brute_force_time else 0
    print(f"\nChunking time: {chunking_time:.4f}s vs brute force time: {brute_force_time:.4f}s")
    print(f"Chunking is {speedup:.2f}% faster for searching than brute force")


if __name__ == "__main__":
    run_demo()
