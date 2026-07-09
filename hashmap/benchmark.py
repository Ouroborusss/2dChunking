"""HashMap benchmark comparing hash-based and brute-force coordinate lookup."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import time

import tqdm

from chunking import config as chunking_config
from coordinates import generate_coordinates, load_coordinates_from_csv
from hashmap import HashMap


def run_benchmark(num_points=1_000_000):
    print("Generating coordinates...")
    generate_coordinates(num_points=num_points, bounds=chunking_config.BOUNDS)

    print("Loading coordinates...")
    coords = load_coordinates_from_csv(chunking_config.COORD_CSV_FILEPATH)
    if coords is None:
        print("No coordinates were loaded")
        return

    print("Creating hash map...")
    hash_map = HashMap()

    print("Inserting coordinates...")
    start = time.time()
    for index, coord in enumerate(tqdm.tqdm(coords)):
        hash_map[index] = coord
    insert_time = time.time() - start
    print(f"Time to insert {num_points} items: {insert_time:.4f} seconds")
    print(hash_map.get_metrics())

    search_index = len(coords) // 2
    search_coord = coords[search_index]
    print(f"\nSearch coordinate at index {search_index}: {search_coord[0]}, {search_coord[1]}")

    print("\nSearching for coordinate brute force...")
    start = time.time()
    for coord in tqdm.tqdm(coords):
        if coord[0] == search_coord[0] and coord[1] == search_coord[1]:
            break
    brute_force_time = time.time() - start
    print(f"Time to find coordinate (brute force): {brute_force_time:.4f} seconds")

    print("\nSearching for coordinate in hash map...")
    start = time.time()
    matching_keys = hash_map.get_keys_by_value(search_coord)
    hash_search_time = time.time() - start
    print(f"Found {len(matching_keys)} key(s) for coordinate")
    print(f"Time to find coordinate (hash search): {hash_search_time:.4f} seconds")

    if matching_keys:
        for key in matching_keys:
            print(f"Value of key {key}: {hash_map[key]}")

    if hash_search_time > 0:
        speedup = brute_force_time / hash_search_time
        print(f"\nHash search was {speedup:.2f}x faster than brute force")


if __name__ == "__main__":
    run_benchmark()
