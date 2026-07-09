"""Shared utilities for loading and generating coordinate data."""

import csv
import random

import numpy as np
import pandas as pd


def generate_coordinates(num_points=1000, output_file="coordinates.csv", bounds=(-1000, 1000)):
    """Generate random coordinate pairs and save them to a CSV file."""
    low, high = bounds
    coordinates = [(random.uniform(low, high), random.uniform(low, high)) for _ in range(num_points)]

    with open(output_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["x", "y"])
        writer.writerows(coordinates)

    print(f"Generated {num_points} coordinate pairs and saved to {output_file}")
    return coordinates


def load_coordinates_from_csv(file_path="coordinates.csv"):
    """Load coordinates from a CSV file as a NumPy array."""
    try:
        return pd.read_csv(file_path).to_numpy()
    except Exception as exc:
        print(f"Error loading coordinates: {exc}")
        return None
