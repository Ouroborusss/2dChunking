import random
import csv

def generate_coordinates(num_points=1000, output_file="coordinates.csv", bounds=[-1000, 1000]):
    """
    Generate random x and y coordinates between -1000 and 1000 and save them to a CSV file.
    
    Args:
        num_points (int): Number of coordinate pairs to generate
        output_file (str): Name of the output CSV file
    
    Returns:
        list: List of tuples containing (x, y) coordinates
    """
    coordinates = []
    boundsBottom = bounds[0]
    boundsTop = bounds[1]
    
    for _ in range(num_points):
        x = random.uniform(boundsBottom, boundsTop)
        y = random.uniform(boundsBottom, boundsTop)
        coordinates.append((x, y))
    
    # Write coordinates to CSV file
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['x', 'y'])  # Header row
        writer.writerows(coordinates)
    
    print(f"Generated {num_points} coordinate pairs and saved to {output_file}")
    return coordinates

if __name__ == "__main__":
    # Generate 100 coordinate pairs by default
    generate_coordinates(num_points=5000)
    
    # Example of generating a different number of points with a custom filename
    # generate_coordinates(500, "custom_coordinates.csv")
