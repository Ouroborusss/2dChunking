from csvGenerator import generate_coordinates
from hashMap import HashMap
import numpy as np
import pandas as pd
import time
import tqdm

BOUNDS = [-100000, 100000]
NUMPOINTS = 1000000
COORD_CSV_FILEPATH = "coordinates.csv"

def load_coordinates_from_csv(file_path='coordinates.csv'):
    """
    Load coordinates from a CSV file and convert them to a NumPy array.
    
    Parameters:
    -----------
    file_path : str, optional
        Path to the CSV file containing coordinates (default: 'coordinates.csv')
        
    Returns:
    --------
    numpy.ndarray
        A 2D NumPy array where each row represents an (x, y) coordinate point
    """
    try:
        # Read the CSV file using pandas
        df = pd.read_csv(file_path)
        
        # Convert to NumPy array
        coordinates = df.to_numpy()
        
        return coordinates
    except Exception as e:
        print(f"Error loading coordinates: {e}")
        return None

def remap( x, oMin, oMax, nMin, nMax ):

    #range check
    if oMin == oMax:
        print("Warning: Zero input range")
        return None

    if nMin == nMax:
        print("Warning: Zero output range")
        return None

    #check reversed input range
    reverseInput = False
    oldMin = min( oMin, oMax )
    oldMax = max( oMin, oMax )
    if not oldMin == oMin:
        reverseInput = True

    #check reversed output range
    reverseOutput = False   
    newMin = min( nMin, nMax )
    newMax = max( nMin, nMax )
    if not newMin == nMin :
        reverseOutput = True

    portion = (x-oldMin)*(newMax-newMin)/(oldMax-oldMin)
    if reverseInput:
        portion = (oldMax-x)*(newMax-newMin)/(oldMax-oldMin)

    result = portion + newMin
    if reverseOutput:
        result = newMax - portion

    return result

if __name__ == "__main__":
    #make the coordinates
    print("Generating coordinates...")
    generate_coordinates(num_points=NUMPOINTS, bounds=[BOUNDS[0], BOUNDS[1]])
    
    # Load coordinates from the CSV file
    print("Loading coordinates...")
    coords = load_coordinates_from_csv(COORD_CSV_FILEPATH)
    
    #checking if the coordinates were loaded
    if coords is None:
        print(f"No coordinates were loaded")
    
    print("Creating hash map...")
    hash_map = HashMap()
    
    print("Inserting coordinates...")
    start = time.time()
    index = 0
    for coord in tqdm.tqdm(coords):
        hash_map[index] = coord
        index += 1
    insert_time = time.time() - start
    print(f"Time to insert {NUMPOINTS} items: {insert_time} seconds")
    print()
    print(hash_map.get_metrics())
    
    print()
    searchPlace = len(coords) // 2
    x = coords[searchPlace]
    print(f"Search coordinate at index {searchPlace}: {x[0]}, {x[1]}")
    
    print()
    print("Searching for coordinate brute force...")
    #make sure to use a numpy array for the search coordinate in this instance since its hashable and we're using a hash map
    search_coord = coords[searchPlace]
    start = time.time()
    for coord in tqdm.tqdm(coords):
        if coord[0] == search_coord[0] and coord[1] == search_coord[1]:
            tqdm.tqdm.write(f"Found coordinate: {coord}")
            break
    search_time = time.time() - start
    print(f"Time to find coordinate (brute force): {search_time} seconds")
    
    print()
    print("Searching for coordinate in hash map...")
    start = time.time()
    val = hash_map.get_keys_by_value(search_coord)
    print(f"Found {len(val)} key/s for coordinate: {coord}")
    search_hash_time = time.time() - start
    print(f"Time to find coordinate (hash search): {search_hash_time} seconds")
    
    print()
    if val:  # Check if we found any keys
        for key in val:
            print(f"Value of key {key}: {hash_map[key]}")
    hashrate = search_time / search_hash_time
    hashrte = search_hash_time / insert_time
    print(f"Hashed search was {hashrate:.2f} times faster than brute force")
    print(f"Hashed search was {(1 - hashrte) * 100}% faster than brute force") 