import time
import numpy as np
import pandas as pd
import tqdm
import math
from csvGenerator import generate_coordinates

BOUNDS = [-10000, 10000]
NUMPOINTS = 500000
CHUNKING_LEVEL = 8
COORD_CSV_FILEPATH = "coordinates.csv"

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

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

def chunkGenerator(level=4):
    #creating a 2D array of empty chunks
    arrayXY = np.zeros((2 ** level, 2 ** level), dtype=object)
    
    #filling the array with empty arrays to be filled with points
    for i in range(2 ** level):
        for j in range(2 ** level):
            arrayXY[i][j] = np.array([], dtype=object)
    
    return arrayXY

#one of the most useful functions ive ever seen. Very nice, good soup.
def num_to_range(num, inMin, inMax, outMin, outMax):
  return outMin + (float(num - inMin) / float(inMax - inMin) * (outMax - outMin))

def chunkFinder(chunkedArrayXY, coord, level):
    #describing the coordinate as a point
    point = Point(coord[0], coord[1])
    point.x = float(coord[0])
    point.y = float(coord[1])
    
    #establishing the bounds of the chunking
    bottomBound = float(BOUNDS[0])
    topBound = float(BOUNDS[1])
    
    #checking if the coordinate is within the bounds
    if point.x >= bottomBound and point.y >= bottomBound and point.x <= topBound and point.y <= topBound:
        #converting the coordinate into a chunk location
        convertedX = num_to_range(point.x, bottomBound, topBound, 0, (2 ** level))
        convertedY = num_to_range(point.y, bottomBound, topBound, 0, (2 ** level))
    elif point.x < bottomBound or point.y < bottomBound or point.x > topBound or point.y > topBound:
        raise ValueError(f"Coordinates ({point.x}, {point.y}) are out of bounds. Must be between {bottomBound} and {topBound}")
    
    #checking if the coordinate is on the bounds
    if point.x == bottomBound:
        convertedX = 0
    
    if point.y == bottomBound:
        convertedY = 0
    
    if point.x == topBound:
        convertedX = num_to_range(point.x, bottomBound, topBound, 0, (2 ** level)) - 1
    
    if point.y == topBound:
        convertedY = num_to_range(point.y, bottomBound, topBound, 0, (2 ** level)) - 1 

    #roudning the chunk location to the nearest integer for indexing
    roundedX = int(convertedX)
    roundedY = int(convertedY)
    
    return roundedX, roundedY, point

def placeCoordInChunks(chunkedArrayXY, coord, level) -> None:
    #finding the where the coordinate should be placed in the chunks
    roundedX, roundedY, point = chunkFinder(chunkedArrayXY, coord, level)

    #appending the coordinate to the chunk
    chunkedArrayXY[roundedX][roundedY] = np.append(chunkedArrayXY[roundedX][roundedY], point)
    
    return roundedX, roundedY, point

def searchAdjacentChunks(chunkedArrayXY, X, Y, searchList = [], radius = 1):
    #converting the radius to a length along the x and y axis
    length = (radius * 2) + 1
    
    #adding the start chunk to the search list if it exists
    if len(searchList) == 0:
        searchList.append(chunkedArrayXY[X][Y])
    
    #searches the top of the radius for non empty chunks then adds them to the search list if they exist
    for i in range(length):
        x = i - (radius)
        try:
            chunk = chunkedArrayXY[X + x][Y + radius]
            if len(chunk) > 0:
                searchList.append(chunk)
        except:
            print(f"Chunk {X + x}, {Y + radius} does not exist")
            continue
    
    #searches the bottom of the radius for non empty chunks then adds them to the search list if they exist
    for i in range(length):
        x = i - (radius)
        try:
            chunk = chunkedArrayXY[X + x][Y - radius]
            if len(chunk) > 0:
                searchList.append(chunk)
        except:
            print(f"Chunk {X + x}, {Y - radius} does not exist")
            continue
    
    #searches the right of the radius for non empty chunks then adds them to the search list if they exist
    for i in range(length):
        y = i - (radius - 1)
        try:
            chunk = chunkedArrayXY[X + radius][Y + y]
            if len(chunk) > 0:
                searchList.append(chunk)
        except:
            print(f"Chunk {X + radius}, {Y + y} does not exist")
            continue
    
    #searches the left of the radius for non empty chunks then adds them to the search list if they exist
    for i in range(length):
        y = i - (radius - 1)
        try:
            chunk = chunkedArrayXY[X - radius][Y + y]
            if len(chunk) > 0:
                searchList.append(chunk)
        except:
            print(f"Chunk {X - radius}, {Y + y} does not exist")
            continue
    
    #if the search only has its own chunk, then it needs to do the search again with radius + 1 recursively
    if len(searchList) <= 1:
        searchList = searchAdjacentChunks(chunkedArrayXY, X, Y, searchList, radius + 1)
    
    return searchList

def nearestNeighbor(chunkedArrayXY, coord, level):
    #finding the chunk that the coordinate is in
    X, Y, point = chunkFinder(chunkedArrayXY, coord, level)

    #filling in the chunkList with the chunks containing points surrounding the coordinate
    chunkList = searchAdjacentChunks(chunkedArrayXY, X, Y)
    
    #making a list of all the points in the chunkList
    pointsList = []
    for chunk in chunkList:
        for point in chunk:
            pointsList.append(point)
    
    #checking the distance of each point in the pointsList to the coordinate then marking the lowest distance point as the nearest point
    nearestPoint = None
    lowestDistance = None
    for point in pointsList:
            distance = math.sqrt((point.x - coord[0])**2 + (point.y - coord[1])**2)
            if lowestDistance is None or distance < lowestDistance:
                lowestDistance = distance
                nearestPoint = point
    
    if nearestPoint is None:
        raise ValueError("No nearest neighbor found")
    
    return nearestPoint, lowestDistance
    
    
    

# Example usage
if __name__ == "__main__":
    #make the coordinates
    generate_coordinates(num_points=NUMPOINTS, bounds=[BOUNDS[0], BOUNDS[1]])
    
    # Load coordinates from the CSV file
    coords = load_coordinates_from_csv(COORD_CSV_FILEPATH)
    level = CHUNKING_LEVEL
    chunkSize = 2 ** level
    xaxis = abs(BOUNDS[0]) + abs(BOUNDS[1])
    yaxis = abs(BOUNDS[0]) + abs(BOUNDS[1])
    
    #checking if the coordinates were loaded
    if coords is None:
        print(f"No coordinates were loaded")
    
    print(f"Loaded {len(coords)} coordinate points")
    
    #generating the chunked array for points to be placed in
    chunkedArrayXY = chunkGenerator(level)
    print(f"{chunkSize} x {chunkSize}: {(chunkSize)*(chunkSize)}: chunks calculated")
    
    #placing the coordinates in the chunks
    print(f"Placing {len(coords)} coordinate points in chunks")
    for coord in tqdm.tqdm(coords, ):
        placeCoordInChunks(chunkedArrayXY, coord, level)
    
    #prints all the chunks in the array
    flag = False
    if flag == True:
        for i in range(2 ** level):
            for j in range(2 ** level):
                tqdm.tqdm.write(f"Chunk {i}, {j}: {chunkedArrayXY[i][j].shape}")
    
    #setting a search coordinate and finding the chunk it is in
    coordSearch = [0, 0]
    X, Y, point = chunkFinder(chunkedArrayXY, coordSearch, level)
    print(f"\nSearch Coordinate: ({coordSearch[0]}, {coordSearch[1]}), Chunks: {X} {Y} {chunkedArrayXY[X][Y].shape}")
    
    #searching for the nearest neighbor to the search coordinate making use of the chunking
    chunkingStart = time.time()
    coord, lowestDistance = nearestNeighbor(chunkedArrayXY, coordSearch, level)
    print(f"\nNearest neighbor to {coordSearch}: {coord.x}, {coord.y}, Distance: {lowestDistance}")
    chunkingEnd = time.time()
    chunkingTime = chunkingEnd - chunkingStart
    
    
    #searching for the nearest neighbor to the search coordinate making use of brute force
    bruteForceStart = time.time()
    lowestDistance = None
    closestCoord = None
    for coord in coords:
        pointcoord = [coord[0], coord[1]]
        distance = math.sqrt((pointcoord[0] - coordSearch[0])**2 + (pointcoord[1] - coordSearch[1])**2)
        if lowestDistance is None or distance < lowestDistance:
            lowestDistance = distance
            closestCoord = coord
    print(f"Nearest neighbor to {coordSearch}: {closestCoord[0]}, {closestCoord[1]}, Distance: {lowestDistance} Via Brute Force")
    bruteForceEnd = time.time()
    bruteTime = bruteForceEnd - bruteForceStart
    
    difference = (1 - (chunkingTime / bruteTime)) * 100
    print(f"\nChunking Time: {round(chunkingTime, 4)} vs Brute Force Time: {round(bruteTime, 4)}")
    print(f"\nChunking is {round(difference, 2)}% faster for searching than brute force")
