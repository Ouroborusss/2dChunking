from hashMap import HashMap
import numpy as np

def test_new_features():
    # Create a hash map
    hash_map = HashMap(initial_size=10)
    
    # Test basic operations with metrics
    print("\n1. Basic Operations with Metrics:")
    for i in range(5):
        hash_map[f"key_{i}"] = i
    print(f"Initial state: {hash_map}")
    print(f"Metrics: {hash_map.get_metrics()}")
    
    # Test iteration
    print("\n2. Iteration:")
    print("Keys:", [key for key in hash_map])
    print("Values:", hash_map.get_values())
    print("Items:", hash_map.get_items())
    
    # Test copy and update
    print("\n3. Copy and Update:")
    copy_map = hash_map.copy()
    copy_map.update({"new_key": 100, "another_key": 200})
    print(f"Original: {hash_map}")
    print(f"Copy after update: {copy_map}")
    
    # Test shrinking
    print("\n4. Shrinking:")
    print(f"Before clearing - {repr(hash_map)}")
    hash_map.clear()
    print(f"After clearing - {repr(hash_map)}")
    
    # Test collisions
    print("\n5. Collision Handling:")
    # Add items that might cause collisions
    for i in range(10):
        hash_map[f"collision_key_{i*10}"] = i
    print(f"Collision metrics: {hash_map.get_metrics()}")

def test_value_search():
    print("\nTesting Value-based Search Features:")
    
    # Create a hash map with some duplicate values
    hash_map = HashMap(initial_size=10)
    hash_map["a"] = 1
    hash_map["b"] = 2
    hash_map["c"] = 1  # Duplicate value
    hash_map["d"] = 3
    hash_map["e"] = 2  # Duplicate value
    
    print("\n1. Value Existence:")
    print("Contains value 1:", hash_map.contains_value(1))
    print("Contains value 4:", hash_map.contains_value(4))
    
    print("\n2. Keys by Value:")
    print("Keys for value 1:", hash_map.get_keys_by_value(1))
    print("Keys for value 2:", hash_map.get_keys_by_value(2))
    print("Keys for value 3:", hash_map.get_keys_by_value(3))
    
    print("\n3. All Unique Values:")
    print("All values:", hash_map.get_all_values())
    
    # Test with typed array
    print("\n4. Typed Array Test:")
    int_map = HashMap(initial_size=10, value_dtype=np.int32)
    int_map["x"] = 1
    int_map["y"] = 2
    int_map["z"] = 1
    print("Contains value 1:", int_map.contains_value(1))
    print("Keys for value 1:", int_map.get_keys_by_value(1))
    print("All values:", int_map.get_all_values())
    
    # Test with array values
    print("\n5. Array Value Test:")
    array_map = HashMap(initial_size=10)
    coord1 = np.array([1, 2])
    coord2 = np.array([3, 4])
    coord3 = np.array([1, 2])  # Same as coord1
    array_map["point1"] = coord1
    array_map["point2"] = coord2
    array_map["point3"] = coord3
    
    print("Contains coord1:", array_map.contains_value(coord1))
    print("Keys for coord1:", array_map.get_keys_by_value(coord1))
    print("Keys for coord2:", array_map.get_keys_by_value(coord2))
    print("All values:", array_map.get_all_values())

if __name__ == "__main__":
    test_new_features()
    test_value_search() 