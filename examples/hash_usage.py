"""Example usage of the HashMap class."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from hashmap import HashMap


def main():
    print("Creating a new HashMap instance for strings...")
    hash_map = HashMap(initial_size=10)

    print("\n1. Basic Operations:")
    hash_map.put("name", "John")
    hash_map["age"] = "30"
    hash_map["city"] = "New York"
    print(f"Name: {hash_map.get('name')}")
    print(f"Age: {hash_map['age']}")
    print(f"City: {hash_map['city']}")

    print("\nCreating a new HashMap instance for integers...")
    int_hash_map = HashMap(initial_size=10, value_dtype=np.int32)

    print("\n2. Integer Operations:")
    int_hash_map["age"] = 30
    int_hash_map["count"] = 42
    int_hash_map["score"] = 100
    print(f"Age: {int_hash_map['age']}")
    print(f"Count: {int_hash_map['count']}")
    print(f"Score: {int_hash_map['score']}")

    print("\n3. Different Types of Keys:")
    int_hash_map[42] = 1000
    int_hash_map[3.14] = 2000
    int_hash_map[("tuple", "key")] = 3000
    print(f"Number key value: {int_hash_map[42]}")
    print(f"Float key value: {int_hash_map[3.14]}")
    print(f"Tuple key value: {int_hash_map[('tuple', 'key')]}")

    print("\n4. Checking Existence and Length:")
    print(f"Is 'age' in int_hash_map? {'age' in int_hash_map}")
    print(f"Is 'nonexistent' in int_hash_map? {'nonexistent' in int_hash_map}")
    print(f"Current number of items: {len(int_hash_map)}")

    print("\n5. Removing Items:")
    print(f"Removing 'age': {int_hash_map.remove('age')}")
    print(f"Is 'age' still in int_hash_map? {'age' in int_hash_map}")
    print(f"Current number of items: {len(int_hash_map)}")

    print("\n6. Error Handling:")
    try:
        print(int_hash_map["nonexistent"])
    except KeyError as exc:
        print(f"Expected error: {exc}")

    print("\n7. Demonstrating Resizing:")
    print(f"Initial size: {int_hash_map.size}")
    for i in range(31):
        int_hash_map[f"key_{i}"] = i
    print(f"Size after adding items: {int_hash_map.size}")
    print(f"Current number of items: {len(int_hash_map)}")


if __name__ == "__main__":
    main()
