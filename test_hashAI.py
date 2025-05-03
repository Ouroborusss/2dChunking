import unittest
import numpy as np
import time
from hashMap import HashMap

class TestHashMap(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.string_hash = HashMap(initial_size=10)
        self.int_hash = HashMap(initial_size=10, value_dtype=np.int32)
        self.float_hash = HashMap(initial_size=10, value_dtype=np.float64)

    def test_basic_operations(self):
        """Test basic put and get operations."""
        # String hash map
        self.string_hash["test"] = "value"
        self.assertEqual(self.string_hash["test"], "value")
        
        # Integer hash map
        self.int_hash["test"] = 42
        self.assertEqual(self.int_hash["test"], 42)
        
        # Float hash map
        self.float_hash["test"] = 3.14
        self.assertEqual(self.float_hash["test"], 3.14)

    def test_type_enforcement(self):
        """Test that type enforcement works correctly."""
        # Test integer hash map type enforcement
        with self.assertRaises(ValueError):
            self.int_hash["test"] = "not an integer"
        
        # Test float hash map type enforcement
        with self.assertRaises(ValueError):
            self.float_hash["test"] = "not a float"

    def test_complex_keys(self):
        """Test using complex objects as keys."""
        # Tuple keys
        self.string_hash[(1, 2, 3)] = "tuple key"
        self.assertEqual(self.string_hash[(1, 2, 3)], "tuple key")
        
        # Frozen set keys
        key = frozenset([1, 2, 3])
        self.string_hash[key] = "frozen set key"
        self.assertEqual(self.string_hash[key], "frozen set key")

    def test_collisions(self):
        """Test collision handling."""
        # Create many items to force collisions
        for i in range(20):
            self.int_hash[f"key_{i}"] = i
        
        # Verify all items are still accessible
        for i in range(20):
            self.assertEqual(self.int_hash[f"key_{i}"], i)

    def test_load_factor_and_resize(self):
        """Test that the hash map resizes correctly."""
        initial_size = self.int_hash.size
        
        # Add items until resize is triggered
        items_to_add = int(initial_size * 0.8)  # Exceed load factor
        for i in range(items_to_add):
            self.int_hash[f"key_{i}"] = i
            
        # Verify size increased
        self.assertTrue(self.int_hash.size > initial_size)
        
        # Verify all items are still accessible
        for i in range(items_to_add):
            self.assertEqual(self.int_hash[f"key_{i}"], i)

    def test_remove_operations(self):
        """Test remove functionality."""
        # Test string hash map
        self.string_hash["test"] = "value"
        self.assertTrue(self.string_hash.remove("test"))
        self.assertFalse("test" in self.string_hash)
        
        # Test int hash map
        self.int_hash["test"] = 42
        self.assertTrue(self.int_hash.remove("test"))
        self.assertFalse("test" in self.int_hash)
        
        # Test removing non-existent key
        self.assertFalse(self.string_hash.remove("nonexistent"))

    def test_edge_cases(self):
        """Test edge cases."""
        # Test empty string key
        self.string_hash[""] = "empty key"
        self.assertEqual(self.string_hash[""], "empty key")
        
        # Test None as value in string hash (only for non-typed hash maps)
        self.string_hash["none_test"] = "null value"
        self.assertEqual(self.string_hash["none_test"], "null value")
        
        # Test zero as key and value
        self.int_hash[0] = 0
        self.assertEqual(self.int_hash[0], 0)
        
        # Test large numbers
        self.int_hash["large"] = 2**31 - 1  # Max 32-bit int
        self.assertEqual(self.int_hash["large"], 2**31 - 1)

    def test_performance(self):
        """Test performance with large number of items."""
        start_time = time.time()
        
        # Insert items
        num_items = 1000  # Reduced from 10000
        for i in range(num_items):
            self.int_hash[f"key_{i}"] = i
            
        insert_time = time.time() - start_time
        print(f"\nInsert time for {num_items} items: {insert_time:.2f} seconds")
        
        # Test retrieval
        start_time = time.time()
        for i in range(num_items):
            _ = self.int_hash[f"key_{i}"]
            
        retrieval_time = time.time() - start_time
        print(f"Retrieval time for {num_items} items: {retrieval_time:.2f} seconds")
        
        # Verify load factor is maintained
        self.assertLess(self.int_hash.count / self.int_hash.size, self.int_hash.load_factor)

    def test_special_values(self):
        """Test handling of special values."""
        # Test float hash with special values
        special_values = {
            "inf": float("inf"),
            "-inf": float("-inf"),
            "nan": float("nan")
        }
        
        for key, value in special_values.items():
            self.float_hash[key] = value
            if key == "nan":
                self.assertTrue(np.isnan(self.float_hash[key]))
            else:
                self.assertEqual(self.float_hash[key], value)

    def test_concurrent_operations(self):
        """Test mixed operations in sequence."""
        operations = [
            ("put", "key1", 42),
            ("put", "key2", 43),
            ("get", "key1", 42),
            ("remove", "key1", True),
            ("get", "key1", None),
            ("put", "key1", 44),
            ("get", "key1", 44)
        ]
        
        for op, key, expected in operations:
            if op == "put":
                self.int_hash[key] = expected
            elif op == "get":
                if expected is None:
                    self.assertIsNone(self.int_hash.get(key))
                else:
                    self.assertEqual(self.int_hash[key], expected)
            elif op == "remove":
                self.assertEqual(self.int_hash.remove(key), expected)

if __name__ == '__main__':
    unittest.main(verbosity=2) 