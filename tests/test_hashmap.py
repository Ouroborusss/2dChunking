import unittest
import time

import numpy as np

from hashmap import HashMap


class TestHashMap(unittest.TestCase):
    def setUp(self):
        self.string_hash = HashMap(initial_size=10)
        self.int_hash = HashMap(initial_size=10, value_dtype=np.int32)
        self.float_hash = HashMap(initial_size=10, value_dtype=np.float64)

    def test_basic_operations(self):
        self.string_hash["test"] = "value"
        self.assertEqual(self.string_hash["test"], "value")

        self.int_hash["test"] = 42
        self.assertEqual(self.int_hash["test"], 42)

        self.float_hash["test"] = 3.14
        self.assertEqual(self.float_hash["test"], 3.14)

    def test_type_enforcement(self):
        with self.assertRaises(ValueError):
            self.int_hash["test"] = "not an integer"

        with self.assertRaises(ValueError):
            self.float_hash["test"] = "not a float"

    def test_complex_keys(self):
        self.string_hash[(1, 2, 3)] = "tuple key"
        self.assertEqual(self.string_hash[(1, 2, 3)], "tuple key")

        key = frozenset([1, 2, 3])
        self.string_hash[key] = "frozen set key"
        self.assertEqual(self.string_hash[key], "frozen set key")

    def test_collisions(self):
        for i in range(20):
            self.int_hash[f"key_{i}"] = i

        for i in range(20):
            self.assertEqual(self.int_hash[f"key_{i}"], i)

    def test_load_factor_and_resize(self):
        initial_size = self.int_hash.size
        items_to_add = int(initial_size * 0.8)

        for i in range(items_to_add):
            self.int_hash[f"key_{i}"] = i

        self.assertTrue(self.int_hash.size > initial_size)

        for i in range(items_to_add):
            self.assertEqual(self.int_hash[f"key_{i}"], i)

    def test_remove_operations(self):
        self.string_hash["test"] = "value"
        self.assertTrue(self.string_hash.remove("test"))
        self.assertFalse("test" in self.string_hash)

        self.int_hash["test"] = 42
        self.assertTrue(self.int_hash.remove("test"))
        self.assertFalse("test" in self.int_hash)
        self.assertFalse(self.string_hash.remove("nonexistent"))

    def test_edge_cases(self):
        self.string_hash[""] = "empty key"
        self.assertEqual(self.string_hash[""], "empty key")

        self.string_hash["none_test"] = "null value"
        self.assertEqual(self.string_hash["none_test"], "null value")

        self.int_hash[0] = 0
        self.assertEqual(self.int_hash[0], 0)

        self.int_hash["large"] = 2**31 - 1
        self.assertEqual(self.int_hash["large"], 2**31 - 1)

    def test_performance(self):
        start_time = time.time()
        num_items = 1000

        for i in range(num_items):
            self.int_hash[f"key_{i}"] = i

        insert_time = time.time() - start_time
        print(f"\nInsert time for {num_items} items: {insert_time:.2f} seconds")

        start_time = time.time()
        for i in range(num_items):
            _ = self.int_hash[f"key_{i}"]

        retrieval_time = time.time() - start_time
        print(f"Retrieval time for {num_items} items: {retrieval_time:.2f} seconds")
        self.assertLess(self.int_hash.count / self.int_hash.size, self.int_hash.load_factor)

    def test_special_values(self):
        special_values = {
            "inf": float("inf"),
            "-inf": float("-inf"),
            "nan": float("nan"),
        }

        for key, value in special_values.items():
            self.float_hash[key] = value
            if key == "nan":
                self.assertTrue(np.isnan(self.float_hash[key]))
            else:
                self.assertEqual(self.float_hash[key], value)

    def test_concurrent_operations(self):
        operations = [
            ("put", "key1", 42),
            ("put", "key2", 43),
            ("get", "key1", 42),
            ("remove", "key1", True),
            ("get", "key1", None),
            ("put", "key1", 44),
            ("get", "key1", 44),
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
