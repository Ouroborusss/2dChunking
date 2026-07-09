from hashmap import HashMap


def test_resize_behavior():
    hash_map = HashMap(initial_size=10)
    print(f"Initial size: {hash_map.size}")

    for i in range(37):
        hash_map[f"key_{i}"] = i
        if i in (7, 14, 28):
            print(f"After {i + 1} items, size: {hash_map.size}")

    print(f"Final size with 37 items: {hash_map.size}")
    print(f"Load factor: {hash_map.count / hash_map.size:.2f}")


if __name__ == "__main__":
    test_resize_behavior()
