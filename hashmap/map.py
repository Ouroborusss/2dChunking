import hashlib
from copy import deepcopy

import numpy as np


class HashMap:
    def __init__(self, initial_size=1_000_000, value_dtype=None, load_factor=0.70):
        self.size = initial_size
        self.keys = np.empty(initial_size, dtype=object)
        self.values = np.empty(initial_size, dtype=value_dtype if value_dtype else object)
        self.count = 0
        self.load_factor = load_factor
        self.min_load_factor = 0.20
        self.value_dtype = value_dtype
        self.empty_value = np.nan if value_dtype in (np.float32, np.float64) else 0
        self.collisions = 0
        self.resize_count = 0
        self._iter_index = 0
        self._value_to_keys = {}

    def _hash(self, key):
        key_str = str(key).encode("utf-8")
        return int(hashlib.sha256(key_str).hexdigest(), 16) % self.size

    def _hash2(self, key):
        key_str = str(key).encode("utf-8")
        step = int(hashlib.md5(key_str).hexdigest(), 16) % self.size
        return step if step != 0 else 1

    def _probe_indices(self, key):
        h1 = self._hash(key)
        h2 = self._hash2(key)
        for i in range(self.size):
            yield (h1 + i * h2) % self.size

    def _find_slot(self, key):
        for i, index in enumerate(self._probe_indices(key)):
            if self.keys[index] is None or self.keys[index] == key:
                if i > 0:
                    self.collisions += 1
                return index
        return None

    def _value_key(self, value):
        if isinstance(value, np.ndarray):
            return tuple(value.tolist())
        return value

    def _add_reverse_lookup(self, key, value):
        value_key = self._value_key(value)
        if value_key not in self._value_to_keys:
            self._value_to_keys[value_key] = set()
        self._value_to_keys[value_key].add(key)

    def _remove_reverse_lookup(self, key, value):
        value_key = self._value_key(value)
        if value_key in self._value_to_keys:
            self._value_to_keys[value_key].discard(key)
            if not self._value_to_keys[value_key]:
                del self._value_to_keys[value_key]

    def _resize(self, new_size=None):
        if new_size is None:
            new_size = int(self.size * 2.5)

        old_keys = self.keys
        old_values = self.values
        old_count = self.count

        self.size = new_size
        self.keys = np.empty(self.size, dtype=object)
        self.values = np.empty(self.size, dtype=self.value_dtype if self.value_dtype else object)
        self.count = 0
        self.resize_count += 1
        self._value_to_keys.clear()

        for key, value in zip(old_keys, old_values):
            if key is not None:
                index = self._find_slot(key)
                if index is not None:
                    self.keys[index] = key
                    self.values[index] = value
                    self.count += 1
                    self._add_reverse_lookup(key, value)

        assert self.count == old_count, f"Lost items during resize: {old_count} -> {self.count}"

    def _maybe_shrink(self):
        if self.size > 10 and self.count / self.size < self.min_load_factor:
            self._resize(max(10, self.size // 2))

    def clear(self):
        self.keys = np.empty(self.size, dtype=object)
        self.values = np.empty(self.size, dtype=self.value_dtype if self.value_dtype else object)
        self.count = 0
        self._value_to_keys.clear()
        self._maybe_shrink()

    def copy(self):
        new_map = HashMap(self.size, self.value_dtype)
        new_map.keys = np.copy(self.keys)
        new_map.values = np.copy(self.values)
        new_map.count = self.count
        new_map._value_to_keys = {k: set(v) for k, v in self._value_to_keys.items()}
        return new_map

    def deepcopy(self):
        return deepcopy(self)

    def update(self, other):
        if hasattr(other, "keys"):
            for key in other.keys():
                self[key] = other[key]
        else:
            for key, value in other:
                self[key] = value

    def get_keys(self):
        return [key for key in self.keys if key is not None]

    def get_values(self):
        return [self.values[i] for i, key in enumerate(self.keys) if key is not None]

    def get_items(self):
        return [(key, self.values[i]) for i, key in enumerate(self.keys) if key is not None]

    def get_metrics(self):
        return {
            "size": self.size,
            "count": self.count,
            "load_factor": self.count / self.size,
            "collisions": self.collisions,
            "resize_count": self.resize_count,
        }

    def put(self, key, value):
        if self.count / self.size >= self.load_factor:
            self._resize()

        index = self._find_slot(key)
        if index is None:
            self._resize()
            self.put(key, value)
            return

        if self.keys[index] is not None:
            self._remove_reverse_lookup(self.keys[index], self.values[index])
        elif self.keys[index] is None:
            self.count += 1

        self.keys[index] = key
        self.values[index] = value
        self._add_reverse_lookup(key, value)

    def get(self, key):
        for index in self._probe_indices(key):
            if self.keys[index] is None:
                return None
            if self.keys[index] == key:
                return self.values[index]
        return None

    def remove(self, key):
        for index in self._probe_indices(key):
            if self.keys[index] is None:
                return False
            if self.keys[index] == key:
                self._remove_reverse_lookup(key, self.values[index])
                self.keys[index] = None
                self.values[index] = None if self.value_dtype is None else self.empty_value
                self.count -= 1
                self._maybe_shrink()
                return True
        return False

    def contains_value(self, value):
        if self.value_dtype is not None and not isinstance(value, np.ndarray):
            return np.any(self.values == value)

        return any(
            isinstance(v, np.ndarray) and np.array_equal(v, value) if isinstance(value, np.ndarray) else v == value
            for v in self.values
            if v is not None
        )

    def get_keys_by_value(self, value):
        return list(self._value_to_keys.get(self._value_key(value), []))

    def get_all_values(self):
        if self.value_dtype is not None:
            return np.unique(self.values[self.keys != None])

        values = []
        seen = set()
        for value in self.values:
            if value is None:
                continue
            key = str(value.tolist()) if isinstance(value, np.ndarray) else value
            if key not in seen:
                seen.add(key)
                values.append(value)
        return values

    def __iter__(self):
        self._iter_index = 0
        return self

    def __next__(self):
        while self._iter_index < self.size:
            if self.keys[self._iter_index] is not None:
                key = self.keys[self._iter_index]
                self._iter_index += 1
                return key
            self._iter_index += 1
        raise StopIteration

    def __len__(self):
        return self.count

    def __contains__(self, key):
        return self.get(key) is not None

    def __getitem__(self, key):
        value = self.get(key)
        if value is None:
            raise KeyError(f"Key '{key}' not found")
        return value

    def __setitem__(self, key, value):
        self.put(key, value)

    def __str__(self):
        items = [f"{key}: {self.values[i]}" for i, key in enumerate(self.keys) if key is not None]
        return "{" + ", ".join(items) + "}"

    def __repr__(self):
        metrics = self.get_metrics()
        return (
            f"HashMap(size={metrics['size']}, count={metrics['count']}, "
            f"load_factor={metrics['load_factor']:.2f}, collisions={metrics['collisions']})"
        )
