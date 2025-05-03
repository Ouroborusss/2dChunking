import numpy as np
import hashlib
from copy import deepcopy

class HashMap:
    def __init__(self, initial_size=1000000, value_dtype=None, load_factor=0.70):
        """
        Initialize a hash map optimized for large datasets.
        
        Args:
            initial_size (int): Initial size of the hash table (default: 1M)
            value_dtype: NumPy dtype for values (None for object type)
            load_factor (float): Load factor threshold (default: 0.50)
        """
        self.size = initial_size
        self.keys = np.empty(initial_size, dtype=object)
        self.values = np.empty(initial_size, dtype=value_dtype if value_dtype else object)
        self.count = 0
        self.load_factor = load_factor
        self.min_load_factor = 0.20
        self.value_dtype = value_dtype
        # Store a special value to indicate empty slots for typed arrays
        self.empty_value = np.nan if value_dtype in (np.float32, np.float64) else 0
        # Performance metrics
        self.collisions = 0
        self.resize_count = 0
        self._iter_index = 0
        # Reverse lookup dictionary for values
        self._value_to_keys = {}

    def _hash(self, key):
        """Primary hash function."""
        key_str = str(key).encode('utf-8')
        return int(hashlib.sha256(key_str).hexdigest(), 16) % self.size

    def _hash2(self, key):
        """Secondary hash function for double hashing."""
        key_str = str(key).encode('utf-8')
        h = int(hashlib.md5(key_str).hexdigest(), 16) % self.size
        return h if h != 0 else 1  # Ensure non-zero

    def _find_slot(self, key):
        """Find an empty slot or the slot containing the key using double hashing."""
        h1 = self._hash(key)
        h2 = self._hash2(key)
        i = 0
        while i < self.size:
            index = (h1 + i * h2) % self.size
            if self.keys[index] is None or self.keys[index] == key:
                if i > 0:  # Count collisions
                    self.collisions += 1
                return index
            i += 1
        return None

    def _resize(self, new_size=None):
        """Resize the hash table with optimized growth factor."""
        if new_size is None:
            # Use a larger growth factor for big datasets
            new_size = int(self.size * 2.5)
        old_keys = self.keys
        old_values = self.values
        old_count = self.count
        self.size = new_size
        self.keys = np.empty(self.size, dtype=object)
        self.values = np.empty(self.size, dtype=self.value_dtype if self.value_dtype else object)
        self.count = 0
        self.resize_count += 1

        # Rehash all existing items
        for key, value in zip(old_keys, old_values):
            if key is not None:
                index = self._find_slot(key)
                if index is not None:
                    self.keys[index] = key
                    self.values[index] = value
                    self.count += 1
        assert self.count == old_count, f"Lost items during resize: {old_count} -> {self.count}"

    def _maybe_shrink(self):
        """Shrink the hash table if load factor is too low."""
        if self.size > 10 and self.count / self.size < self.min_load_factor:
            new_size = max(10, self.size // 2)
            self._resize(new_size)

    def clear(self):
        """Remove all items from the hash map."""
        self.keys = np.empty(self.size, dtype=object)
        self.values = np.empty(self.size, dtype=self.value_dtype if self.value_dtype else object)
        self.count = 0
        self._maybe_shrink()

    def copy(self):
        """Return a shallow copy of the hash map."""
        new_map = HashMap(self.size, self.value_dtype)
        new_map.keys = np.copy(self.keys)
        new_map.values = np.copy(self.values)
        new_map.count = self.count
        return new_map

    def deepcopy(self):
        """Return a deep copy of the hash map."""
        return deepcopy(self)

    def update(self, other):
        """Update the hash map with elements from another mapping or iterable."""
        if hasattr(other, 'keys'):
            for key in other.keys():
                self[key] = other[key]
        else:
            for key, value in other:
                self[key] = value

    def get_keys(self):
        """Return a list of all keys in the hash map."""
        return [key for key in self.keys if key is not None]

    def get_values(self):
        """Return a list of all values in the hash map."""
        return [self.values[i] for i, key in enumerate(self.keys) if key is not None]

    def get_items(self):
        """Return a list of all key-value pairs in the hash map."""
        return [(key, self.values[i]) for i, key in enumerate(self.keys) if key is not None]

    def get_metrics(self):
        """Return performance metrics."""
        return {
            'size': self.size,
            'count': self.count,
            'load_factor': self.count / self.size,
            'collisions': self.collisions,
            'resize_count': self.resize_count
        }

    def __iter__(self):
        """Make the hash map iterable."""
        self._iter_index = 0
        return self

    def __next__(self):
        """Return the next key in iteration."""
        while self._iter_index < self.size:
            if self.keys[self._iter_index] is not None:
                key = self.keys[self._iter_index]
                self._iter_index += 1
                return key
            self._iter_index += 1
        raise StopIteration

    def __str__(self):
        """Return a string representation of the hash map."""
        items = [f"{key}: {self.values[i]}" for i, key in enumerate(self.keys) if key is not None]
        return "{" + ", ".join(items) + "}"

    def __repr__(self):
        """Return a detailed string representation of the hash map."""
        metrics = self.get_metrics()
        return f"HashMap(size={metrics['size']}, count={metrics['count']}, " \
               f"load_factor={metrics['load_factor']:.2f}, collisions={metrics['collisions']})"

    def put(self, key, value):
        """
        Insert a key-value pair into the hash map using linear probing.
        """
        if self.count / self.size >= self.load_factor:
            self._resize()

        index = self._find_slot(key)
        if index is not None:
            # Remove old value from reverse lookup if it exists
            if self.keys[index] is not None:
                old_value = self.values[index]
                if isinstance(old_value, np.ndarray):
                    old_value = tuple(old_value.tolist())
                if old_value in self._value_to_keys:
                    self._value_to_keys[old_value].remove(self.keys[index])
                    if not self._value_to_keys[old_value]:
                        del self._value_to_keys[old_value]
            
            if self.keys[index] is None:
                self.count += 1
            self.keys[index] = key
            self.values[index] = value
            
            # Add to reverse lookup
            if isinstance(value, np.ndarray):
                value_key = tuple(value.tolist())
            else:
                value_key = value
            if value_key not in self._value_to_keys:
                self._value_to_keys[value_key] = set()
            self._value_to_keys[value_key].add(key)
        else:
            self._resize()
            self.put(key, value)

    def get(self, key):
        """
        Retrieve the value associated with a key using linear probing.
        """
        h = self._hash(key)
        i = 0
        
        while i < self.size:
            index = (h + i) % self.size
            if self.keys[index] is None:
                return None
            if self.keys[index] == key:
                return self.values[index]
            i += 1
        return None

    def remove(self, key):
        """
        Remove a key-value pair from the hash map using linear probing.
        """
        h = self._hash(key)
        i = 0
        
        while i < self.size:
            index = (h + i) % self.size
            if self.keys[index] is None:
                return False
            if self.keys[index] == key:
                # Remove from reverse lookup
                value = self.values[index]
                if isinstance(value, np.ndarray):
                    value_key = tuple(value.tolist())
                else:
                    value_key = value
                if value_key in self._value_to_keys:
                    self._value_to_keys[value_key].remove(key)
                    if not self._value_to_keys[value_key]:
                        del self._value_to_keys[value_key]
                
                self.keys[index] = None
                if self.value_dtype is None:
                    self.values[index] = None
                else:
                    self.values[index] = self.empty_value
                self.count -= 1
                self._maybe_shrink()
                return True
            i += 1
        return False

    def __len__(self):
        """Return the number of items in the hash map."""
        return self.count

    def __contains__(self, key):
        """Check if a key exists in the hash map."""
        return self.get(key) is not None

    def __getitem__(self, key):
        """Enable dictionary-style access with square brackets."""
        value = self.get(key)
        if value is None:
            raise KeyError(f"Key '{key}' not found")
        return value

    def __setitem__(self, key, value):
        """Enable dictionary-style assignment with square brackets."""
        self.put(key, value)

    def contains_value(self, value):
        """Check if a value exists in the hash map."""
        if self.value_dtype is not None:
            # For typed arrays, use numpy's comparison
            if isinstance(value, np.ndarray):
                # For array values, use numpy's array_equal
                return any(isinstance(v, np.ndarray) and np.array_equal(v, value) 
                         for v in self.values if v is not None)
            else:
                # For scalar values, use direct comparison
                return np.any(self.values == value)
        else:
            # For object arrays, use Python's equality
            if isinstance(value, np.ndarray):
                # For array values, use numpy's array_equal
                return any(isinstance(v, np.ndarray) and np.array_equal(v, value) 
                         for v in self.values if v is not None)
            else:
                # For scalar values, use direct comparison
                return any(v == value for v in self.values if v is not None)

    def get_keys_by_value(self, value):
        """Return all keys that map to the given value."""
        if isinstance(value, np.ndarray):
            value_key = tuple(value.tolist())
        else:
            value_key = value
        return list(self._value_to_keys.get(value_key, []))

    def get_all_values(self):
        """Return a list of all unique values in the hash map."""
        if self.value_dtype is not None:
            # For typed arrays, use numpy's unique
            return np.unique(self.values[self.keys != None])
        else:
            # For object arrays, handle both regular objects and numpy arrays
            values = []
            seen = set()
            for v in self.values:
                if v is not None:
                    if isinstance(v, np.ndarray):
                        # For arrays, use their string representation as a key
                        v_str = str(v.tolist())
                        if v_str not in seen:
                            seen.add(v_str)
                            values.append(v)
                    else:
                        # For regular objects, use the object itself as a key
                        if v not in seen:
                            seen.add(v)
                            values.append(v)
            return values
