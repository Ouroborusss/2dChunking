# 2dChunking

A 2D chunking algorithm that loads coordinates from CSV and uses spatial chunking to find nearest neighbors. Also includes a custom NumPy-backed hash map with reverse value lookup.

## Project layout

```
chunking/          # Spatial grid and nearest-neighbor search
  config.py        # Demo parameters (bounds, point count, chunk level)
  grid.py          # Grid creation, coordinate placement
  search.py        # Adjacent chunk search and nearest neighbor
  __main__.py      # Chunking benchmark demo

hashmap/           # Custom hash map implementation
  map.py           # HashMap class (double hashing, resize, value lookup)
  benchmark.py     # Hash map vs brute-force benchmark

coordinates.py     # Shared CSV load/generate utilities
examples/          # Usage examples
tests/             # Unit and feature tests
```

## Setup

From the project root (the folder containing `hashmap/`, `chunking/`, and `coordinates.py`):

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
```

`pip install -e .` installs the project in editable mode so imports like `from hashmap import HashMap` work from anywhere in the repo.

If you prefer not to install, you can also run scripts with:

```bash
PYTHONPATH=. python examples/hash_usage.py
```

## Usage

**Chunking nearest-neighbor demo** (edit parameters in `chunking/config.py`):

```bash
python -m chunking
# or
python main.py
```

**Hash map benchmark**:

```bash
python hashmap/benchmark.py
# or
python hash_check.py
```

**Hash map example**:

```bash
python examples/hash_usage.py
```

**Run tests**:

```bash
python -m unittest discover -s tests -v
```

## Notes

- Large point counts will take time to chunk and can be memory-intensive.
- Search performance improves significantly once coordinates are placed in the grid.
- `chunking/config.py` controls bounds, number of points, and chunking level.
