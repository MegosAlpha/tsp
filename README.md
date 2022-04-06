# Traveling Salesperson Problem

CS312 Group Project: Section 1, Team 33

On this branch:
- Noah's implementation of the Held-Karp Algorithm
  - Written primarily in Rust using PyO3 to connect with Python
  - Uses Rayon at the combination level for multiprocessing / parallelism
  - Bounded by 32-bit signed integers for all path lengths
  - Assumes that all input problems are solvable
  - All path lengths must be nonnegative

## How to Use

Since this implementation uses a native extension written in Rust, it is more difficult to set up than the usual Python project.

1. [Install the Rust toolchain](https://www.rust-lang.org/learn/get-started).
2. Clone this repository, change directory, and select the correct branch.
3. Create virtual environment: `python3 -m venv .env`
4. Activate virtual environment `source .env/bin/activate`
5. Install `maturin` and other dependencies: `pip install maturin numpy PyQt6`
4. Build the optimized Python module: `maturin develop -r`
5. Launch as usual: `python Proj5GUI.py`

