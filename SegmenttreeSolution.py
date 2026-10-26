"""
Segment Tree for Real-Time FinTech Stock Price Monitoring.

Menu Operations:
  1. Display current timestamp and prices
  2. Query (by index range or timestamp range)
  3. Regenerate stock prices
  4. Load stock prices from file
  5. Load queries from file
  6. Update price at index
  7. Exit
"""

import math
import random
import os
import bisect
from datetime import datetime, timedelta


class Node:
    """Represents a node in the segment tree storing interval statistics."""
    def __init__(self, min_price, max_price, min_idx, max_idx):
        self.min_price = min_price
        self.max_price = max_price
        self.min_idx = min_idx
        self.max_idx = max_idx
        self.spread = round(max_price - min_price, 4)

    def get_spread(self):
        return self.spread

    def __iter__(self):
        """Allows tuple unpacking: min_p, max_p = node."""
        yield self.min_price
        yield self.max_price

    def __eq__(self, other):
        """Supports comparison with either another Node or a (min_p, max_p) tuple."""
        if isinstance(other, tuple) and len(other) == 2:
            return (self.min_price, self.max_price) == other
        if isinstance(other, Node):
            return (self.min_price, self.max_price, self.min_idx, self.max_idx) == \
                   (other.min_price, other.max_price, other.min_idx, other.max_idx)
        return False

    def __repr__(self):
        return (f"Node(min=${self.min_price:.2f} at idx {self.min_idx}, "
                f"max=${self.max_price:.2f} at idx {self.max_idx}, "
                f"spread=${self.spread:.2f})")


class SegmentTree:
    def __init__(self, prices=None, timestamps=None):
        self.prices = []
        self.timestamps = []
        self.n = 0
        self.tree = []
        if prices is not None:
            self.load_data(prices, timestamps)

    def load_data(self, prices, timestamps=None):
        """Loads price list and optional timestamps, then builds tree in O(N)."""
        if isinstance(prices, tuple) and len(prices) == 2 and isinstance(prices[0], list):
            prices, timestamps = prices

        self.prices = [float(p) for p in prices]
        self.n = len(self.prices)

        if timestamps and len(timestamps) == self.n:
            self.timestamps = list(timestamps)
        else:
            base_time = datetime.strptime("09:30:00", "%H:%M:%S")
            self.timestamps = [
                (base_time + timedelta(seconds=i)).strftime("%H:%M:%S")
                for i in range(self.n)
            ]

        if self.n > 0:
            self.tree = [None] * (4 * self.n)
            self._build(1, 0, self.n - 1)
        else:
            self.tree = []

    def _merge(self, left, right):
        """Merges left and right child nodes into a parent node."""
        if left is None:
            return right
        if right is None:
            return left

        if left.min_price <= right.min_price:
            min_p, min_i = left.min_price, left.min_idx
        else:
            min_p, min_i = right.min_price, right.min_idx

        if left.max_price >= right.max_price:
            max_p, max_i = left.max_price, left.max_idx
        else:
            max_p, max_i = right.max_price, right.max_idx

        return Node(min_p, max_p, min_i, max_i)

    def _build(self, node_idx, start, end):
        if start == end:
            val = self.prices[start]
            self.tree[node_idx] = Node(val, val, start, start)
            return

        mid = (start + end) // 2
        left_child = 2 * node_idx
        right_child = 2 * node_idx + 1

        self._build(left_child, start, mid)
        self._build(right_child, mid + 1, end)

        self.tree[node_idx] = self._merge(self.tree[left_child], self.tree[right_child])

    def update(self, index, new_price):
        """Updates price at target index in O(log N)."""
        if index < 0 or index >= self.n:
            raise IndexError(f"Index {index} out of bounds (0 to {self.n - 1})")

        self.prices[index] = float(new_price)
        self._update(1, 0, self.n - 1, index, float(new_price))

    def _update(self, node_idx, start, end, target_idx, new_price):
        if start == end:
            self.tree[node_idx] = Node(new_price, new_price, start, start)
            return

        mid = (start + end) // 2
        left_child = 2 * node_idx
        right_child = 2 * node_idx + 1

        if target_idx <= mid:
            self._update(left_child, start, mid, target_idx, new_price)
        else:
            self._update(right_child, mid + 1, end, target_idx, new_price)

        self.tree[node_idx] = self._merge(self.tree[left_child], self.tree[right_child])

    def query(self, left, right):
        """Queries interval [left, right] returning a Node in O(log N)."""
        if left < 0 or right >= self.n or left > right:
            raise ValueError(f"Invalid range [{left}, {right}] for size {self.n}")

        return self._query(1, 0, self.n - 1, left, right)

    def _query(self, node_idx, start, end, q_left, q_right):
        if q_right < start or q_left > end:
            return None

        if q_left <= start and end <= q_right:
            return self.tree[node_idx]

        mid = (start + end) // 2
        left_res = self._query(2 * node_idx, start, mid, q_left, q_right)
        right_res = self._query(2 * node_idx + 1, mid + 1, end, q_left, q_right)

        return self._merge(left_res, right_res)

    def query_by_timestamp(self, start_time, end_time):
        """Finds index bounds for timestamp range and queries in O(log N)."""
        if not self.timestamps:
            raise ValueError("No timestamps loaded.")

        l_idx = bisect.bisect_left(self.timestamps, start_time)
        r_idx = bisect.bisect_right(self.timestamps, end_time) - 1

        if l_idx > r_idx or l_idx >= self.n or r_idx < 0:
            raise ValueError(f"No price records found between {start_time} and {end_time}")

        return self.query(l_idx, r_idx), l_idx, r_idx

    def get_spread(self, left, right):
        """Convenience method returning spread in O(log N)."""
        node = self.query(left, right)
        return node.spread if node else 0.0

    def get_timestamp(self, idx):
        if 0 <= idx < len(self.timestamps):
            return self.timestamps[idx]
        return "N/A"

# ==========================================
# Helpers: File I/O and Synthetic Data
# ==========================================
def generate_synthetic_prices(n, start_price, start_time):
    """Generates synthetic stock prices with timestamps using a random walk."""
    prices = [float(start_price)]
    curr_time = datetime.strptime(start_time, "%H:%M:%S")
    timestamps = [curr_time.strftime("%H:%M:%S")]

    for i in range(1, n):
        step = round(random.uniform(-0.50, 0.50), 2)
        next_price = max(1.0, round(prices[-1] + step, 2))
        prices.append(next_price)
        curr_time += timedelta(seconds=1)
        timestamps.append(curr_time.strftime("%H:%M:%S"))

    return prices, timestamps


def load_prices_file(filepath):
    """Loads prices and timestamps from a CSV or text file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: '{filepath}'")

    prices = []
    timestamps = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            clean = line.strip()
            if not clean or clean.startswith("#"):
                continue

            parts = [p.strip() for p in clean.replace(",", " ").split()]

            if len(parts) >= 2:
                try:
                    price = float(parts[1])
                    timestamps.append(parts[0])
                    prices.append(price)
                except ValueError:
                    continue
            elif len(parts) == 1:
                try:
                    price = float(parts[0])
                    prices.append(price)
                except ValueError:
                    continue

    if not prices:
        raise ValueError(f"No valid price data could be parsed from '{filepath}'.")

    return prices, (timestamps if len(timestamps) == len(prices) else None)

def load_queries_file(filepath):
    """Loads range queries (indices or timestamps) from a file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: '{filepath}'")

    queries = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            clean = line.strip()
            if not clean or clean.startswith("#"):
                continue

            parts = [p.strip() for p in clean.replace(",", " ").split()]
            if len(parts) < 2:
                continue

            try:
                l, r = int(parts[0]), int(parts[1])
                queries.append(("index", l, r))
            except ValueError:
                queries.append(("time", parts[0], parts[1]))

    return queries


# ==========================================
# Menu Operations Implementation
# ==========================================
def op_display_prices(st):
    """Operation 1: Display current timestamp and prices."""
    if st.n == 0:
        print("\nNo stock prices loaded. Please load or generate prices first.")
        return

    print(f"\n--- Current Stock Feed ({st.n} records) ---")
    print(f"{'Index':<8} {'Timestamp':<14} {'Price':<12}")
    print("-" * 34)

    if st.n <= 50:
        for i in range(st.n):
            print(f"[{i:<4}]  {st.timestamps[i]:<14} ${st.prices[i]:.2f}")
    else:
        for i in range(15):
            print(f"[{i:<4}]  {st.timestamps[i]:<14} ${st.prices[i]:.2f}")
        print(f"{'...':^34}")
        for i in range(st.n - 5, st.n):
            print(f"[{i:<4}]  {st.timestamps[i]:<14} ${st.prices[i]:.2f}")
        print(f"\n(Showing 20 of {st.n} total records)")


def op_query(st):
    """Operation 2: Query by index or timestamp."""
    if st.n == 0:
        print("\nNo stock prices loaded. Please load or generate prices first.")
        return

    print("\n--- Query Options ---")
    print("  1. Query by Index range [L, R]")
    print("  2. Query by Timestamp range [Start_Time, End_Time]")
    choice = input("Select query type (1 or 2): ").strip()

    try:
        if choice == "2":
            start_t = input(f"Enter start timestamp (e.g. {st.timestamps[0]}): ").strip()
            end_t = input(f"Enter end timestamp   (e.g. {st.timestamps[-1]}): ").strip()
            res, l_idx, r_idx = st.query_by_timestamp(start_t, end_t)
            print(f"\n[Result for Timestamps: {start_t} to {end_t} (Indices {l_idx} to {r_idx})]")
        else:
            l_str = input(f"Enter start index (0 to {st.n - 1}): ").strip()
            r_str = input(f"Enter end index   (0 to {st.n - 1}): ").strip()
            l_idx, r_idx = int(l_str), int(r_str)
            res = st.query(l_idx, r_idx)
            print(f"\n[Result for Indices [{l_idx}, {r_idx}] ({st.timestamps[l_idx]} to {st.timestamps[r_idx]})]")

        min_time = st.get_timestamp(res.min_idx)
        max_time = st.get_timestamp(res.max_idx)
        print(f"  Min Price : ${res.min_price:.2f} (at {min_time}, index {res.min_idx})")
        print(f"  Max Price : ${res.max_price:.2f} (at {max_time}, index {res.max_idx})")
        print(f"  Spread    : ${res.spread:.2f}")

    except (ValueError, IndexError) as err:
        print(f"Query Error: {err}")


def op_regenerate_prices(st):
    """Operation 3: Regenerate stock prices."""
    print("\n--- Regenerate Stock Prices ---")
    try:
        count = int(input("Enter number of price ticks to generate: ").strip())
        start_p = float(input("Enter starting stock price: ").strip())
        start_t = input("Enter starting timestamp (e.g. 09:30:00): ").strip()

        prices, timestamps = generate_synthetic_prices(count, start_p, start_t)
        st.load_data(prices, timestamps)
        print(f"Successfully generated and loaded {st.n} stock prices into Segment Tree.")
    except Exception as err:
        print(f"Error generating prices: {err}")
