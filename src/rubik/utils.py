import numpy as np


class VisitedSet:
    def __init__(self, initial_capacity: int = 1024):
        self.data = np.empty(initial_capacity, dtype=np.int64)
        self.size = 0
        self.capacity = initial_capacity

    def _grow(self):
        new_capacity = self.capacity * 2
        new_data = np.empty(new_capacity, dtype=np.int64)
        new_data[:self.size] = self.data[:self.size]
        self.data = new_data
        self.capacity = new_capacity

    def add(self, value):
        # Check if already exists using binary search on sorted array
        if self.size > 0:
            idx = np.searchsorted(self.data[:self.size], value)
            if idx < self.size and self.data[idx] == value:
                return  # Already exists

        # Grow if needed
        if self.size >= self.capacity:
            self._grow()

        # Insert in sorted position
        if self.size == 0:
            self.data[0] = value
        else:
            idx = np.searchsorted(self.data[:self.size], value)
            # Shift elements to make room
            self.data[idx+1:self.size+1] = self.data[idx:self.size]
            self.data[idx] = value

        self.size += 1

    def __contains__(self, value):
        if self.size == 0:
            return False
        idx = np.searchsorted(self.data[:self.size], value)
        return idx < self.size and self.data[idx] == value

    def __len__(self):
        return self.size


class Queue:
    """
    More memory efficient queue using a numpy array.
    max_depth: maximum number of moves to store per path
    """
    def __init__(self, max_depth=30, max_state_size=30, initial_capacity=10000):
        self.max_depth = max_depth
        self.max_state_size = max_state_size
        self.capacity = initial_capacity

        # Record: [state_length(1), state_data(max_size), moves_length(1), moves(max_depth)]
        self.record_size = 1 + max_state_size + 1 + max_depth
        self.data = np.empty((initial_capacity, self.record_size), dtype=np.uint8)

        self.head = 0
        self.size = 0

    def _resize(self):
        """Double the capacity when full"""
        capacity = self.capacity * 2

        try:
            data = np.empty((capacity, self.record_size), dtype=np.uint8)
        except MemoryError as e:
            raise MemoryError(f"Failed to allocate memory for queue resize: {capacity} items") from e

        # Copy existing data, handling circular buffer wraparound
        if self.head + self.size <= self.capacity:
            data[:self.size] = self.data[self.head:self.head + self.size]
        else:
            first_part_size = self.capacity - self.head
            data[:first_part_size] = self.data[self.head:]
            data[first_part_size:self.size] = self.data[:self.size - first_part_size]

        # Update queue state
        self.data = data
        self.head = 0
        self.capacity = capacity

    def enqueue(self, compressed_state: bytes, moves_array: np.ndarray):
        if self.size >= self.capacity:
            print(f"Resizing queue from capacity {self.capacity} to {self.capacity * 2}")
            self._resize()

        # Validate inputs
        state_len = len(compressed_state)
        if state_len > self.max_state_size:
            raise ValueError(f"Compressed state size {state_len} exceeds max_state_size {self.max_state_size}")

        moves_len = len(moves_array)
        if moves_len > self.max_depth:
            raise ValueError(f"Moves array length {moves_len} exceeds max_depth {self.max_depth}")

        tail = (self.head + self.size) % self.capacity
        record = self.data[tail]

        record[0] = state_len

        for i, byte_val in enumerate(compressed_state):
            record[1 + i] = byte_val

        record[1 + self.max_state_size] = moves_len
        record[1 + self.max_state_size + 1:1 + self.max_state_size + 1 + moves_len] = moves_array[:moves_len]

        self.size += 1

    def dequeue(self):
        if self.size == 0:
            raise IndexError("Queue is empty")

        record = self.data[self.head]

        state_len = record[0]
        compressed_state = bytes(record[1:1 + state_len])

        moves_len = record[1 + self.max_state_size]
        moves_array = record[1 + self.max_state_size + 1:1 + self.max_state_size + 1 + moves_len].copy()

        self.head = (self.head + 1) % self.capacity
        self.size -= 1

        return compressed_state, moves_array

    def is_empty(self):
        return self.size == 0

    def __len__(self):
        return self.size

    def __bool__(self):
        return self.size > 0

    def memory_usage_mb(self):
        """Calculate memory usage in MB"""
        total_bytes = self.capacity * self.record_size
        return total_bytes / (1024 * 1024)

