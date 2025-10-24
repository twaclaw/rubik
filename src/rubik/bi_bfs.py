from collections import deque

import numpy as np

from .cube import Cube, Move


def reverse_sequence(sequence: np.ndarray) -> np.ndarray:
    return np.array([Move(x).inverse().value for x in sequence[::-1]])


def solution_path(sequence: np.ndarray) -> list[str]:
    return [Move(x).name for x in sequence]


def bi_bfs(cube: Cube) -> tuple[list[str] | None, int]:
    """
    Bidirectional BFS. Meet in the middle.
    This is just a possible way of implemented bidirectional BFS. This one doesn't guarantee that the shortest path is found.
    A more general way would be to have a function that expands one the frontier and call this
    function from a terminate function that evaluates when to stop the search.
    """
    if cube.size > 2:
        raise NotImplementedError(
            "Are you nuts, BFS, in Python, for a cube larger than 2x2x2?"
        )

    if cube.is_solution():
        return [], 0

    solved_cube = Cube(size=cube.size, initial="solved")

    queue_front = deque(
        [(cube.compress(), np.array([], dtype=np.uint8))]
    )  # (compressed_state, seq_frontuence)
    queue_back = deque([(solved_cube.compress(), np.array([], dtype=np.uint8))])

    states_processed = 0
    visited_front: dict[bytes, np.ndarray] = {}
    visited_front[cube.hashable()] = np.array([], dtype=np.uint8)
    visited_back: dict[bytes, np.ndarray] = {}
    visited_back[solved_cube.hashable()] = np.array([], dtype=np.uint8)

    while queue_front or queue_back:
        states_processed += 1
        state_front, seq_front = queue_front.popleft()
        state_back, seq_back = queue_back.popleft()

        cube.decompress(state_front)
        solved_cube.decompress(state_back)

        # Forward search
        for move_val in cube.possible_moves:
            move = Move(move_val)
            cube.move(move)

            if cube.is_solution():
                return solution_path(np.append(seq_front, [move])), len(visited_front)

            hash_value = cube.hashable()

            new_seq = np.append(seq_front.copy(), [move.value])

            if hash_value in visited_back:
                seq_back_sol = reverse_sequence(visited_back[hash_value])
                return solution_path(np.append(new_seq, seq_back_sol)), len(visited_front) + len(visited_back)

            if hash_value not in visited_front:
                visited_front[hash_value] = new_seq
                queue_front.append((cube.compress(), new_seq))

            cube.move(move.inverse())  # Undo the move

        # Backward search
        for move_val in solved_cube.possible_moves:
            move = Move(move_val)
            solved_cube.move(move)

            hash_value = solved_cube.hashable()

            new_seq = np.append(seq_back.copy(), [move.value])

            if hash_value in visited_front:
                seq_front_sol = visited_front[hash_value]
                return solution_path(np.append(seq_front_sol, reverse_sequence(new_seq))), len(visited_front) + len(visited_back)

            if hash_value not in visited_back:
                visited_back[hash_value] = new_seq
                queue_back.append((solved_cube.compress(), new_seq))

            solved_cube.move(move.inverse())  # Undo the move

    return None, len(visited_front)
