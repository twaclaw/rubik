from collections import deque

import numpy as np

from .cube import Cube, Move


def bfs(cube: Cube) -> tuple[list[str] | None, int]:
    if cube.size > 2:
        raise NotImplementedError(
            "Are you nuts, BFS, in Python, for a cube larger than 2x2x2?"
        )

    if cube.is_solution():
        return [], 0

    queue = deque([(cube.compress(), np.array([], dtype=np.uint8))])  # (compressed_state, move_sequence)

    states_processed = 0
    visited = {cube.hashable()}

    np.random.shuffle(cube.possible_moves)


    while queue:
        states_processed += 1
        current_state, move_seq = queue.popleft()

        if states_processed % 50000 == 0:
            print(
                f"States processed: {states_processed:,}, Seq len: {len(move_seq):,},  queue: {len(queue):,}, visited: {len(visited):,}"
            )

        cube.decompress(current_state)  # updates cube.faces

        for move_val in cube.possible_moves:
            move = Move(move_val)
            cube.move(move)

            if cube.is_solution():
                solution_path = [
                    Move(x).name for x in np.append(move_seq, [move.value])
                ]
                return solution_path, len(visited)

            hash_value = cube.hashable()
            if hash_value not in visited:
                visited.add(hash_value)
                new_move_seq = np.append(move_seq.copy(), [move.value])
                queue.append((cube.compress(), new_move_seq))

            cube.move(move.inverse())  # Undo the move

    return None, len(visited)

