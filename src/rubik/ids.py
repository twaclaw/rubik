"""
Iterative Deepening Search (IDS)
"""

from collections import deque

import numpy as np

from .cube import Cube, Move


def ids(cube: Cube, max_depth: int) -> list[str]:
    if cube.size > 2:
        raise NotImplementedError("Not implemented for cubes larger than 2x2")

    if cube.is_solution():
        return []

    c0 = cube.faces.copy()
    for depth in range(max_depth + 1):
        cube.faces = c0.copy()
        solution_path = dfs(cube, depth)
        if solution_path is not None:
            return solution_path

    return None

def dfs(cube: Cube, max_depth: int) -> tuple[list[str] | None, int]:
    queue = deque([(cube.compress(), np.array([], dtype=np.uint8))])

    i = 0
    while queue:
        current_state, move_seq = queue.pop()
        cube.decompress(current_state)  # updates cube.faces

        i += 1
        if i % 100000 == 0:
            print(
                f"Current depth: {len(move_seq)}, Max depth: {max_depth}, Queue size: {len(queue)}"
            )

        if len(move_seq) < max_depth:
            for move_val in cube.possible_moves:
                move = Move(move_val)
                cube.move(move)

                new_move_seq = np.append(move_seq.copy(), [move.value])

                if cube.is_solution():
                    return [Move(x).name for x in new_move_seq]

                queue.append((cube.compress(), new_move_seq))

                cube.move(move.inverse())  # Undo the move

    return None
