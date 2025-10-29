"""
Bi-directional Iterative Deepening Search (IDS)
"""

from collections import deque

import numpy as np

from .cube import Cube, Move


def bi_ids(cube: Cube, max_depth: int) -> list[str]:
    if cube.size > 3:
        raise NotImplementedError("Not implemented for cubes larger than 2x2")

    if cube.is_solution():
        return [], 0

    visited_front: dict[bytes, np.ndarray] = {}
    visited_back: dict[bytes, np.ndarray] = {}

    c0 = cube.faces.copy()

    for depth in range(max_depth + 1):
        cube.faces = c0.copy()
        solution_path, nvisited = dfs(cube, depth, visited_front, visited_back)
        if solution_path is not None:
            return solution_path, nvisited

    return None, len(visited_front) + len(visited_back)


def dfs(
    cube: Cube,
    max_depth: int,
    visited_front: dict[bytes, np.ndarray],
    visited_back: dict[bytes, np.ndarray],
) -> tuple[list[str] | None, int]:

    cube_solved = Cube(size=cube.size, initial="solved")

    queue_front = deque([(cube.compress(), np.array([], dtype=np.uint8))])
    queue_back = deque([(cube_solved.compress(), np.array([], dtype=np.uint8))])

    print(f"DFS at depth {max_depth}, front visited: {len(visited_front)}, back visited: {len(visited_back)}")
    while queue_front:
        current_state, move_seq = queue_front.pop()
        cube.decompress(current_state)  # updates cube.faces

        if len(move_seq) < max_depth:
            for move_val in cube.possible_moves:
                move = Move(move_val)
                cube.move(move)

                new_move_seq = np.append(move_seq.copy(), [move.value])

                if cube.is_solution():
                    return [Move(x).name for x in new_move_seq], len(visited_front) + len(visited_back)

                hash_value = cube.hashable()

                if hash_value in visited_back:
                    seq_back_sol = cube.reverse_sequence(visited_back[hash_value])
                    return cube.solution_path(
                        np.append(new_move_seq, seq_back_sol)
                    ), len(visited_front) + len(visited_back)

                if hash_value not in visited_front:
                    visited_front[hash_value] = new_move_seq

                queue_front.append((cube.compress(), new_move_seq))

                cube.move(move.inverse())  # Undo the move

    while queue_back:
        current_state, move_seq = queue_back.pop()
        cube_solved.decompress(current_state)  # updates cube.faces

        if len(move_seq) < max_depth:
            for move_val in cube_solved.possible_moves:
                move = Move(move_val)
                cube_solved.move(move)

                new_move_seq = np.append(move_seq.copy(), [move.value])

                hash_value = cube_solved.hashable()

                if hash_value in visited_front:
                    seq_front_sol = visited_front[hash_value]
                    return cube_solved.solution_path(
                        np.append(
                            seq_front_sol, cube_solved.reverse_sequence(new_move_seq)
                        )
                    ), len(visited_front) + len(visited_back)

                if hash_value not in visited_back:
                    visited_back[hash_value] = new_move_seq

                queue_back.append((cube_solved.compress(), new_move_seq))

                cube_solved.move(move.inverse())  # Undo the move

    return None, 0
