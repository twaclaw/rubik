"""
Iterative Deepening Search (IDS)
"""

from collections import deque

import numpy as np
from rich.progress import Progress, TaskID

from .cube import N_STATES_CUBE_2, N_STATES_CUBE_3, Cube, Move

INFO: dict[str, str] = {
    "algorithm": "Iterative Deepening Depth First Search (IDDFS)",
}


def ids(cube: Cube, max_depth: int = 20) -> tuple[list[str] | None, int]:
    if cube.size > 2:
        raise NotImplementedError("Not implemented for cubes larger than 2x2")

    if cube.is_solution():
        return [], 0

    c0 = cube.faces.copy()
    total_visited = 0

    with Progress() as progress:
        task = progress.add_task("Solving...", total=N_STATES_CUBE_2 if cube.size == 2 else N_STATES_CUBE_3)

        for depth in range(max_depth + 1):
            cube.faces = c0.copy()
            solution_path, visited = dfs(cube, depth, progress, task)
            total_visited += visited
            if solution_path is not None:
                return solution_path, total_visited

    return None, total_visited

def dfs(
    cube: Cube, max_depth: int, progress: Progress, task: TaskID
) -> tuple[list[str] | None, int]:
    queue = deque([(cube.compress(), np.array([], dtype=np.uint8))])

    i = 0
    last_update = 0
    while queue:
        current_state, move_seq = queue.pop()
        cube.decompress(current_state)  # updates cube.faces

        i += 1
        if i - last_update >= 1000:
            progress.update(
                task,
                advance=(i - last_update),
                description=f"Depth: {max_depth} (Queue: {len(queue)})",
            )
            last_update = i

        if len(move_seq) < max_depth:
            for move_val in cube.possible_moves:
                move = Move(move_val)
                cube.move(move)

                new_move_seq = np.append(move_seq.copy(), [move.value])

                if cube.is_solution():
                    progress.update(task, advance=(i - last_update))
                    return [Move(x).name for x in new_move_seq], i

                queue.append((cube.compress(), new_move_seq))

                cube.move(move.inverse())  # Undo the move

    progress.update(task, advance=(i - last_update))
    return None, i
