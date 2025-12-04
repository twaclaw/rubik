from collections import deque

import numpy as np
from rich.progress import Progress

from .cube import Cube, Move, Results

INFO: dict[str, str] = {
    "algorithm": "Breadth-First Search (BFS)",
}

def bfs(cube: Cube) -> Results:
    # if cube.size != 2:
    #     raise NotImplementedError(
    #         "Are you nuts, BFS, in Python, for a cube larger than 2x2?"
    #     )

    if cube.is_solution():
        return Results()

    queue = deque([(cube.compress(), np.array([], dtype=np.uint8))])  # (compressed_state, move_sequence)

    i = 0
    visited = {cube.hashable()}

    np.random.shuffle(cube.possible_moves)


    with Progress() as progress:
        task = progress.add_task("Solving...", total=None)

        while queue:
            i += 1
            current_state, move_seq = queue.popleft()

            if i % 1000 == 0:
                progress.update(
                    task,
                    completed=len(visited),
                    description=f"States: {len(visited):,} (Queue: {len(queue):,})",
                )

            cube.decompress(current_state)  # updates cube.faces

            for move_val in cube.possible_moves:
                move = Move(move_val)
                cube.move(move)

                new_move_seq = np.append(move_seq.copy(), [move.value])
                if cube.is_solution():
                    return Results(solution_path=[Move(x).name for x in new_move_seq], nvisited=i)

                hash_value = cube.hashable()
                if hash_value not in visited:
                    visited.add(hash_value)
                    queue.append((cube.compress(), new_move_seq))

                cube.move(move.inverse())  # Undo the move

    return Results(nvisited=len(visited))

