"""
Bi-directional Iterative Deepening Search (IDS)
"""

from collections import deque

import numpy as np
from rich.progress import Progress, TaskID

from .cube import Cube, Move, Results

INFO: dict[str, str] = {
    "algorithm": "Bidirectional Iterative Deepening Depth First Search (Bi-IDDFS)",
}


def bi_ids(cube: Cube, max_depth: int | None = None) -> Results:
    if cube.size > 3:
        raise NotImplementedError("Not implemented for cubes larger than 3x3")

    if max_depth is None:
        max_depth = 14 if cube.size == 2 else 20

    if cube.is_solution():
        return Results()

    visited_front: dict[bytes, np.ndarray] = {}
    visited_back: dict[bytes, np.ndarray] = {}

    c0 = cube.faces.copy()
    total_visited = 0

    with Progress() as progress:
        task = progress.add_task("Solving...", total=None)

        for depth in range(max_depth + 1):
            cube.faces = c0.copy()
            results = dfs(
                cube, depth, visited_front, visited_back, progress, task
            )
            total_visited += results.nvisited

            if results.solution_path:
                results.nvisited = total_visited
                return results

    return Results(nvisited=total_visited)


def dfs(
    cube: Cube,
    max_depth: int,
    visited_front: dict[bytes, np.ndarray],
    visited_back: dict[bytes, np.ndarray],
    progress: Progress,
    task: TaskID,
) -> Results:

    cube_solved = Cube(size=cube.size, initial="solved")

    queue_front = deque([(cube.compress(), np.array([], dtype=np.uint8))])
    queue_back = deque([(cube_solved.compress(), np.array([], dtype=np.uint8))])

    # print(f"DFS at depth {max_depth}, front visited: {len(visited_front)}, back visited: {len(visited_back)}")
    i = 0
    last_update = 0
    while queue_front:
        current_state, move_seq = queue_front.pop()
        cube.decompress(current_state)  # updates cube.faces

        i += 1
        if i - last_update >= 1000:
            progress.update(
                task,
                advance=(i - last_update),
                description=f"Depth: {max_depth} (Queue: {len(queue_front) + len(queue_back)})",
            )
            last_update = i

        if len(move_seq) < max_depth:
            for move_val in cube.possible_moves:
                move = Move(move_val)
                cube.move(move)

                new_move_seq = np.append(move_seq.copy(), [move.value])

                if cube.is_solution():
                    progress.update(task, advance=(i - last_update))
                    front = [Move(x).name for x in new_move_seq]
                    n_visited = len(visited_front) + len(visited_back)
                    return Results(solution_path=front, nvisited=n_visited)

                hash_value = cube.hashable()

                if hash_value in visited_back:
                    seq_back_sol = cube.reverse_sequence(visited_back[hash_value])
                    progress.update(task, advance=(i - last_update))

                    forward_path = [Move(x).name for x in new_move_seq]
                    backward_path = [Move(x).name for x in visited_back[hash_value]]
                    solution_path = cube.solution_path(
                        np.append(new_move_seq, seq_back_sol)
                    )
                    n_visited = len(visited_front) + len(visited_back)

                    return Results(
                        forward_path=forward_path,
                        backward_path=backward_path,
                        solution_path=solution_path,
                        nvisited=n_visited,
                    )

                if hash_value not in visited_front:
                    visited_front[hash_value] = new_move_seq

                queue_front.append((cube.compress(), new_move_seq))

                cube.move(move.inverse())  # Undo the move

    while queue_back:
        current_state, move_seq = queue_back.pop()
        cube_solved.decompress(current_state)  # updates cube.faces

        i += 1
        if i - last_update >= 1000:
            progress.update(
                task,
                advance=(i - last_update),
                description=f"Depth: {max_depth} (Queue: {len(queue_front) + len(queue_back)})",
            )
            last_update = i

        if len(move_seq) < max_depth:
            for move_val in cube_solved.possible_moves:
                move = Move(move_val)
                cube_solved.move(move)

                new_move_seq = np.append(move_seq.copy(), [move.value])

                hash_value = cube_solved.hashable()

                if hash_value in visited_front:
                    seq_front_sol = visited_front[hash_value]
                    progress.update(task, advance=(i - last_update))

                    forward_path = [Move(x).name for x in seq_front_sol]
                    backward_path = [Move(x).name for x in new_move_seq]
                    solution_path = cube_solved.solution_path(
                        np.append(
                            seq_front_sol, cube_solved.reverse_sequence(new_move_seq)
                        )
                    )
                    n_visited = len(visited_front) + len(visited_back)

                    return Results(
                        forward_path=forward_path,
                        backward_path=backward_path,
                        solution_path=solution_path,
                        nvisited=n_visited,
                    )

                if hash_value not in visited_back:
                    visited_back[hash_value] = new_move_seq

                queue_back.append((cube_solved.compress(), new_move_seq))

                cube_solved.move(move.inverse())  # Undo the move

    progress.update(task, advance=(i - last_update))
    return Results(nvisited=i)
