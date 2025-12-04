from collections import deque

import numpy as np
from rich.progress import Progress

from .cube import Cube, Move, Results

INFO: dict[str, str] = {
    "algorithm": "Met-in-the-middle: Bidirectional Breadth-First Search (Bi-BFS)",
}


def bi_bfs(cube: Cube) -> Results:
    """
    Bidirectional BFS. Meet in the middle.
    This is just a possible way of implemented bidirectional BFS. This one doesn't guarantee that the shortest path is found.
    A more general way would be to have a function that expands one the frontier and call this
    function from a terminate function that evaluates when to stop the search.
    """
    if cube.is_solution():
        return Results()

    solved_cube = Cube(size=cube.size, initial="solved")

    queue_front = deque([(cube.compress(), np.array([], dtype=np.uint8))])
    queue_back = deque([(solved_cube.compress(), np.array([], dtype=np.uint8))])

    states_processed = 0
    visited_front: dict[bytes, np.ndarray] = {}
    visited_front[cube.hashable()] = np.array([], dtype=np.uint8)
    visited_back: dict[bytes, np.ndarray] = {}
    visited_back[solved_cube.hashable()] = np.array([], dtype=np.uint8)

    with Progress() as progress:
        task = progress.add_task("Solving...", total=None)

        while queue_front or queue_back:
            states_processed += 1
            state_front, seq_front = queue_front.popleft()
            state_back, seq_back = queue_back.popleft()

            if states_processed % 1000 == 0:
                progress.update(
                    task,
                    completed=len(visited_front) + len(visited_back),
                    description=f"States: {len(visited_front) + len(visited_back):,} (Queue: {len(queue_front) + len(queue_back):,})",
                )

            cube.decompress(state_front)
            solved_cube.decompress(state_back)

            # Forward search
            for move_val in cube.possible_moves:
                move = Move(move_val)
                cube.move(move)

                if cube.is_solution():
                    solution_path = cube.solution_path(np.append(seq_front, [move]))
                    return Results(solution_path=solution_path, nvisited=len(visited_front))

                hash_value = cube.hashable()

                new_seq = np.append(seq_front.copy(), [move.value])

                if hash_value in visited_back:
                    seq_back_sol = cube.reverse_sequence(visited_back[hash_value])

                    forward_path = [Move(x).name for x in new_seq]
                    backward_path = [Move(x).name for x in visited_back[hash_value]]
                    solution_path = cube.solution_path(np.append(new_seq, seq_back_sol))
                    n_visited = len(visited_front) + len(visited_back)

                    return Results(
                        forward_path=forward_path,
                        backward_path=backward_path,
                        solution_path=solution_path,
                        nvisited=n_visited,
                    )

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

                    forward_path = [Move(x).name for x in seq_front_sol]
                    backward_path = [Move(x).name for x in new_seq]
                    solution_path = cube.solution_path(
                        np.append(seq_front_sol, cube.reverse_sequence(new_seq))
                    )
                    n_visited = len(visited_front) + len(visited_back)

                    return Results(
                        forward_path=forward_path,
                        backward_path=backward_path,
                        solution_path=solution_path,
                        nvisited=n_visited,
                    )

                if hash_value not in visited_back:
                    visited_back[hash_value] = new_seq
                    queue_back.append((solved_cube.compress(), new_seq))

                solved_cube.move(move.inverse())  # Undo the move

    return Results(nvisited=len(visited_front))
