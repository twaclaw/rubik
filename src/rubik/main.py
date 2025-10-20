from .bfs import bfs
from .cube import Cube


def main():
    c = Cube(size=2, initial="random")
    c0 = c.faces.copy()

    c.plot_cube()
    solution_path, nodes_visted = bfs(c)
    if solution_path:
        print(f"Solved in {len(solution_path)} moves, visiting {nodes_visted} nodes.")
        print("Solution path:", solution_path)
        print(c.is_solution())

        c.plot_cube()
        c.faces = c0.copy()
        c.plot_cube()
        print(f"Applying solution moves: {solution_path}")
        c.moves(solution_path)
        c.plot_cube()