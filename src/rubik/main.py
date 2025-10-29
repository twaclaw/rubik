from .bi_ids import bi_ids
from .cube import Cube


def main():
    c = Cube(size=3, initial="random", number_of_scramble_moves=20)
    c0 = c.faces.copy()

    c.plot_cube()
    solution_path, nodes_visited = bi_ids(c, max_depth=11)
    if solution_path:
        print(f"Solved in {len(solution_path)} moves, visiting {nodes_visited} nodes.")
        print("Solution path:", solution_path)
        print(c.is_solution())

        c.plot_cube()
        c.faces = c0.copy()
        c.plot_cube()
        print(f"Applying solution moves: {solution_path}")
        c.moves(solution_path)
        c.plot_cube()