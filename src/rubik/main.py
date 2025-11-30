import argparse
import time

from rich.console import Console

from .bfs import INFO as bfs_info
from .bfs import bfs
from .bi_bfs import INFO as bi_bfs_info
from .bi_bfs import bi_bfs
from .bi_ids import INFO as bi_ids_info
from .bi_ids import bi_ids
from .cube import Cube
from .ids import INFO as ids_info
from .ids import ids

console = Console()
def run_solver(cube: Cube, algorithm: str):
    solver_fn: callable | None = None
    if algorithm == "bfs":
        solver_fn = bfs
        algorithm = bfs_info["algorithm"]
    elif algorithm == "iddfs":
        solver_fn = ids
        algorithm = ids_info["algorithm"]
    elif algorithm == "bi-bfs":
        solver_fn = bi_bfs
        algorithm = bi_bfs_info["algorithm"]
    elif algorithm == "bi-iddfs":
        solver_fn = bi_ids
        algorithm = bi_ids_info["algorithm"]
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    t0 = time.perf_counter()
    cube_string = cube.to_string()
    cube0 = cube.copy()

    try:
        solution, n_visited = solver_fn(cube)
    except KeyboardInterrupt:
        console.print("[bold red]Solving interrupted by user[/bold red]")
        solution, n_visited = None, 0
    except Exception as ex:
        console.print(f"[bold red]Exception: {ex}[/bold red]")
        solution, n_visited = None, 0

    t = time.perf_counter() - t0

    console.rule("[bold] Results")
    console.print(f"Algorithm: [bold]{algorithm}[/bold]")
    console.print(f"Number of visited states: [bold]{n_visited:,} - ({n_visited//t:,}/s)[/bold]")
    console.print(f"Cube String: [bold cyan]{cube_string}[/bold cyan]")
    solution_str = "".join(solution) if solution is not None else "No solution found"
    console.print(f"Solution: [bold cyan]{solution_str}[/bold cyan]")
    console.print(f"Solved in [bold]{t:.2f} s[/bold].")
    console.rule("[bold] Original Cube")
    cube0.plot_cube()
    if solution is not None:
        console.rule("[bold] Solved Cube")
        cube0.moves(solution)
        cube0.plot_cube()


def main():
    parser = argparse.ArgumentParser(description="Rubik's Cube Solver")
    parser.add_argument(
        "-c",
        "--cube-string",
        type=str,
        default="random",
        help="Initial cube state as a string or 'random' or 'solved'",
    )
    parser.add_argument(
        "-n",
        "--number-of-scrambles",
        type=int,
        default=10,
        help="Number of random scramble moves to apply if cube-string is 'random'. Default is 10",
    )
    parser.add_argument(
        "-s",
        "--size",
        type=int,
        default=3,
        help="Size of the cube. Default is 3 for a 3x3 cube",
    )
    parser.add_argument(
        "-a",
        "--algorithm",
        type=str,
        choices=["bfs", "bi-bfs", "iddfs", "bi-iddfs"],
        default="bi-iddfs",
        help="Solving algorithm to use: 'bfs' or 'bi_ids'. Default is 'bfs'",
    )
    args = parser.parse_args()

    cube = Cube(
        initial=args.cube_string,
        size=args.size,
        number_of_scramble_moves=args.number_of_scrambles,
    )

    if args.size > 2:
        console.print(
            "[bold red]Warning: Solving cubes larger than 2x2 may be very slow or even intractable![/bold red]"
        )
        if args.size == 3:
            console.print(
                "[bold yellow]Use the kociemba command line tool for faster solving of 3x3 cubes[/bold yellow]"
            )

    run_solver(cube, args.algorithm)