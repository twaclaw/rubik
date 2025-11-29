import argparse
import time

from rich.console import Console

from .cube import Cube


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

    console = Console()
    cube = Cube(
        initial=args.cube_string,
        size=args.size,
        number_of_scramble_moves=args.number_of_scrambles,
    )
    cube0 = cube.copy()

    if args.size > 2:
        console.print(
            "[bold red]Warning: Solving cubes larger than 2x2 may be very slow or even intractable![/bold red]"
        )
        if args.size == 3:
            console.print(
                "[bold yellow]Use the kociemba command line tool for faster solving of 3x3 cubes[/bold yellow]"
            )

    cube_string = cube.to_string()
    algorithm: str = ""

    if args.algorithm == "bfs":
        from .bfs import ALGORITHM, bfs

        t0 = time.perf_counter()
        solution, n_visited = bfs(cube)
        t = time.perf_counter() - t0
        algorithm = ALGORITHM

    console.rule("[bold] Results")
    console.print(f"Algorithm: {algorithm}")
    console.print(f"Cube String: [bold yellow]{cube_string}[/bold yellow]")
    solution_str = "".join(solution) if solution is not None else "No solution found"
    console.print(f"Solution: [bold cyan]{solution_str}[/bold cyan]")
    console.print(f"Solved in [bold]{t:.2f} s[/bold].")
    console.rule("[bold] Original Cube")
    cube0.plot_cube()
    console.rule("[bold] Solved Cube")
    cube.plot_cube()
