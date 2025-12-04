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


def create_video(args, moves_fwd, moves_bcw, cube_string, format:str = "mp4"):


    import os
    import sys
    from contextlib import contextmanager

    from manim import tempconfig

    from rubik.animation.manim_rubik import MeetInTheMiddleAnimation

    # Set configuration for low quality video (faster)
    manim_config = {
        "pixel_height": 480,
        "pixel_width": 854,
        "frame_rate": 15,
        "quality": "low_quality",
        "preview": False,
        "write_to_movie": True,
        "save_last_frame": False,
        "output_file": f"video.{format}",
        "verbosity": "CRITICAL",
        "progress_bar": "none",
    }

    if format == "gif":
        manim_config["format"] = "gif"

    @contextmanager
    def suppress_stdout_stderr():
        with open(os.devnull, "w") as devnull:
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            sys.stdout = devnull
            sys.stderr = devnull
            try:
                yield
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr

    with tempconfig(manim_config):
        with suppress_stdout_stderr():
            scene = MeetInTheMiddleAnimation(
                moves_1=moves_fwd, moves_2=moves_bcw, initial_state_1=cube_string,
                initial_state_2="UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB"
            )
            scene.render()


def run_solver(cube: Cube, args):
    algorithm = args.algorithm
    solver_fn: callable | None = None
    if algorithm == "bfs":
        solver_fn = bfs
        algorithm_name = bfs_info["algorithm"]
    elif algorithm == "iddfs":
        solver_fn = ids
        algorithm_name = ids_info["algorithm"]
    elif algorithm == "bi-bfs":
        solver_fn = bi_bfs
        algorithm_name = bi_bfs_info["algorithm"]
    elif algorithm == "bi-iddfs":
        solver_fn = bi_ids
        algorithm_name = bi_ids_info["algorithm"]
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    t0 = time.perf_counter()
    cube_string = cube.to_string()
    cube0 = cube.copy()

    results = None
    try:
        results = solver_fn(cube)
        solution = results.solution_path
        n_visited = results.nvisited
    except KeyboardInterrupt:
        console.print("[bold red]Solving interrupted by user[/bold red]")
        solution, n_visited = None, 0
    except Exception as ex:
        console.print(f"[bold red]Exception: {ex}[/bold red]")
        solution, n_visited = None, 0

    t = time.perf_counter() - t0

    console.rule("[bold] Results")
    console.print(f"Algorithm: [bold]{algorithm_name}[/bold]")
    console.print(f"Number of visited states: [bold]{n_visited:,} - ({n_visited//t:,}/s)[/bold]")
    console.print(f"Cube String: [bold cyan]{cube_string}[/bold cyan]")
    solution_str = "".join(solution) if solution is not None else "No solution found"
    console.print(f"Solution: [bold cyan]{solution_str}[/bold cyan]")
    console.print(f"Elapsed time: [bold]{t:.2f} s[/bold].")
    console.rule("[bold] Original Cube")
    cube0.plot_cube()
    if solution is not None:
        console.rule("[bold] Solved Cube")
        cube0.moves(solution)
        cube0.plot_cube()
        if args.size == 3:
            if algorithm.startswith("bi-") and args.video:
                console.rule("[bold]Generating video")
                moves_fwd = results.forward_path
                moves_bcw = results.backward_path
                create_video(args, moves_fwd, moves_bcw, cube_string, format="mp4")
                console.print("[bold green]Video saved as video.mp4[/bold green]")
            if algorithm.startswith("bi-") and args.gif:
                console.print("[bold]Generating GIF...[/bold]")
                moves_fwd = results.forward_path
                moves_bcw = results.backward_path
                create_video(args, moves_fwd, moves_bcw, cube_string, format="gif")
                console.print("[bold green]GIF saved as video.gif[/bold green]")



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

    parser.add_argument("--video", action="store_true", help="Generate a video of the solution")
    parser.add_argument("--gif", action="store_true", help="Generate a GIF of the solution")

    args = parser.parse_args()

    cube = Cube(
        initial=args.cube_string,
        size=args.size,
        number_of_scramble_moves=args.number_of_scrambles,
    )

    if args.size > 2:
        console.rule("[bold] Warnings")
        console.print(
            "[bold]Solving cubes larger than 2x2 may be very slow or even intractable![/bold]"
        )
        # if args.size == 3:
        #     console.print(
        #         "[bold]Use the [yellow]kociemba[/yellow] command line tool for faster solving of 3x3 cubes[/bold]"
        #     )

    run_solver(cube, args)