import argparse
import time

from rich.console import Console

from .coord import Coord
from .cubie import Symmetries
from .moves import Moves
from .pruning import Pruning
from .solver import Solver

console = Console()


def create_video(args, moves_ph1, moves_ph2, moves_ph1_htm=None, moves_ph2_htm=None, format:str = "mp4"):
    import os
    import sys
    from contextlib import contextmanager

    from manim import tempconfig

    from rubik.animation.manim_rubik import RubiksCubeAnimation, RubiksCubeStatic

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
    elif format == "png":
        manim_config["write_to_movie"] = False
        manim_config["save_last_frame"] = True
        manim_config["format"] = "png"
        manim_config["output_file"] = "image.png"

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
            if format == "png":
                scene = RubiksCubeStatic(
                    moves_ph1=moves_ph1, moves_ph2=moves_ph2, initial_state=args.cube_string,
                    moves_ph1_htm=moves_ph1_htm, moves_ph2_htm=moves_ph2_htm
                )
            else:
                scene = RubiksCubeAnimation(
                    moves_ph1=moves_ph1, moves_ph2=moves_ph2, initial_state=args.cube_string,
                    moves_ph1_htm=moves_ph1_htm, moves_ph2_htm=moves_ph2_htm
                )
            scene.render()


def create_tables(args):
    start_time = time.time()

    symmetries = Symmetries(folder=args.path, show_progress=args.verbose)
    symmetries.create_tables()

    moves = Moves(folder=args.path, show_progress=args.verbose)
    moves.create_tables()

    pruning = Pruning(folder=args.path, show_progress=args.verbose)
    pruning.create_tables()

    coord = Coord(folder=args.path, show_progress=args.verbose)
    coord.create_tables()

    end_time = time.time()
    duration = end_time - start_time
    console.print(f"[bold green]Tables created in {duration:.2f} seconds.[/bold green]")


def call_solver(args):
    from rubik.cube import Cube
    from rubik.kociemba.cubie import CubieCube

    solver = Solver(folder=args.path, show_progress=args.verbose)
    cc = CubieCube()
    if not args.cube_string:
        c = Cube(initial="random", number_of_scramble_moves=args.num_scrambles)
        cc.from_cube(c)
        args.cube_string = cc.to_string()
    else:
        cc.from_string(args.cube_string)
        c = cc.to_cube()

    r = solver.solve(args.cube_string)

    ph1_str = "".join(r.ph1_htm)
    ph2_str = "".join(r.ph2_htm)
    console.rule("[bold] Results")
    console.print("Algorithm: Kociemba's Two-Phase")
    console.print(f"Cube String: [bold yellow]{args.cube_string}[/bold yellow]")
    console.print(
        f"Solution HTM: [bold cyan]{ph1_str}[/bold cyan][bold magenta]{ph2_str}[/bold magenta]"
    )
    console.print(f"Solved in [bold]{r.execution_time * 1000:.2f} ms[/bold].")

    console.rule("[bold] Original Cube")
    c.plot_cube(plot_with_labels=args.with_labels)
    c.moves(ph1_str)
    console.rule("[bold] After Phase 1")
    console.print(f"[bold cyan]Phase 1 moves: {ph1_str}[/bold cyan]")
    c.plot_cube(plot_with_labels=args.with_labels)
    c.moves(ph2_str)
    console.rule("[bold] Solved Cube")
    console.print(f"[bold magenta]Phase 2 moves: {ph2_str}[/bold magenta]")
    c.plot_cube(plot_with_labels=args.with_labels)

    if args.video:
        console.rule("[bold] Generating Video")
        create_video(args, r.ph1_str, r.ph2_str, moves_ph1_htm=r.ph1_htm, moves_ph2_htm=r.ph2_htm)
        console.print("[bold green]Video generated as video.mp4[/bold green]")

    if args.gif:
        console.rule("[bold] Generating GIF")
        create_video(args, r.ph1_str, r.ph2_str, moves_ph1_htm=r.ph1_htm, moves_ph2_htm=r.ph2_htm, format="gif")
        console.print("[bold green]GIF generated as video.gif[/bold green]")

    if args.image:
        console.rule("[bold] Generating Image")
        create_video(args, r.ph1_str, r.ph2_str, moves_ph1_htm=r.ph1_htm, moves_ph2_htm=r.ph2_htm, format="png")
        console.print("[bold green]Image generated as image.png[/bold green]")


def main():
    parser = argparse.ArgumentParser(description="Kociemba Algorithm")
    parser.add_argument("--verbose", action="store_true", help="Show progress")

    subparsers = parser.add_subparsers(dest="command", required=True)

    create_table_parser = subparsers.add_parser(
        "create-tables", help="Create symmetry tables"
    )

    create_table_parser.add_argument(
        "--path", required=True, help="Path to store the tables"
    )
    solver_parser = subparsers.add_parser("solve", help="Solve a given cube")
    solver_parser.add_argument(
        "--path", required=False, help="Path to the tables", default="./tables"
    )
    solver_parser.add_argument(
        "-c", "--cube-string", required=False, help="String representation of the cube"
    )
    solver_parser.add_argument(
        "-n",
        "--num-scrambles",
        type=int,
        required=False,
        help="Number of random scramble moves if cube string not provided. Ignored if cube string is given.",
        default=20,
    )
    solver_parser.add_argument(
        "--with-labels",
        action="store_true",
        help="Plot 2D  cubes with labels",
    )

    solver_parser.add_argument(
        "--video", action="store_true", help="Generate a video of the solution"
    )
    solver_parser.add_argument(
        "--image", action="store_true", help="Generate an image of the solution"
    )
    solver_parser.add_argument(
        "--gif", action="store_true", help="Generate a GIF of the solution"
    )

    args = parser.parse_args()

    if args.command == "create-tables":
        create_tables(args.path, args.verbose)
    elif args.command == "solve":
        call_solver(args)


if __name__ == "__main__":
    main()
