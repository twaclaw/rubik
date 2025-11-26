import argparse
import time

from rich.console import Console

from .coord import Coord
from .cubie import Symmetries
from .moves import Moves
from .pruning import Pruning

console = Console()


def main():
    parser = argparse.ArgumentParser(description="Kociemba table generation  CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    create_table_parser = subparsers.add_parser(
        "create-tables", help="Create symmetry tables"
    )
    create_table_parser.add_argument(
        "--path", required=True, help="Path to store the tables"
    )
    create_table_parser.add_argument(
        "--show-progress", action="store_true", help="Show progress bar"
    )

    args = parser.parse_args()

    if args.command == "create-tables":

        start_time = time.time()

        symmetries = Symmetries(folder=args.path, show_progress=args.show_progress)
        symmetries.create_tables()

        moves = Moves(folder=args.path, show_progress=args.show_progress)
        moves.create_tables()

        pruning = Pruning(folder=args.path, show_progress=args.show_progress)
        pruning.create_tables()

        coord = Coord(folder=args.path, show_progress=args.show_progress)
        coord.create_tables()

        end_time = time.time()
        duration = end_time - start_time
        console.print(
            f"[bold green]Tables created in {duration:.2f} seconds.[/bold green]"
        )


if __name__ == "__main__":
    main()
