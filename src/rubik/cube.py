from dataclasses import dataclass
from enum import IntEnum
from typing import Literal

import numpy as np
from rich.columns import Columns
from rich.console import Console, Group
from rich.table import Table
from rich.text import Text


class Face(IntEnum):
    U = 0  # Up
    R = 1  # Right
    F = 2  # Front
    D = 3  # Down
    L = 4  # Left
    B = 5  # Back


class RotationType(IntEnum):
    CLOCKWISE = -1
    ACW = 1  # AntiClockWise

    def inverse(self) -> "RotationType":
        return RotationType(-self.value)


class SliceType(IntEnum):
    TOP_ROW = 0
    BOTTOM_ROW = 1
    LEFT_COL = 2
    RIGHT_COL = 3
    MIDDLE_ROW = 4
    MIDDLE_COL = 5


@dataclass
class CycleElement:
    face: Face
    slice_type: SliceType
    reverse: bool  # whether to reverse when this element is the source in the clowckwise rotation
    reverse_acw: bool  # whether to reverse when this element is the source in the anti-clockwise rotation


@dataclass
class Rotation:
    """
    Elements:
    - face: the main face to be rotated according to `rotation`
    - rotation: the direction of the rotation
    - cycle: list[CycleElement] a list representing the adjacent faces that rotate when the main face rotates.
      cycle[0] -> cycle[1] -> cycle[2] -> cycle[3] -> cycle[0]
      In the inverse movement, the direction is changed.
    """

    face: Face
    rotation: RotationType
    cycle: list[CycleElement]

    def inverse(self) -> "Rotation":
        """
        Returns the inverse of this rotation. The inverse of a rotation is a rotation of the same face in the oposite direction,
        for instance anti clockwise instead of clockwise.
        """
        cycle_reversed = []
        for i in range(4):
            rev_cycle = self.cycle[4 - i - 1]
            cycle_reversed.append(
                CycleElement(
                    rev_cycle.face,
                    rev_cycle.slice_type,
                    rev_cycle.reverse_acw,
                    rev_cycle.reverse,
                )
            )

        return Rotation(self.face, self.rotation.inverse(), cycle_reversed)


class Move(IntEnum):
    F = 0
    B = 1
    L = 2
    R = 3
    U = 4
    D = 5
    E = 6  # Equatorial slice, in the D direction
    M = 7  # Middle slice, in the L direction
    S = 8  # Standing slice, in the F direction
    f = 9
    b = 10
    l = 11
    r = 12
    u = 13
    d = 14
    e = 15
    m = 16
    s = 17

    def inverse(self) -> "Move":
        if self.value < 9:
            return Move(self.value + 9)
        else:
            return Move(self.value - 9)


# Apply to cubes of any size
default_rotations: dict[Move, Rotation] = {
    Move.F: Rotation(
        face=Face.F,
        rotation=RotationType.CLOCKWISE,
        cycle=[
            CycleElement(Face.L, SliceType.RIGHT_COL, True, False),
            CycleElement(Face.U, SliceType.BOTTOM_ROW, False, True),
            CycleElement(Face.R, SliceType.LEFT_COL, True, False),
            CycleElement(Face.D, SliceType.TOP_ROW, False, True),
        ],
    ),
    Move.B: Rotation(
        face=Face.B,
        rotation=RotationType.CLOCKWISE,
        cycle=[
            CycleElement(Face.D, SliceType.BOTTOM_ROW, True, False),
            CycleElement(Face.R, SliceType.RIGHT_COL, False, True),
            CycleElement(Face.U, SliceType.TOP_ROW, True, False),
            CycleElement(Face.L, SliceType.LEFT_COL, False, True),
        ],
    ),
    Move.L: Rotation(
        face=Face.L,
        rotation=RotationType.CLOCKWISE,
        cycle=[
            CycleElement(Face.D, SliceType.LEFT_COL, True, False),
            CycleElement(Face.B, SliceType.RIGHT_COL, True, True),
            CycleElement(Face.U, SliceType.LEFT_COL, False, True),
            CycleElement(Face.F, SliceType.LEFT_COL, False, False),
        ],
    ),
    Move.R: Rotation(
        face=Face.R,
        rotation=RotationType.CLOCKWISE,
        cycle=[
            CycleElement(Face.F, SliceType.RIGHT_COL, False, False),
            CycleElement(Face.U, SliceType.RIGHT_COL, True, False),
            CycleElement(Face.B, SliceType.LEFT_COL, True, True),
            CycleElement(Face.D, SliceType.RIGHT_COL, False, True),
        ],
    ),
    Move.U: Rotation(
        face=Face.U,
        rotation=RotationType.CLOCKWISE,
        cycle=[
            CycleElement(Face.L, SliceType.TOP_ROW, False, False),
            CycleElement(Face.B, SliceType.TOP_ROW, False, False),
            CycleElement(Face.R, SliceType.TOP_ROW, False, False),
            CycleElement(Face.F, SliceType.TOP_ROW, False, False),
        ],
    ),
    Move.D: Rotation(
        face=Face.D,
        rotation=RotationType.CLOCKWISE,
        cycle=[
            CycleElement(Face.F, SliceType.BOTTOM_ROW, False, False),
            CycleElement(Face.R, SliceType.BOTTOM_ROW, False, False),
            CycleElement(Face.B, SliceType.BOTTOM_ROW, False, False),
            CycleElement(Face.L, SliceType.BOTTOM_ROW, False, False),
        ],
    ),
}

cube_3_additional_rotations: dict[Move, Rotation] = {
    Move.E: Rotation(
        face=None,
        rotation=RotationType.CLOCKWISE,
        cycle=[
            CycleElement(Face.F, SliceType.MIDDLE_ROW, False, False),
            CycleElement(Face.R, SliceType.MIDDLE_ROW, False, False),
            CycleElement(Face.B, SliceType.MIDDLE_ROW, False, False),
            CycleElement(Face.L, SliceType.MIDDLE_ROW, False, False),
        ],
    ),
    Move.M: Rotation(
        face=None,
        rotation=RotationType.CLOCKWISE,
        cycle=[
            CycleElement(Face.F, SliceType.MIDDLE_COL, False, False),
            CycleElement(Face.U, SliceType.MIDDLE_COL, True, False),
            CycleElement(Face.B, SliceType.MIDDLE_COL, True, True),
            CycleElement(Face.D, SliceType.MIDDLE_COL, False, True),
        ],
    ),
    Move.S: Rotation(
        face=None,
        rotation=RotationType.CLOCKWISE,
        cycle=[
            CycleElement(Face.L, SliceType.MIDDLE_ROW, True, False),
            CycleElement(Face.U, SliceType.MIDDLE_ROW, False, True),
            CycleElement(Face.R, SliceType.MIDDLE_ROW, True, False),
            CycleElement(Face.D, SliceType.MIDDLE_ROW, False, True),
        ]
    ),
    # To be implemented if needed
}


class Cube:
    def __init__(
        self,
        size: int = 3,
        initial: np.ndarray | Literal["random", "solved"] = "random",
        number_of_scramble_moves: int = 10,
    ):
        self.size = size
        self.rotations = default_rotations
        if size == 3:
            self.rotations |= cube_3_additional_rotations
        self.rotations |= {
            k.inverse(): v.inverse() for k, v in self.rotations.items()
        }
        self.possible_moves = np.array(list(self.rotations.keys()))
        self.basic_moves = np.array(list(default_rotations.keys()))

        if isinstance(initial, np.ndarray):
            self.faces = initial.copy()
        elif isinstance(initial, str) and (initial == "random" or initial == "solved"):
            self.faces = np.arange(6, dtype=np.uint8)[:, np.newaxis, np.newaxis]
            self.faces = np.broadcast_to(self.faces, (6, size, size)).copy()
            self._solved = self.faces.copy()
            if initial == "random":
                initial_seq = []
                for _ in range(number_of_scramble_moves):
                    move = np.random.choice(self.basic_moves)
                    self.move(move)
                    initial_seq.append(Move(move).name)

    def _get_slice(self, slice_type: SliceType):
        if slice_type == SliceType.TOP_ROW:
            return (0, slice(None))
        elif slice_type == SliceType.BOTTOM_ROW:
            return (self.size - 1, slice(None))
        elif slice_type == SliceType.LEFT_COL:
            return (slice(None), 0)
        elif slice_type == SliceType.RIGHT_COL:
            return (slice(None), self.size - 1)
        elif slice_type == SliceType.MIDDLE_ROW:
            mid = self.size // 2
            return (mid, slice(None))
        elif slice_type == SliceType.MIDDLE_COL:
            mid = self.size // 2
            return (slice(None), mid)
        else:
            raise ValueError(f"Unknown slice type: {slice_type}")

    def compress(self) -> bytes:
        """Compresses the cube's state to use 4 bits per facelet instead of 8 bits.
        The size of the flattened cube is always even.
        """
        flat = self.faces.copy().flatten().reshape(-1, 2)
        compressed = (flat[:, 0] << 4) | flat[:, 1]
        return compressed.tobytes()

    def decompress(self, data: bytes):
        """Decompresses the cube's state from 4 bits per facelet back to 8 bits."""
        compressed = np.frombuffer(data, dtype=np.uint8)
        decompressed = np.empty((compressed.size * 2,), dtype=np.uint8)
        decompressed[0::2] = (compressed >> 4) & 0x0F
        decompressed[1::2] = compressed & 0x0F
        self.faces = decompressed.reshape((6, self.size, self.size)).copy()

    def move(self, move: Move):
        rotation = self.rotations[move]
        # Rotate the face itself
        if rotation.face is not None:
            self.faces[rotation.face] = np.rot90(
                self.faces[rotation.face], k=rotation.rotation.value
            )

        # Rotate the adjacent faces
        sources = []
        for i in range(4):
            src_idx = (i - 1) % 4
            face = rotation.cycle[src_idx].face
            slice_type = rotation.cycle[src_idx].slice_type
            data = self.faces[face][self._get_slice(slice_type)].copy()
            reverse = rotation.cycle[src_idx].reverse
            if reverse:
                data = data[::-1]
            sources.append(data)

        for i in range(4):
            dest = rotation.cycle[i]
            self.faces[dest.face][self._get_slice(dest.slice_type)] = sources[i]

    def moves(self, moves: list[str]):
        for move in moves:
            self.move(Move[move])

    def hashable(self) -> bytes:
        # Use bytes directly as key to avoid hash collisions. This is guaranteed to be unique.
        return self.faces.copy().tobytes()

        # return hash(self.compress())

    def is_solution(self) -> bool:
        """Verifies solution independent of rotations."""
        face_uniform = np.all(self.faces == self.faces[:, 0:1, 0:1], axis=(1, 2))
        return np.all(face_uniform)

    @staticmethod
    def reverse_sequence(sequence: np.ndarray) -> np.ndarray:
        return np.array([Move(x).inverse().value for x in sequence[::-1]])

    @staticmethod
    def solution_path(sequence: np.ndarray) -> list[str]:
        return [Move(x).name for x in sequence]

    def _color_to_rich(self, face: Face) -> str:
        color_map = {
            Face.F: "red",
            Face.U: "white",
            Face.D: "yellow",
            Face.L: "green",
            Face.R: "blue",
            Face.B: "orange3",
        }
        return color_map[face]

    def plot_face(self, face_index: int, print_table: bool = False) -> Table:
        table = Table(show_header=False, show_edge=True, expand=False, padding=(0, 0))

        for _ in range(self.size):
            table.add_column(justify="center", width=4)

        for row in range(self.size):
            # Add multiple rows for each face to make cells more square
            for _ in range(2):
                row_data = []
                for col in range(self.size):
                    face_value = self.faces[face_index, row, col]
                    rich_color = self._color_to_rich(Face(face_value))
                    colored_text = Text("████", style=f"{rich_color}")
                    row_data.append(colored_text)

                table.add_row(*row_data)

            if row < self.size - 1:
                table.add_section()

        if print_table:
            console = Console()
            console.print(table)

        return table

    def plot_face_with_labels(self, face_index: int, print_table: bool = False) -> Table:
        table = Table(show_header=False, show_edge=True, expand=False, padding=(0, 0))

        for _ in range(self.size):
            table.add_column(justify="center", width=4)

        for row in range(self.size):
            row_data = []
            for col in range(self.size):
                face_letter = Face(face_index).name
                position_number = row * self.size + col + 1
                face_value = self.faces[face_index, row, col]
                rich_color = self._color_to_rich(Face(face_value))
                label_text = f"[bold {rich_color}]{face_letter}{position_number}[/]"
                row_data.append(label_text)

            table.add_row(*row_data)

            if row < self.size - 1:
                table.add_section()

        if print_table:
            console = Console()
            console.print(table)

        return table

    def _plot_invisible_face(self) -> Text:
        total_lines = self.size * 2 + (self.size - 1)
        line_width = 5 * self.size + 1  # Approximate width including borders/spacing
        invisible_lines = []
        for _ in range(total_lines):
            invisible_lines.append(" " * line_width)

        return Text("\n".join(invisible_lines), style="")

    def _plot_invisible_face_with_labels(self) -> Table:
        table = Table(show_header=False, show_edge=True, show_lines=True, expand=False, padding=(0, 0), style="black")

        for _ in range(self.size):
            table.add_column(justify="center", width=4)

        return table

    def plot_cube(self, plot_with_labels: bool = False):
        """Plot the entire cube with all faces."""
        console = Console()

        face_plotter = (
            self.plot_face_with_labels if plot_with_labels else self.plot_face
        )
        # Create tables for each face
        up_table = face_plotter(Face.U)
        down_table = face_plotter(Face.D)
        front_table = face_plotter(Face.F)
        back_table = face_plotter(Face.B)
        left_table = face_plotter(Face.L)
        right_table = face_plotter(Face.R)

        invisible = self._plot_invisible_face_with_labels() if plot_with_labels else self._plot_invisible_face()

        top_row = Columns(
            [invisible, up_table], equal=True, expand=False, padding=(0, 0)
        )
        middle_row = Columns(
            [left_table, front_table, right_table, back_table],
            equal=True,
            expand=False,
            padding=(0, 0),
        )
        bottom_row = Columns(
            [invisible, down_table], equal=True, expand=False, padding=(0, 0)
        )

        # TODO: fix vertical spacing
        cube_layout = Group(top_row, middle_row, bottom_row)

        console.print(cube_layout)
