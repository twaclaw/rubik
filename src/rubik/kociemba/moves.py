"""Movetables describe the transformation of the coordinates by cube moves."""

import os
from os import path

import numpy as np
from rich.console import Console
from rich.progress import track

from .cubie import BasicMoves, Color, CubieCube
from .defs import Constants as k


class Moves:
    def __init__(self, folder: str = k.FOLDER, show_progress: bool = True):
        self.folder = folder
        self.show_progress = show_progress
        self.console = Console()

        if not os.path.exists(self.folder):
            os.mkdir(self.folder)

        self.twist_move = None
        self.flip_move = None
        self.slice_sorted_move = None
        self.u_edges_move = None
        self.d_edges_move = None
        self.ud_edges_move = None
        self.corners_move = None

    def create_tables(self):
        self.create_twist_move_table()
        self.create_flip_move_table()
        self.create_slice_sorted_move_table()
        self.create_u_edges_move_table()
        self.create_d_edges_move_table()
        self.create_ud_edges_move_table()
        self.create_corners_move_table()

    def create_twist_move_table(self):
        """Move table for the twists of the corners.

        The twist coordinate describes the 3^7 = 2187 possible orientations of the 8 corners
        0 <= twist < 2187 in phase 1, twist = 0 in phase 2
        """
        fname = "move_twist.npy"
        fpath = path.join(self.folder, fname)
        if not path.isfile(fpath):
            # if self.show_progress:
            # self.console.print(f"[bold blue]Creating {fname} table...[/bold blue]")
            self.twist_move = np.zeros(k.N_TWIST * k.N_MOVE, dtype=np.uint16)
            a = CubieCube()

            iterator = range(k.N_TWIST)
            if self.show_progress:
                iterator = track(
                    iterator,
                    description=f"Generating {fname}...".ljust(
                        k.PROGRESS_BAR_DESC_WIDTH
                    ),
                )

            for i in iterator:
                a.set_twist(i)
                for j in Color:  # six faces U, R, F, D, L, B
                    for m in range(
                        3
                    ):  # three moves for each face, for example U, U2, U3 = U'
                        a.corner_multiply(BasicMoves[j])
                        self.twist_move[k.N_MOVE * i + 3 * j + m] = a.get_twist()
                    a.corner_multiply(BasicMoves[j])  # 4. move restores face
            np.save(fpath, self.twist_move)
        else:
            if self.show_progress:
                self.console.print(f"[bold green]Loading {fname} table...[/bold green]")
            self.twist_move = np.load(fpath, mmap_mode="r")
        return self.twist_move

    def create_flip_move_table(self):
        """Move table for the flip of the edges.

        The flip coordinate describes the 2^11 = 2048 possible orientations of the 12 edges
        0 <= flip < 2048 in phase 1, flip = 0 in phase 2
        """
        fname = "move_flip.npy"
        fpath = path.join(self.folder, fname)
        if not path.isfile(fpath):
            # if self.show_progress:
            # self.console.print(f"[bold blue]Creating {fname} table...[/bold blue]")
            self.flip_move = np.zeros(k.N_FLIP * k.N_MOVE, dtype=np.uint16)
            a = CubieCube()

            iterator = range(k.N_FLIP)
            if self.show_progress:
                iterator = track(
                    iterator,
                    description=f"Generating {fname}...".ljust(
                        k.PROGRESS_BAR_DESC_WIDTH
                    ),
                )

            for i in iterator:
                a.set_flip(i)
                for j in Color:
                    for m in range(3):
                        a.edge_multiply(BasicMoves[j])
                        self.flip_move[k.N_MOVE * i + 3 * j + m] = a.get_flip()
                    a.edge_multiply(BasicMoves[j])
            np.save(fpath, self.flip_move)
        else:
            if self.show_progress:
                self.console.print(f"[bold green]Loading {fname} table...[/bold green]")
            self.flip_move = np.load(fpath, mmap_mode="r")
        return self.flip_move

    def create_slice_sorted_move_table(self):
        """Move table for the four UD-slice edges FR, FL, Bl and BR.

        The slice_sorted coordinate describes the 12!/8! = 11880 possible positions of the FR, FL, BL and BR edges.
        Though for phase 1 only the "unsorted" slice coordinate with Binomial(12,4) = 495 positions is relevant, using the
        slice_sorted coordinate gives us the permutation of the FR, FL, BL and BR edges at the beginning of phase 2 for free.
        0 <= slice_sorted < 11880 in phase 1, 0 <= slice_sorted < 24 in phase 2, slice_sorted = 0 for solved cube
        """
        fname = "move_slice_sorted.npy"
        fpath = path.join(self.folder, fname)
        if not path.isfile(fpath):
            # if self.show_progress:
            # self.console.print(f"[bold blue]Creating {fname} table...[/bold blue]")
            self.slice_sorted_move = np.zeros(
                k.N_SLICE_SORTED * k.N_MOVE, dtype=np.uint16
            )
            a = CubieCube()

            iterator = range(k.N_SLICE_SORTED)
            if self.show_progress:
                iterator = track(
                    iterator,
                    description=f"Generating {fname}...".ljust(
                        k.PROGRESS_BAR_DESC_WIDTH
                    ),
                )

            for i in iterator:
                a.set_slice_sorted(i)
                for j in Color:
                    for m in range(3):
                        a.edge_multiply(BasicMoves[j])
                        self.slice_sorted_move[k.N_MOVE * i + 3 * j + m] = (
                            a.get_slice_sorted()
                        )
                    a.edge_multiply(BasicMoves[j])
            np.save(fpath, self.slice_sorted_move)
        else:
            if self.show_progress:
                self.console.print(f"[bold green]Loading {fname} table...[/bold green]")
            self.slice_sorted_move = np.load(fpath, mmap_mode="r")
        return self.slice_sorted_move

    def create_u_edges_move_table(self):
        """Move table for the u_edges coordinate for transition phase 1 -> phase 2

        The u_edges coordinate describes the 12!/8! = 11880 possible positions of the UR, UF, UL and UB edges. It is needed at
        the end of phase 1 to set up the coordinates of phase 2
        0 <= u_edges < 11880 in phase 1, 0 <= u_edges < 1680 in phase 2, u_edges = 1656 for solved cube.
        """
        fname = "move_u_edges.npy"
        fpath = path.join(self.folder, fname)
        if not path.isfile(fpath):
            # if self.show_progress:
            # self.console.print(f"[bold blue]Creating {fname} table...[/bold blue]")
            self.u_edges_move = np.zeros(k.N_SLICE_SORTED * k.N_MOVE, dtype=np.uint16)
            a = CubieCube()

            iterator = range(k.N_SLICE_SORTED)
            if self.show_progress:
                iterator = track(
                    iterator,
                    description=f"Generating {fname}...".ljust(
                        k.PROGRESS_BAR_DESC_WIDTH
                    ),
                )

            for i in iterator:
                a.set_u_edges(i)
                for j in Color:
                    for m in range(3):
                        a.edge_multiply(BasicMoves[j])
                        self.u_edges_move[k.N_MOVE * i + 3 * j + m] = a.get_u_edges()
                    a.edge_multiply(BasicMoves[j])
            np.save(fpath, self.u_edges_move)
        else:
            if self.show_progress:
                self.console.print(f"[bold green]Loading {fname} table...[/bold green]")
            self.u_edges_move = np.load(fpath, mmap_mode="r")
        return self.u_edges_move

    def create_d_edges_move_table(self):
        """Move table for the d_edges coordinate for transition phase 1 -> phase 2.

        The d_edges coordinate describes the 12!/8! = 11880 possible positions of the DR, DF, DL and DB edges. It is needed at
        the end of phase 1 to set up the coordinates of phase 2
        0 <= d_edges < 11880 in phase 1, 0 <= d_edges < 1680 in phase 2, d_edges = 0 for solved cube.
        """
        fname = "move_d_edges.npy"
        fpath = path.join(self.folder, fname)
        if not path.isfile(fpath):
            # if self.show_progress:
            # self.console.print(f"[bold blue]Creating {fname} table...[/bold blue]")
            self.d_edges_move = np.zeros(k.N_SLICE_SORTED * k.N_MOVE, dtype=np.uint16)
            a = CubieCube()

            iterator = range(k.N_SLICE_SORTED)
            if self.show_progress:
                iterator = track(
                    iterator,
                    description=f"Generating {fname}...".ljust(
                        k.PROGRESS_BAR_DESC_WIDTH
                    ),
                )

            for i in iterator:
                a.set_d_edges(i)
                for j in Color:
                    for m in range(3):
                        a.edge_multiply(BasicMoves[j])
                        self.d_edges_move[k.N_MOVE * i + 3 * j + m] = a.get_d_edges()
                    a.edge_multiply(BasicMoves[j])
            np.save(fpath, self.d_edges_move)
        else:
            if self.show_progress:
                self.console.print(f"[bold green]Loading {fname} table...[/bold green]")
            self.d_edges_move = np.load(fpath, mmap_mode="r")
        return self.d_edges_move

    def create_ud_edges_move_table(self):
        """Move table for the edges in the U-face and D-face.

        The ud_edges coordinate describes the 40320 permutations of the edges UR, UF, UL, UB, DR, DF, DL and DB in phase 2
        ud_edges undefined in phase 1, 0 <= ud_edges < 40320 in phase 2, ud_edges = 0 for solved cube.
        """
        fname = "move_ud_edges.npy"
        fpath = path.join(self.folder, fname)
        if not path.isfile(fpath):
            # if self.show_progress:
            # self.console.print(f"[bold blue]Creating {fname} table...[/bold blue]")
            self.ud_edges_move = np.zeros(k.N_UD_EDGES * k.N_MOVE, dtype=np.uint16)
            a = CubieCube()

            iterator = range(k.N_UD_EDGES)
            if self.show_progress:
                iterator = track(
                    iterator,
                    description=f"Generating {fname}...".ljust(
                        k.PROGRESS_BAR_DESC_WIDTH
                    ),
                )

            for i in iterator:
                a.set_ud_edges(i)
                for j in Color:
                    for m in range(3):
                        a.edge_multiply(BasicMoves[j])
                        # only R2, F2, L2 and B2 in phase 2
                        if j in [Color.R, Color.F, Color.L, Color.B] and m != 1:
                            continue
                        self.ud_edges_move[k.N_MOVE * i + 3 * j + m] = a.get_ud_edges()
                    a.edge_multiply(BasicMoves[j])
            np.save(fpath, self.ud_edges_move)
        else:
            if self.show_progress:
                self.console.print(f"[bold green]Loading {fname} table...[/bold green]")
            self.ud_edges_move = np.load(fpath, mmap_mode="r")
        return self.ud_edges_move

    def create_corners_move_table(self):
        """Move table for the corners coordinate in phase 2

        The corners coordinate describes the 8! = 40320 permutations of the corners.
        0 <= corners < 40320 defined but unused in phase 1, 0 <= corners < 40320 in phase 2, corners = 0 for solved cube
        """
        fname = "move_corners.npy"
        fpath = path.join(self.folder, fname)
        if not path.isfile(fpath):
            # if self.show_progress:
            # self.console.print(f"[bold blue]Creating {fname} table...[/bold blue]")
            self.corners_move = np.zeros(k.N_CORNERS * k.N_MOVE, dtype=np.uint16)
            a = CubieCube()

            iterator = range(k.N_CORNERS)
            if self.show_progress:
                iterator = track(
                    iterator,
                    description=f"Generating {fname}...".ljust(
                        k.PROGRESS_BAR_DESC_WIDTH
                    ),
                )

            for i in iterator:
                a.set_corners(i)
                for j in Color:
                    for m in range(3):
                        a.corner_multiply(BasicMoves[j])
                        self.corners_move[k.N_MOVE * i + 3 * j + m] = a.get_corners()
                    a.corner_multiply(BasicMoves[j])
            np.save(fpath, self.corners_move)
        else:
            if self.show_progress:
                self.console.print(f"[bold green]Loading {fname} table...[/bold green]")
            self.corners_move = np.load(fpath, mmap_mode="r")
        return self.corners_move
