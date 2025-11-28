"""The cube on the coordinate level. It is described by a 3-tuple of natural numbers in phase 1 and phase 2."""

import os
from os import path

import numpy as np
from rich.console import Console
from rich.progress import track

from .cubie import CubieCube
from .cubie import Edge as Ed
from .cubie import Move, Symmetries
from .defs import Constants as k
from .moves import Moves
from .pruning import Pruning


class Coord:
    def __init__(self, folder: str = k.FOLDER, show_progress: bool = True):
        self.folder = folder
        self.show_progress = show_progress
        self.console = Console()

        if not os.path.exists(self.folder):
            os.mkdir(self.folder)

        self.u_edges_plus_d_edges_to_ud_edges = None

    def create_tables(self):
        self.create_phase2_edgemerge_table()

    def create_phase2_edgemerge_table(self):
        """phase2_edgemerge retrieves the initial phase 2 ud_edges coordinate from the u_edges and d_edges coordinates."""

        fname = "phase2_edgemerge.npy"
        fpath = path.join(self.folder, fname)
        if not path.isfile(fpath):
            self.u_edges_plus_d_edges_to_ud_edges = np.zeros(
                k.N_U_EDGES_PHASE2 * k.N_PERM_4, dtype=np.uint16
            )
            c_u = CubieCube()
            c_d = CubieCube()
            c_ud = CubieCube()

            edge_u = [Ed.UR, Ed.UF, Ed.UL, Ed.UB]
            edge_d = [Ed.DR, Ed.DF, Ed.DL, Ed.DB]
            edge_ud = [Ed.UR, Ed.UF, Ed.UL, Ed.UB, Ed.DR, Ed.DF, Ed.DL, Ed.DB]

            iterator = range(k.N_U_EDGES_PHASE2)
            if self.show_progress:
                iterator = track(
                    iterator,
                    description=f"Generating {fname}...".ljust(
                        k.PROGRESS_BAR_DESC_WIDTH
                    ),
                )

            for i in iterator:
                c_u.set_u_edges(i)
                for j in range(k.N_CHOOSE_8_4):
                    c_d.set_d_edges(j * k.N_PERM_4)
                    invalid = False
                    for e in edge_ud:
                        c_ud.edges[e, 0] = -1  # invalidate edges
                        if c_u.edges[e, 0] in edge_u:
                            c_ud.edges[e, 0] = c_u.edges[e, 0]
                        if c_d.edges[e, 0] in edge_d:
                            c_ud.edges[e, 0] = c_d.edges[e, 0]
                        if c_ud.edges[e, 0] == -1:
                            invalid = True  # edge collision
                            break
                    if not invalid:
                        for l in range(k.N_PERM_4):
                            c_d.set_d_edges(j * k.N_PERM_4 + l)
                            for e in edge_ud:
                                if c_u.edges[e, 0] in edge_u:
                                    c_ud.edges[e, 0] = c_u.edges[e, 0]
                                if c_d.edges[e, 0] in edge_d:
                                    c_ud.edges[e, 0] = c_d.edges[e, 0]
                            self.u_edges_plus_d_edges_to_ud_edges[
                                k.N_PERM_4 * i + l
                            ] = c_ud.get_ud_edges()
            np.save(fpath, self.u_edges_plus_d_edges_to_ud_edges)
        else:
            if self.show_progress:
                self.console.print(f"[bold green]Loading {fname} table...[/bold green]")
            self.u_edges_plus_d_edges_to_ud_edges = np.load(fpath, mmap_mode="r")


class CoordCube:
    """Represent a cube on the coordinate level.
    In phase 1 a state is uniquely determined by the three coordinates flip, twist and slice = slicesorted // 24.
    In phase 2 a state is uniquely determined by the three coordinates corners, ud_edges and slice_sorted % 24.
    """

    # Class level references to tables
    moves: Moves = None
    pruning: Pruning = None
    symmetries: Symmetries = None
    coord: Coord = None

    SOLVED = 0  # 0 is index of solved state (except for u_edges coordinate)

    def __init__(self, cc: CubieCube = None):
        """
        Create cube on coordinate level from Id-cube of from CubieCube

        Args:
        - cc: CubieCube The CubieCube
        """
        if cc is None:
            self.twist = self.SOLVED  # twist of corners
            self.flip = self.SOLVED  # flip of edges
            self.slice_sorted = (
                self.SOLVED
            )  # Position of FR, FL, BL, BR edges. Valid in phase 1 (<11880) and phase 2 (<24)
            # The phase 1 slice coordinate is given by slice_sorted // 24

            self.u_edges = 1656  # Valid in phase 1 (<11880) and phase 2 (<1680). 1656 is the index of solved u_edges.
            self.d_edges = (
                self.SOLVED
            )  # Valid in phase 1 (<11880) and phase 2 (<1680)
            self.corners = (
                self.SOLVED
            )  # corner permutation. Valid in phase1 and phase2
            self.ud_edges = (
                self.SOLVED
            )  # permutation of the ud-edges. Valid only in phase 2
        else:
            self.twist = cc.get_twist()
            self.flip = cc.get_flip()
            self.slice_sorted = cc.get_slice_sorted()
            self.u_edges = cc.get_u_edges()
            self.d_edges = cc.get_d_edges()
            self.corners = cc.get_corners()
            if self.slice_sorted < k.N_PERM_4:  # phase 2 cube
                self.ud_edges = cc.get_ud_edges()
            else:
                self.ud_edges = -1  # invalid

            # symmetry reduced flipslice coordinate used in phase 1
            self.flipslice_classidx = self.symmetries.flipslice_classidx[
                k.N_FLIP * (self.slice_sorted // k.N_PERM_4) + self.flip
            ]
            self.flipslice_sym = self.symmetries.flipslice_sym[
                k.N_FLIP * (self.slice_sorted // k.N_PERM_4) + self.flip
            ]
            self.flipslice_rep = self.symmetries.flipslice_rep[
                self.flipslice_classidx
            ]
            # symmetry reduced corner permutation coordinate used in phase 2
            self.corner_classidx = self.symmetries.corner_classidx[self.corners]
            self.corner_sym = self.symmetries.corner_sym[self.corners]
            self.corner_rep = self.symmetries.corner_rep[self.corner_classidx]

    def __str__(self):
        s = (
            "(twist: "
            + str(self.twist)
            + ", flip: "
            + str(self.flip)
            + ", slice: "
            + str(self.slice_sorted // 24)
            + ", U-edges: "
            + str(self.u_edges)
            + ", D-edges: "
            + str(self.d_edges)
            + ", E-edges: "
            + str(self.slice_sorted)
            + ", Corners: "
            + str(self.corners)
            + ", UD-Edges : "
            + str(self.ud_edges)
            + ")"
        )
        s = (
            s
            + "\n"
            + str(self.flipslice_classidx)
            + " "
            + str(self.flipslice_sym)
            + " "
            + str(self.flipslice_rep)
        )
        s = (
            s
            + "\n"
            + str(self.corner_classidx)
            + " "
            + str(self.corner_sym)
            + " "
            + str(self.corner_rep)
        )
        return s

    def phase1_move(self, m):
        """
        Update phase 1 coordinates when move is applied.
        :param m: The move
        """
        self.twist = self.moves.twist_move[k.N_MOVE * int(self.twist) + m]
        self.flip = self.moves.flip_move[k.N_MOVE * int(self.flip) + m]
        self.slice_sorted = self.moves.slice_sorted_move[
            k.N_MOVE * int(self.slice_sorted) + m
        ]
        # optional:
        self.u_edges = self.moves.u_edges_move[
            k.N_MOVE * int(self.u_edges) + m
        ]  # u_edges and d_edges retrieve ud_edges easily
        self.d_edges = self.moves.d_edges_move[
            k.N_MOVE * int(self.d_edges) + m
        ]  # if phase 1 is finished and phase 2 starts
        self.corners = self.moves.corners_move[
            k.N_MOVE * int(self.corners) + m
        ]  # Is needed only in phase 2

        self.flipslice_classidx = self.symmetries.flipslice_classidx[
            k.N_FLIP * (self.slice_sorted // k.N_PERM_4) + self.flip
        ]
        self.flipslice_sym = self.symmetries.flipslice_sym[
            k.N_FLIP * (self.slice_sorted // k.N_PERM_4) + self.flip
        ]
        self.flipslice_rep = self.symmetries.flipslice_rep[
            self.flipslice_classidx
        ]

        self.corner_classidx = self.symmetries.corner_classidx[self.corners]
        self.corner_sym = self.symmetries.corner_sym[self.corners]
        self.corner_rep = self.symmetries.corner_rep[self.corner_classidx]

    def phase2_move(self, m):
        """
        Update phase 2 coordinates when move is applied.
        :param m: The move
        """
        self.slice_sorted = self.moves.slice_sorted_move[
            k.N_MOVE * int(self.slice_sorted) + m
        ]
        self.corners = self.moves.corners_move[k.N_MOVE * int(self.corners) + m]
        self.ud_edges = self.moves.ud_edges_move[k.N_MOVE * int(self.ud_edges) + m]

    def get_depth_phase1(self):
        """
        Compute the distance to the cube subgroup H where flip=slice=twist=0
        :return: The distance to H
        """
        slice_ = self.slice_sorted // k.N_PERM_4
        flip = self.flip
        twist = self.twist
        flipslice = k.N_FLIP * slice_ + flip
        classidx = self.symmetries.flipslice_classidx[flipslice]
        sym = self.symmetries.flipslice_sym[flipslice]
        depth_mod3 = self.pruning.get_flipslice_twist_depth3(
            k.N_TWIST * int(classidx) + int(self.symmetries.twist_conj[twist, sym])
        )

        depth = 0
        while flip != self.SOLVED or slice_ != self.SOLVED or twist != self.SOLVED:
            if depth > 25: # TODO: check if this is required
                break
            if depth_mod3 == 0:
                depth_mod3 = 3
            for m in Move:
                twist1 = self.moves.twist_move[k.N_MOVE * int(twist) + m]
                flip1 = self.moves.flip_move[k.N_MOVE * int(flip) + m]
                slice1 = (
                    self.moves.slice_sorted_move[
                        k.N_MOVE * int(slice_) * k.N_PERM_4 + m
                    ]
                    // k.N_PERM_4
                )
                flipslice1 = k.N_FLIP * int(slice1) + int(flip1)
                classidx1 = self.symmetries.flipslice_classidx[flipslice1]
                sym = self.symmetries.flipslice_sym[flipslice1]
                if (
                    self.pruning.get_flipslice_twist_depth3(
                        k.N_TWIST * int(classidx1)
                        + int(self.symmetries.twist_conj[twist1, sym])
                    )
                    == depth_mod3 - 1
                ):
                    depth += 1
                    twist = twist1
                    flip = flip1
                    slice_ = slice1
                    depth_mod3 -= 1
                    break
            else:
                return 12 # TODO: check if this is really required
        return depth

    def get_depth_phase2(self, corners, ud_edges):
        """
        Get distance to subgroup where only the UD-slice edges may be permuted in their slice (only 24/2 = 12 possible
        ways due to overall even parity). This is a lower bound for the number of moves to solve phase 2.
        :param corners: Corners coordinate
        :param ud_edges: Coordinate of the 8 edges of U and D face.
        :return:
        """
        classidx = self.symmetries.corner_classidx[corners]
        sym = self.symmetries.corner_sym[corners]
        depth_mod3 = self.pruning.get_corners_ud_edges_depth3(
            k.N_UD_EDGES * int(classidx)
            + int(self.symmetries.ud_edges_conj[ud_edges, sym])
        )
        if depth_mod3 == 3:  # unfilled entry, depth >= 11
            return 11
        depth = 0
        while corners != self.SOLVED or ud_edges != self.SOLVED:
            if depth > 20:
                break
            if depth_mod3 == 0:
                depth_mod3 = 3
            # only iterate phase 2 moves
            for m in (
                Move.U1,
                Move.U2,
                Move.U3,
                Move.R2,
                Move.F2,
                Move.D1,
                Move.D2,
                Move.D3,
                Move.L2,
                Move.B2,
            ):
                corners1 = self.moves.corners_move[k.N_MOVE * int(corners) + m]
                ud_edges1 = self.moves.ud_edges_move[k.N_MOVE * int(ud_edges) + m]
                classidx1 = self.symmetries.corner_classidx[corners1]
                sym = self.symmetries.corner_sym[corners1]
                if (
                    self.pruning.get_corners_ud_edges_depth3(
                        k.N_UD_EDGES * int(classidx1)
                        + int(self.symmetries.ud_edges_conj[ud_edges1, sym])
                    )
                    == depth_mod3 - 1
                ):
                    depth += 1
                    corners = corners1
                    ud_edges = ud_edges1
                    depth_mod3 -= 1
                    break
            else:
                break
        return depth
