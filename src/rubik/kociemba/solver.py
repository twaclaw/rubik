"""The SolverThread class solves implements the two phase algorithm"""

import time
from dataclasses import dataclass

from rich.console import Console

from .coord import Coord, CoordCube
from .cubie import CubieCube, Move, Symmetries
from .defs import Constants as k
from .moves import Moves
from .pruning import Pruning


@dataclass
class SolverResult:
    ph1_str: list[str]
    ph2_str: list[str]
    ph1_htm: list[str]
    ph2_htm: list[str]
    execution_time: float


class Solver:
    def __init__(self, folder: str = k.FOLDER, show_progress: bool = True):
        self.folder = folder
        self.show_progress = show_progress
        self.console = Console()
        self.solution_ph1: str | None = None
        self.solution_ph2: str | None = None

        # Initialize tables
        self.symmetries = Symmetries(folder=folder, show_progress=show_progress)
        self.moves = Moves(folder=folder, show_progress=show_progress)
        self.pruning = Pruning(folder=folder, show_progress=show_progress)
        self.coord = Coord(folder=folder, show_progress=show_progress)

        # Load tables
        self.symmetries.create_tables()
        self.moves.create_tables()
        self.pruning.create_tables()
        self.coord.create_tables()

        # Inject dependencies into CoordCube
        CoordCube.moves = self.moves
        CoordCube.pruning = self.pruning
        CoordCube.symmetries = self.symmetries
        CoordCube.coord = self.coord

        # Internal cube
        self.cb_cube = CubieCube()

    def search_phase2(self, corners, ud_edges, slice_sorted, dist, togo_phase2):
        if self.phase2_done:
            return

        if togo_phase2 == 0 and slice_sorted == 0:
            self.solution_ph1 = self.sofar_phase1.copy()
            self.solution_ph2 = self.sofar_phase2.copy()
            self.phase2_done = True
        else:
            for m in Move:
                if m in [
                    Move.R1,
                    Move.R3,
                    Move.F1,
                    Move.F3,
                    Move.L1,
                    Move.L3,
                    Move.B1,
                    Move.B3,
                ]:
                    continue

                if len(self.sofar_phase2) > 0:
                    diff = self.sofar_phase2[-1] // 3 - m // 3
                    if (
                        diff in [0, 3]
                    ):  # successive moves: on same face or on same axis with wrong order
                        continue
                else:
                    if len(self.sofar_phase1) > 0:
                        diff = self.sofar_phase1[-1] // 3 - m // 3
                        if (
                            diff in [0, 3]
                        ):  # successive moves: on same face or on same axis with wrong order
                            continue

                corners_new = self.moves.corners_move[18 * int(corners) + m]
                ud_edges_new = self.moves.ud_edges_move[18 * int(ud_edges) + m]
                slice_sorted_new = self.moves.slice_sorted_move[
                    18 * int(slice_sorted) + m
                ]

                classidx = self.symmetries.corner_classidx[corners_new]
                sym = self.symmetries.corner_sym[corners_new]
                dist_new_mod3 = self.pruning.get_corners_ud_edges_depth3(
                    40320 * int(classidx)
                    + int(self.symmetries.ud_edges_conj[ud_edges_new, sym])
                )
                dist_new = self.pruning.distance[3 * dist + dist_new_mod3]
                if (
                    max(
                        dist_new,
                        self.pruning.cornslice_depth[
                            24 * int(corners_new) + int(slice_sorted_new)
                        ],
                    )
                    >= togo_phase2
                ):
                    continue  # impossible to reach solved cube in togo_phase2 - 1 moves

                self.sofar_phase2.append(m)
                self.search_phase2(
                    corners_new,
                    ud_edges_new,
                    slice_sorted_new,
                    dist_new,
                    togo_phase2 - 1,
                )
                self.sofar_phase2.pop(-1)

    def search(self, flip, twist, slice_sorted, dist, togo_phase1):
        if togo_phase1 == 0:  # phase 1 solved
            if flip != 0 or twist != 0 or slice_sorted // 24 != 0:
                return

            # compute initial phase 2 coordinates
            if self.sofar_phase1:  # check if list is not empty
                m = self.sofar_phase1[-1]
            else:
                m = Move.U1  # value is irrelevant here, no phase 1 moves

            if m in [
                Move.R3,
                Move.F3,
                Move.L3,
                Move.B3,
            ]:  # phase 1 solution come in pairs
                corners = self.moves.corners_move[
                    18 * int(self.cornersave) + m - 1
                ]  # apply R2, F2, L2 ord B2 on last ph1 solution
            else:
                corners = self.co_cube.corners
                for m in self.sofar_phase1:  # get current corner configuration
                    corners = self.moves.corners_move[18 * int(corners) + m]
                self.cornersave = corners

            # new solution must be shorter and we do not use phase 2 maneuvers with length > 11 - 1 = 10
            togo2_limit = (
                11  # min(self.shortest_length[0] - len(self.sofar_phase1), 11)
            )
            if (
                self.pruning.cornslice_depth[24 * int(corners) + int(slice_sorted)]
                >= togo2_limit
            ):  # precheck speeds up the computation
                return

            u_edges = self.co_cube.u_edges
            d_edges = self.co_cube.d_edges
            for m in self.sofar_phase1:
                u_edges = self.moves.u_edges_move[18 * int(u_edges) + m]
                d_edges = self.moves.d_edges_move[18 * int(d_edges) + m]
            ud_edges = self.coord.u_edges_plus_d_edges_to_ud_edges[
                24 * int(u_edges) + int(d_edges) % 24
            ]

            dist2 = self.co_cube.get_depth_phase2(corners, ud_edges)
            for togo2 in range(
                dist2, togo2_limit
            ):  # do not use more than togo2_limit - 1 moves in phase 2
                self.sofar_phase2 = []
                self.phase2_done = False
                self.search_phase2(corners, ud_edges, slice_sorted, dist2, togo2)
                if self.phase2_done:  # solution already found
                    break

        else:
            for m in Move:
                # dist = 0 means that we are already are in the subgroup H. If there are less than 5 moves left
                # this forces all remaining moves to be phase 2 moves. So we can forbid these at the end of phase 1
                # and generate these moves in phase 2.
                if (
                    dist == 0
                    and togo_phase1 < 5
                    and m
                    in [
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
                    ]
                ):
                    continue

                if len(self.sofar_phase1) > 0:
                    diff = self.sofar_phase1[-1] // 3 - m // 3
                    if (
                        diff in [0, 3]
                    ):  # successive moves: on same face or on same axis with wrong order
                        continue

                flip_new = self.moves.flip_move[18 * int(flip) + m]  # N_MOVE = 18
                twist_new = self.moves.twist_move[18 * int(twist) + m]
                slice_sorted_new = self.moves.slice_sorted_move[
                    18 * int(slice_sorted) + m
                ]

                flipslice = 2048 * int(slice_sorted_new // 24) + int(
                    flip_new
                )  # N_FLIP * (slice_sorted // N_PERM_4) + flip
                classidx = self.symmetries.flipslice_classidx[flipslice]
                sym = self.symmetries.flipslice_sym[flipslice]
                dist_new_mod3 = self.pruning.get_flipslice_twist_depth3(
                    2187 * int(classidx)
                    + int(self.symmetries.twist_conj[twist_new, sym])
                )
                dist_new = self.pruning.distance[3 * dist + dist_new_mod3]
                if (
                    dist_new >= togo_phase1
                ):  # impossible to reach subgroup H in togo_phase1 - 1 moves
                    continue

                self.sofar_phase1.append(m)
                self.search(
                    flip_new, twist_new, slice_sorted_new, dist_new, togo_phase1 - 1
                )
                if self.phase2_done:
                    return
                self.sofar_phase1.pop(-1)

    def solve(self, cube_string: str) -> SolverResult:
        """Solve the cube given in cube_string using the two phase algorithm"""
        t0 = time.perf_counter()
        self.cb_cube.from_string(cube_string)
        self.phase2_done = False
        self.co_cube = CoordCube(
            self.cb_cube
        )  # the rotated/inverted cube in coordinate representation

        dist = self.co_cube.get_depth_phase1()
        for togo1 in range(
            dist, 20
        ):  # iterative deepening, solution has at least dist moves
            self.sofar_phase1 = []
            self.search(
                self.co_cube.flip,
                self.co_cube.twist,
                self.co_cube.slice_sorted,
                dist,
                togo1,
            )
            if self.phase2_done:
                break
        t = time.perf_counter() - t0

        ph1_htm = [Move.to_htm(m) for m in self.solution_ph1]
        ph2_htm = [Move.to_htm(m) for m in self.solution_ph2]
        ph1_str = [Move.to_manim_str(m) for m in self.solution_ph1]
        ph2_str = [Move.to_manim_str(m) for m in self.solution_ph2]
        return SolverResult(
            ph1_str=ph1_str,
            ph2_str=ph2_str,
            ph1_htm=ph1_htm,
            ph2_htm=ph2_htm,
            execution_time=t,
        )
