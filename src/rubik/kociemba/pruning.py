"""The pruning tables cut the search tree during the search.
The pruning values are stored modulo 3 which saves a lot of memory.
"""

import os
from os import path

import numpy as np
from rich.console import Console
from rich.progress import track

from .cubie import CubieCube, Symmetries, inv_idx, symCube
from .defs import Constants as k
from .moves import Moves


class Pruning:
    def __init__(self, folder: str = k.FOLDER, show_progress: bool = True, console: Console | None = None):
        self.folder = folder
        self.show_progress = show_progress
        self.console = console if console is not None else Console()

        if not os.path.exists(self.folder):
            os.mkdir(self.folder)

        self.flipslice_twist_depth3 = None
        self.corners_ud_edges_depth3 = None
        self.cornslice_depth = None
        self.distance = None

        # Dependencies
        self.moves = Moves(folder=self.folder, show_progress=False, console=self.console)
        self.symmetries = Symmetries(folder=self.folder, show_progress=False, console=self.console)

        # Load dependencies
        self.moves.create_tables()
        self.symmetries.create_tables()

        # Load symmetry tables needed for pruning
        self.flipslice_classidx, self.flipslice_sym, self.flipslice_rep = (
            self.symmetries.create_flipslice_tables()
        )
        self.corner_classidx, self.corner_sym, self.corner_rep = (
            self.symmetries.create_corner_tables()
        )
        self.twist_conj = self.symmetries.create_twist_conj_table()
        self.ud_edges_conj = self.symmetries.create_ud_edges_conj_table()

        self._init_distance()

    def _init_distance(self):
        """Array distance computes the new distance from the old_distance i and the new_distance_mod3 j.
        We need this array because the pruning tables only store the distances mod 3.
        """
        self.distance = np.zeros(60, dtype=np.int8)
        for i in range(20):
            for j in range(3):
                self.distance[3 * i + j] = (i // 3) * 3 + j
                if i % 3 == 2 and j == 0:
                    self.distance[3 * i + j] += 3
                elif i % 3 == 0 and j == 2:
                    self.distance[3 * i + j] -= 3

    def create_tables(self):
        self.create_phase1_prun_table()
        self.create_phase2_prun_table()
        self.create_phase2_cornsliceprun_table()

    # ####################### functions to extract or set values in the pruning tables #####################################

    def get_flipslice_twist_depth3(self, ix):
        """get_fst_depth3(ix) is *exactly* the number of moves % 3 to solve phase 1 of a cube with index ix"""
        y = self.flipslice_twist_depth3[ix // 16]
        y >>= (ix % 16) * 2
        return y & 3

    def get_corners_ud_edges_depth3(self, ix):
        """corners_ud_edges_depth3(ix) is *at least* the number of moves % 3 to solve phase 2 of a cube with index ix"""
        y = self.corners_ud_edges_depth3[ix // 16]
        y >>= (ix % 16) * 2
        return y & 3

    def set_flipslice_twist_depth3(self, ix, value):
        shift = (ix % 16) * 2
        base = ix >> 4

        self.flipslice_twist_depth3[base] &= ~(3 << shift) & 0xFFFFFFFF
        self.flipslice_twist_depth3[base] |= value << shift

    def set_corners_ud_edges_depth3(self, ix, value):
        shift = (ix % 16) * 2
        base = ix >> 4
        self.corners_ud_edges_depth3[base] &= ~(3 << shift) & 0xFFFFFFFF
        self.corners_ud_edges_depth3[base] |= value << shift

    ########################################################################################################################

    def create_phase1_prun_table(self):
        """Create/load the flipslice_twist_depth3 pruning table for phase 1."""
        total = k.N_FLIPSLICE_CLASS * k.N_TWIST
        fname = "phase1_prun.npy"
        fpath = path.join(self.folder, fname)

        if not path.isfile(fpath):
            # if self.show_progress:
                # self.console.print(f"[bold blue]Creating {fname} table...[/bold blue]")

            # #################### create table with the symmetries of the flipslice classes ###############################
            cc = CubieCube()
            fs_sym = np.zeros(k.N_FLIPSLICE_CLASS, dtype=np.uint16)

            iterator = range(k.N_FLIPSLICE_CLASS)
            if self.show_progress:
                iterator = track(
                    iterator,
                    description="Symmetries of flipslice classes".ljust(
                        k.PROGRESS_BAR_DESC_WIDTH
                    ),
                    console=self.console,
                )

            for i in iterator:
                rep = self.flipslice_rep[i]
                cc.set_slice(rep // k.N_FLIP)
                cc.set_flip(rep % k.N_FLIP)

                for s in range(k.N_SYM_D4h):
                    ss = CubieCube(
                        symCube[s].corners.copy(), symCube[s].edges.copy()
                    )  # copy cube
                    ss.edge_multiply(cc)  # s*cc
                    ss.edge_multiply(symCube[inv_idx[s]])  # s*cc*s^-1
                    if (
                        ss.get_slice() == rep // k.N_FLIP
                        and ss.get_flip() == rep % k.N_FLIP
                    ):
                        fs_sym[i] |= 1 << s
            # ##################################################################################################################

            # Use uint8 for faster generation (1 byte per entry)
            # Initialize with 3 (empty)
            prun_table = np.full(total, 3, dtype=np.uint8)

            # Set solved state (fs_classidx=0, twist=0) -> idx=0
            prun_table[0] = 0

            done = 1
            depth = 0

            # Pre-fetch tables for faster access
            twist_move = self.moves.twist_move
            flip_move = self.moves.flip_move
            slice_sorted_move = self.moves.slice_sorted_move
            flipslice_classidx = self.flipslice_classidx
            flipslice_sym = self.flipslice_sym
            twist_conj = self.twist_conj
            flipslice_rep = self.flipslice_rep

            while done < total:
                # if self.show_progress:
                    # self.console.print(f"Depth: {depth}, Done: {done}/{total}")

                depth3 = depth % 3
                next_depth = (depth + 1) % 3

                # Heuristic: Switch to backward search when the table is mostly full
                # or at a specific depth (depth 9 is used in the original implementation)
                backsearch = depth >= 9

                if backsearch:
                    # Backward search: Find unvisited nodes (3) and check if they have neighbors at current depth
                    # This is efficient when unvisited nodes are few
                    indices = np.flatnonzero(prun_table == 3)
                else:
                    # Forward search: Find nodes at current depth and expand to neighbors
                    indices = np.flatnonzero(prun_table == depth3)

                if indices.size == 0:
                    print(f"Depth {depth}: No more indices to expand. Done: {done}/{total}")
                    break

                # Process in batches to avoid memory overflow
                batch_size = 1000000
                num_batches = (indices.size + batch_size - 1) // batch_size

                iterator = range(num_batches)
                if self.show_progress:
                    iterator = track(
                        iterator,
                        description=f"Depth {depth} done: {done}/{total} [{'Back' if backsearch else 'Fwd'}]".ljust(
                            k.PROGRESS_BAR_DESC_WIDTH
                        ),
                        console=self.console,
                    )

                for i in iterator:
                    batch_indices = indices[i * batch_size : (i + 1) * batch_size]

                    # Calculate coordinates from indices
                    fs_classidx = batch_indices // k.N_TWIST
                    twist = batch_indices % k.N_TWIST

                    # Get representatives
                    rep = flipslice_rep[fs_classidx]
                    flip = rep % k.N_FLIP
                    slice_ = rep >> 11

                    # Iterate over all 18 moves
                    for m in range(k.N_MOVE):
                        # Vectorized move application
                        twist1 = twist_move[18 * twist + m]
                        flip1 = flip_move[18 * flip + m]
                        slice1 = slice_sorted_move[432 * slice_ + m] // 24

                        flipslice1 = (slice1.astype(np.int64) << 11) + flip1.astype(np.int64)

                        fs1_classidx = flipslice_classidx[flipslice1]
                        fs1_sym_val = flipslice_sym[flipslice1]

                        # Conjugate twist
                        twist1_conj = twist_conj[twist1, fs1_sym_val]

                        idx1 = k.N_TWIST * fs1_classidx.astype(
                            np.int64
                        ) + twist1_conj.astype(np.int64)

                        if backsearch:
                            # If neighbor is at current depth, then current node (batch_indices) is next_depth
                            # We check if idx1 (neighbor) has value depth3
                            mask = prun_table[idx1] == depth3
                            if np.any(mask):
                                # Set the original indices that reached a visited node
                                prun_table[batch_indices[mask]] = next_depth
                                # No symmetry handling needed for backward search (symmetries are also unvisited and will be processed)
                        else:
                            # Forward search
                            # Update table if neighbor is empty
                            mask = prun_table[idx1] == 3
                            if np.any(mask):
                                prun_table[idx1[mask]] = next_depth

                                # Handle symmetries
                                updated_fs1_classidx = fs1_classidx[mask]
                                updated_twist1 = twist1_conj[mask]

                                sym_class = fs_sym[updated_fs1_classidx]

                                mask_sym = sym_class != 1
                                if np.any(mask_sym):
                                    for sym_idx in range(1, 16):
                                        k_mask_all = (
                                            (sym_class >> sym_idx) & 1
                                        ).astype(bool)
                                        if np.any(k_mask_all):
                                            twist2 = twist_conj[
                                                updated_twist1[k_mask_all], sym_idx
                                            ]
                                            idx2 = k.N_TWIST * updated_fs1_classidx[
                                                k_mask_all
                                            ].astype(np.int64) + twist2.astype(np.int64)

                                            mask2 = prun_table[idx2] == 3
                                            prun_table[idx2[mask2]] = next_depth

                depth += 1
                done = np.count_nonzero(prun_table != 3)

            # Pack table
            # if self.show_progress:
                # self.console.print("[bold blue]Packing table...[/bold blue]")
            self.flipslice_twist_depth3 = np.zeros(total // 16 + 1, dtype=np.uint32)

            pad_len = (16 - (total % 16)) % 16
            padded = np.pad(prun_table, (0, pad_len), constant_values=3)
            reshaped = padded.reshape(-1, 16)

            for i in range(16):
                self.flipslice_twist_depth3[: reshaped.shape[0]] |= (
                    reshaped[:, i].astype(np.uint32) & 3
                ) << (2 * i)

            np.save(fpath, self.flipslice_twist_depth3)
        else:
            if self.show_progress:
                self.console.print(f"[bold green]Loading {fname} table...[/bold green]")
            self.flipslice_twist_depth3 = np.load(fpath, mmap_mode="r")
        return self.flipslice_twist_depth3

    def create_phase2_prun_table(self):
        """Create/load the corners_ud_edges_depth3 pruning table for phase 2."""
        total = k.N_CORNERS_CLASS * k.N_UD_EDGES
        fname = "phase2_prun.npy"
        fpath = path.join(self.folder, fname)

        if not path.isfile(fpath):
            # if self.show_progress:
                # self.console.print(f"[bold blue]Creating {fname} table...[/bold blue]")

            self.corners_ud_edges_depth3 = np.full(
                total // 16, 0xFFFFFFFF, dtype=np.uint32
            )
            # ##################### create table with the symmetries of the corners classes ################################
            cc = CubieCube()
            c_sym = np.zeros(k.N_CORNERS_CLASS, dtype=np.uint16)

            iterator = range(k.N_CORNERS_CLASS)
            if self.show_progress:
                iterator = track(
                    iterator,
                    description="Symmetries of corner classes".ljust(
                        k.PROGRESS_BAR_DESC_WIDTH
                    ),
                    console=self.console,
                )

            for i in iterator:
                rep = self.corner_rep[i]
                cc.set_corners(int(rep))
                for s in range(k.N_SYM_D4h):
                    ss = CubieCube(
                        symCube[s].corners.copy(), symCube[s].edges.copy()
                    )  # copy cube
                    ss.corner_multiply(cc)  # s*cc
                    ss.corner_multiply(symCube[inv_idx[s]])  # s*cc*s^-1
                    if ss.get_corners() == rep:
                        c_sym[i] |= 1 << s
            ################################################################################################################

            c_classidx = 0  # value for solved phase 2
            ud_edge = 0
            self.set_corners_ud_edges_depth3(k.N_UD_EDGES * c_classidx + ud_edge, 0)
            done = 1
            depth = 0

            self.console.print(
                "[bold yellow]Generating phase 2 pruning table. This may take some time...[/bold yellow]"
            )
            while depth < 10:  # we fill the table only do depth 9 + 1
                # if self.show_progress:
                    # self.console.print(f"Depth: {depth}, Done: {done}/{total}")

                depth3 = depth % 3
                idx = 0

                iterator = range(k.N_CORNERS_CLASS)
                if self.show_progress:
                    iterator = track(
                        iterator,
                        description=f"Depth {depth} done: {done}/{total}".ljust(k.PROGRESS_BAR_DESC_WIDTH),
                        console=self.console,
                    )

                for c_classidx in iterator:
                    ud_edge = 0
                    while ud_edge < k.N_UD_EDGES:
                        # ################ if table entries are not populated, this is very fast: ##########################
                        if (
                            idx % 16 == 0
                            and self.corners_ud_edges_depth3[idx // 16] == 0xFFFFFFFF
                            and ud_edge < k.N_UD_EDGES - 16
                        ):
                            ud_edge += 16
                            idx += 16
                            continue
                        ####################################################################################################

                        if self.get_corners_ud_edges_depth3(idx) == depth3:
                            corner = int(self.corner_rep[c_classidx])
                            # only iterate phase 2 moves
                            # U1, U2, U3, R2, F2, D1, D2, D3, L2, B2
                            # Color enum: U=0, R=1, F=2, D=3, L=4, B=5
                            # Moves: U(0), U2(1), U3(2), R2(4), F2(7), D(9), D2(10), D3(11), L2(13), B2(16)
                            # Wait, Color enum is just faces. BasicMoves has 6 faces.
                            # Moves are 18 total. 3 per face.
                            # U: 0, 1, 2
                            # R: 3, 4, 5
                            # F: 6, 7, 8
                            # D: 9, 10, 11
                            # L: 12, 13, 14
                            # B: 15, 16, 17

                            # Phase 2 moves:
                            # U, U2, U3 -> 0, 1, 2
                            # R2 -> 4
                            # F2 -> 7
                            # D, D2, D3 -> 9, 10, 11
                            # L2 -> 13
                            # B2 -> 16

                            phase2_moves = [0, 1, 2, 4, 7, 9, 10, 11, 13, 16]

                            for m in phase2_moves:
                                ud_edge1 = self.moves.ud_edges_move[18 * ud_edge + m]
                                corner1 = self.moves.corners_move[18 * corner + m]
                                c1_classidx = self.corner_classidx[corner1]
                                c1_sym = self.corner_sym[corner1]
                                ud_edge1 = self.ud_edges_conj[ud_edge1, c1_sym]
                                idx1 = k.N_UD_EDGES * int(c1_classidx) + int(
                                    ud_edge1
                                )  # N_UD_EDGES = 40320
                                if (
                                    self.get_corners_ud_edges_depth3(idx1) == 3
                                ):  # entry not yet filled
                                    self.set_corners_ud_edges_depth3(
                                        idx1, (depth + 1) % 3
                                    )  # depth + 1 <= 10
                                    done += 1
                                    # ######symmetric position has eventually more than one representation #############
                                    sym = c_sym[c1_classidx]
                                    if sym != 1:
                                        for sym_idx in range(1, 16):
                                            sym >>= 1
                                            if sym % 2 == 1:
                                                ud_edge2 = self.ud_edges_conj[
                                                    ud_edge1, sym_idx
                                                ]
                                                # c1_classidx does not change
                                                idx2 = k.N_UD_EDGES * int(
                                                    c1_classidx
                                                ) + int(ud_edge2)
                                                if (
                                                    self.get_corners_ud_edges_depth3(
                                                        idx2
                                                    )
                                                    == 3
                                                ):
                                                    self.set_corners_ud_edges_depth3(
                                                        idx2, (depth + 1) % 3
                                                    )
                                                    done += 1
                                    ####################################################################################

                        ud_edge += 1
                        idx += 1  # idx = N_UD_EDGEPERM * corner_classidx + ud_edge

                depth += 1

            np.save(fpath, self.corners_ud_edges_depth3)
        else:
            if self.show_progress:
                self.console.print(f"[bold green]Loading {fname} table...[/bold green]")
            self.corners_ud_edges_depth3 = np.load(fpath, mmap_mode="r")
        return self.corners_ud_edges_depth3

    def create_phase2_cornsliceprun_table(self):
        """Create/load the cornslice_depth pruning table for phase 2. With this table we do a fast precheck
        at the beginning of phase 2.
        """

        fname = "phase2_cornsliceprun.npy"
        fpath = path.join(self.folder, fname)

        if not path.isfile(fpath):
            # if self.show_progress:
                # self.console.print(f"[bold blue]Creating {fname} table...[/bold blue]")
            self.cornslice_depth = np.full(k.N_CORNERS * k.N_PERM_4, -1, dtype=np.int8)
            corners = 0  # values for solved phase 2
            slice_ = 0
            self.cornslice_depth[k.N_PERM_4 * corners + slice_] = 0
            done = 1
            depth = 0

            phase2_moves = [0, 1, 2, 4, 7, 9, 10, 11, 13, 16]

            iterator = range(13)
            if self.show_progress:
                iterator = track(
                    iterator,
                    description=f"Generating {fname}...".ljust(
                        k.PROGRESS_BAR_DESC_WIDTH
                    ),
                    console=self.console,
                )

            for depth in iterator:
                if done >= k.N_CORNERS * k.N_PERM_4:
                    break

                indices = np.flatnonzero(self.cornslice_depth == depth)
                if indices.size == 0:
                    break

                corners = indices // k.N_PERM_4
                slice_ = indices % k.N_PERM_4

                for m in phase2_moves:
                    corners1 = self.moves.corners_move[18 * corners + m]
                    slice_1 = self.moves.slice_sorted_move[18 * slice_ + m]
                    idx1 = k.N_PERM_4 * corners1.astype(np.int64) + slice_1.astype(
                        np.int64
                    )

                    mask = self.cornslice_depth[idx1] == -1
                    if np.any(mask):
                        self.cornslice_depth[idx1[mask]] = depth + 1
                        done += np.count_nonzero(mask)

            np.save(fpath, self.cornslice_depth)
        else:
            if self.show_progress:
                self.console.print(f"[bold green]Loading {fname} table...[/bold green]")
            self.cornslice_depth = np.load(fpath, mmap_mode="r")
        return self.cornslice_depth
