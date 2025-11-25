"""
                ┌────┬────┬────┐
                │ U1 │ U2 │ U3 │
                ├────┼────┼────┤
                │ U4 │ U5 │ U6 │
                ├────┼────┼────┤
                │ U7 │ U8 │ U9 │
                └────┴────┴────┘
┌────┬────┬────┐┌────┬────┬────┐┌────┬────┬────┐┌────┬────┬────┐
│ L1 │ L2 │ L3 ││ F1 │ F2 │ F3 ││ R1 │ R2 │ R3 ││ B1 │ B2 │ B3 │
├────┼────┼────┤├────┼────┼────┤├────┼────┼────┤├────┼────┼────┤
│ L4 │ L5 │ L6 ││ F4 │ F5 │ F6 ││ R4 │ R5 │ R6 ││ B4 │ B5 │ B6 │
├────┼────┼────┤├────┼────┼────┤├────┼────┼────┤├────┼────┼────┤
│ L7 │ L8 │ L9 ││ F7 │ F8 │ F9 ││ R7 │ R8 │ R9 ││ B7 │ B8 │ B9 │
└────┴────┴────┘└────┴────┴────┘└────┴────┴────┘└────┴────┴────┘
                ┌────┬────┬────┐
                │ D1 │ D2 │ D3 │
                ├────┼────┼────┤
                │ D4 │ D5 │ D6 │
                ├────┼────┼────┤
                │ D7 │ D8 │ D9 │
                └────┴────┴────┘
"""

import math
import os
from enum import IntEnum
from os import path

import numpy as np
from lehmer import Lehmer
from rich.console import Console
from rich.progress import track

from rubik.cube import Cube

from .defs import Constants as k


class Color(IntEnum):
    U = 0  # Up
    R = 1  # Right
    F = 2  # Front
    D = 3  # Down
    L = 4  # Left
    B = 5  # Back


# Map the corner positions to facelet colors.
cornerColor = [
    [Color.U, Color.R, Color.F],
    [Color.U, Color.F, Color.L],
    [Color.U, Color.L, Color.B],
    [Color.U, Color.B, Color.R],
    [Color.D, Color.F, Color.R],
    [Color.D, Color.L, Color.F],
    [Color.D, Color.B, Color.L],
    [Color.D, Color.R, Color.B],
]

# Map the edge positions to facelet colors.
edgeColor = [
    [Color.U, Color.R],
    [Color.U, Color.F],
    [Color.U, Color.L],
    [Color.U, Color.B],
    [Color.D, Color.R],
    [Color.D, Color.F],
    [Color.D, Color.L],
    [Color.D, Color.B],
    [Color.F, Color.R],
    [Color.F, Color.L],
    [Color.B, Color.L],
    [Color.B, Color.R],
]


class Corner(IntEnum):
    """The names of the corner positions of the cube. Corner URF e.g. has an U(p), a R(ight) and a F(ront) facelet."""

    URF = 0
    UFL = 1
    ULB = 2
    UBR = 3
    DFR = 4
    DLF = 5
    DBL = 6
    DRB = 7


class Edge(IntEnum):
    """The names of the edge positions of the cube. Edge UR e.g. has an U(p) and R(ight) facelet."""

    UR = 0
    UF = 1
    UL = 2
    UB = 3
    DR = 4
    DF = 5
    DL = 6
    DB = 7
    FR = 8
    FL = 9
    BL = 10
    BR = 11


class CubieCube:
    def __init__(
        self, corners: np.ndarray | None = None, edges: np.ndarray | None = None
    ):
        """
        Args:
        - corners: shape (8, 2) -- first column: which corner, second column: orientation
        - edges: shape (12, 2) -- first column: which edge, second column: orientation
        - Orientations can be:
         - 0 -> correctly oriented
         - 1 -> rotated clockwise (for corners) or flipped (for edges)
         - 2 -> rotated counter-clockwise (for corners) or flipped (for edges)
        """
        self.num_corners = 8
        self.num_edges = 12
        self._comb_vectorized = np.vectorize(self.c_nk)

        # used for slice coordinate calculations
        self._slice_edge = np.array([Edge.FR, Edge.FL, Edge.BL, Edge.BR])
        self._other_edge = np.array(
            [Edge.UR, Edge.UF, Edge.UL, Edge.UB, Edge.DR, Edge.DF, Edge.DL, Edge.DB]
        )

        if corners is None:
            self.corners = np.array([[Corner(i), 0] for i in range(8)], dtype=int)
        else:
            self.corners = corners.copy()

        if edges is None:
            self.edges = np.array([[Edge(i), 0] for i in range(12)], dtype=int)
        else:
            self.edges = edges

        self.lc8 = Lehmer(n=8, squeeze=True, dtype=np.uint16)  # for corners
        self.lc4 = Lehmer(n=4, squeeze=True, dtype=np.uint8)  # for UD edges

    @staticmethod
    def c_nk(n: int, k: int) -> int:
        """Compute the binomial coefficient C(n, k)"""
        return math.comb(n, k)

    def __eq__(self, other: object) -> bool:
        return np.array_equal(self.corners, other.corners) and np.array_equal(
            self.edges, other.edges
        )

    def copy(self) -> "CubieCube":
        """Create a copy of this CubieCube."""
        return CubieCube(corners=self.corners.copy(), edges=self.edges.copy())

    def corner_multiply(self, other: "CubieCube") -> None:
        """Multiply this cubie cube with another cubie cube, restricted to the corners."""
        # Extract permutations and orientations
        cp_a = self.corners[:, 0]  # corner permutation
        co_a = self.corners[:, 1]  # corner orientation
        cp_b = other.corners[:, 0]
        ori_b = other.corners[:, 1]

        c_perm = cp_a[cp_b]
        ori_a = co_a[cp_b]

        # Initialize result orientation array
        c_ori = np.zeros(8, dtype=np.uint8)

        # Case 1: both regular cubes (ori_a < 3 and ori_b < 3)
        ori = ori_a + ori_b
        mask1 = (ori_a < 3) & (ori_b < 3)
        ori = np.where(ori >= 3, ori - 3, ori)
        c_ori = np.where(mask1, ori, c_ori)

        # Case 2: cube b is in a mirrored state (ori_a < 3 and ori_b >= 3)
        mask2 = (ori_a < 3) & (ori_b >= 3)
        ori = ori_a + ori_b
        ori = np.where(ori >= 6, ori - 3, ori)
        c_ori = np.where(mask2, ori, c_ori)

        # Case 3: cube a is in a mirrored state (ori_a >= 3 and ori_b < 3)
        mask3 = (ori_a >= 3) & (ori_b < 3)
        ori = ori_a - ori_b
        ori = np.where(ori < 3, ori + 3, ori)
        c_ori = np.where(mask3, ori, c_ori)

        # Case 4: both cubes are in mirrored states (ori_a >= 3 and ori_b >= 3)
        mask4 = (ori_a >= 3) & (ori_b >= 3)
        ori = ori_a - ori_b
        ori = np.where(ori < 0, ori + 3, ori)
        c_ori = np.where(mask4, ori, c_ori)

        # Update self.corners
        self.corners[:, 0] = c_perm
        self.corners[:, 1] = c_ori

    def edge_multiply(self, other: "CubieCube") -> None:
        """Multiply this cubie cube with another cubie cube b, restricted to the edges. Does not change b."""
        ep_a = self.edges[:, 0]  # permutation
        eo_a = self.edges[:, 1]  # orientation
        ep_b = other.edges[:, 0]
        eo_b = other.edges[:, 1]

        e_perm = ep_a[ep_b]

        # new orientation
        e_ori = np.mod((eo_b + eo_a[ep_b]), 2)

        self.edges[:, 0] = e_perm
        self.edges[:, 1] = e_ori

    def multiply(self, other: "CubieCube") -> None:
        """Multiply this cubie cube with another cubie cube b. Does not change b."""
        self.corner_multiply(other)
        self.edge_multiply(other)

    def inverse(self) -> "CubieCube":
        inv = CubieCube()
        inv.edges[self.edges[:, 0], 0] = np.arange(self.num_edges, dtype=int)
        inv.edges[:, 1] = self.edges[:, 1][inv.edges[:, 0]]

        inv.corners[self.corners[:, 0], 0] = np.arange(self.num_corners, dtype=int)

        ori = self.corners[:, 1][inv.corners[:, 0]]
        mask = ori >= 3
        inv.corners[:, 1] = np.where(mask, ori, -ori)
        inv.corners[:, 1] = np.where(
            inv.corners[:, 1] < 0, inv.corners[:, 1] + 3, inv.corners[:, 1]
        )

        return inv

    def symmetries(self) -> list[int]:
        """Generate a list of the symmetries and antisymmetries of the cubie cube."""
        s = []
        for j in range(k.N_SYM):
            c = CubieCube(symCube[j].corners.copy(), symCube[j].edges.copy())
            c.multiply(self)
            c.multiply(symCube[inv_idx[j]])
            if self == c:
                s.append(j)

            d = c.inverse()
            if self == d:  # then we have antisymmetry
                s.append(j + k.N_SYM)
        return s

    # -------- Beginning of coordinate functions --------

    def get_twist(self) -> int:
        """Compute the twist coordinate (corner orientations) of this CubieCube."""
        return np.polyval(self.corners[:-1, 1], 3).astype(int)

    def set_twist(self, twist: int) -> None:
        """Set the corner orientations of this CubieCube from the given twist coordinate."""
        powers = 3 ** np.arange(self.num_corners - 1)
        div = twist // powers
        new_corners_1 = np.flip(np.mod(div, 3))
        twistparity = np.sum(new_corners_1) % 3
        self.corners[:, 1] = np.concatenate([new_corners_1, [(3 - twistparity) % 3]])

    def get_flip(self) -> int:
        """Compute the flip coordinate (edge orientations) of this CubieCube."""
        return np.polyval(self.edges[:-1, 1], 2).astype(int)

    def set_flip(self, flip: int) -> None:
        """Set the edge orientations of this CubieCube from the given flip coordinate."""
        powers = 2 ** np.arange(self.num_edges - 1)
        div = flip // powers
        new_edges_1 = np.flip(np.mod(div, 2))
        flipparity = np.sum(new_edges_1) % 2
        self.edges[:, 1] = np.concatenate([new_edges_1, [(2 - flipparity) % 2]])

    def get_slice(self) -> int:
        """Get the location of the UD-slice edges FR,FL,BL and BR ignoring their permutation.
        0<= slice < 495 in phase 1, slice = 0 in phase 2."""
        mask = (self.edges[:, 0] >= Edge.FR) & (self.edges[:, 0] <= Edge.BR)
        j = np.where(mask)[0][::-1]  # flipped order
        x = np.arange(len(j))
        return np.sum(self._comb_vectorized(11 - j, x + 1))

    def set_slice(self, idx: int):
        idx = int(idx)
        ep = np.full(12, -1, dtype=np.int8)  # invalidate all edge positions
        x = 4
        # TODO: check if this can be optimized
        for j in np.arange(self.num_edges):
            if x == 0:
                break
            comb = self.c_nk(11 - j, x)
            if idx - comb >= 0:
                ep[j] = self._slice_edge[4 - x]
                idx -= comb
                x -= 1

        mask = ep == -1
        ep[mask] = self._other_edge[: np.sum(mask)]
        self.edges[:, 0] = ep

    def get_slice_sorted(self) -> int:
        """Get the permutation and location of the UD-slice edges FR,FL,BL and BR.
        0 <= slice_sorted < 11880 in phase 1, 0 <= slice_sorted < 24 in phase 2, slice_sorted = 0 for solved cube."""
        mask = (self.edges[:, 0] >= Edge.FR) & (self.edges[:, 0] <= Edge.BR)
        indices = np.where(mask)[0]
        j_indices = indices[::-1]

        x = np.arange(len(j_indices))
        a = np.sum(self._comb_vectorized(11 - j_indices, x + 1))

        edge4 = self.edges[indices, 0]
        b = self.lc4.encode(edge4, minvalue=Edge.FR)
        return 24 * a + b

    def set_slice_sorted(self, idx: int):
        other_edge = np.array(
            [Edge.UR, Edge.UF, Edge.UL, Edge.UB, Edge.DR, Edge.DF, Edge.DL, Edge.DB]
        )
        b = idx % 24
        a = idx // 24

        ep = np.full(12, -1, dtype=int)

        slice_edge = self.lc4.decode(b, minvalue=Edge.FR)

        x = 4
        for j in range(12):
            if x == 0:
                break
            comb = self.c_nk(11 - j, x)
            if a >= comb:
                ep[j] = slice_edge[4 - x]
                a -= comb
                x -= 1

        mask = ep == -1
        ep[mask] = other_edge
        self.edges[:, 0] = ep

    def get_u_edges(self) -> int:
        """Get the permutation and location of edges UR, UF, UL and UB.
        0 <= u_edges < 11880 in phase 1, 0 <= u_edges < 1680 in phase 2, u_edges = 1656 for solved cube."""
        ep_mod = np.roll(self.edges[:, 0], 4)

        mask = (ep_mod >= Edge.UR) & (ep_mod <= Edge.UB)
        indices = np.where(mask)[0]
        j_indices = indices[::-1]

        x = np.arange(len(j_indices))
        a = np.sum(self._comb_vectorized(11 - j_indices, x + 1))

        edge4 = ep_mod[indices]
        b = self.lc4.encode(edge4, minvalue=Edge.UR)
        return 24 * a + b

    def set_u_edges(self, idx: int):
        other_edge = np.array(
            [Edge.DR, Edge.DF, Edge.DL, Edge.DB, Edge.FR, Edge.FL, Edge.BL, Edge.BR]
        )
        b = idx % 24
        a = idx // 24

        ep = np.full(12, -1, dtype=int)
        slice_edge = self.lc4.decode(b, minvalue=Edge.UR)

        x = 4
        for j in range(12):
            if x == 0:
                break
            comb = self.c_nk(11 - j, x)
            if a >= comb:
                ep[j] = slice_edge[4 - x]
                a -= comb
                x -= 1

        mask = ep == -1
        ep[mask] = other_edge
        self.edges[:, 0] = np.roll(ep, -4)

    def get_d_edges(self) -> int:
        """Get the permutation and location of the edges DR, DF, DL and DB.
        0 <= d_edges < 11880 in phase 1, 0 <= d_edges < 1680 in phase 2, d_edges = 0 for solved cube."""
        ep_mod = np.roll(self.edges[:, 0], 4)

        mask = (ep_mod >= Edge.DR) & (ep_mod <= Edge.DB)
        indices = np.where(mask)[0]
        j_indices = indices[::-1]

        x = np.arange(len(j_indices))
        a = np.sum(self._comb_vectorized(11 - j_indices, x + 1))

        edge4 = ep_mod[indices]
        b = self.lc4.encode(edge4, minvalue=Edge.DR)
        return 24 * a + b

    def set_d_edges(self, idx: int):
        other_edge = np.array(
            [Edge.FR, Edge.FL, Edge.BL, Edge.BR, Edge.UR, Edge.UF, Edge.UL, Edge.UB]
        )
        b = idx % 24
        a = idx // 24

        ep = np.full(12, -1, dtype=int)
        slice_edge = self.lc4.decode(b, minvalue=Edge.DR)

        x = 4
        for j in range(12):
            if x == 0:
                break
            comb = self.c_nk(11 - j, x)
            if a >= comb:
                ep[j] = slice_edge[4 - x]
                a -= comb
                x -= 1

        mask = ep == -1
        ep[mask] = other_edge
        self.edges[:, 0] = np.roll(ep, -4)

    def get_corners(self) -> int:
        """Get the permutation of the 8 corners.
        0 <= corners < 40320 defined but unused in phase 1, 0 <= corners < 40320 in phase 2,
        corners = 0 for solved cube"""
        return self.lc8.encode(self.corners[:, 0], minvalue=0)

    def set_corners(self, idx: int):
        self.corners[:, 0] = self.lc8.decode(idx, minvalue=0)

    def get_ud_edges(self) -> int:
        """Get the permutation of the 8 U and D edges.
        ud_edges undefined in phase 1, 0 <= ud_edges < 40320 in phase 2, ud_edges = 0 for solved cube."""
        perm = self.edges[0:8, 0].copy()
        return self.lc8.encode(perm, minvalue=0)

    def set_ud_edges(self, idx: int):
        # positions of FR FL BL BR edges are not affected
        self.edges[0:8, 0] = self.lc8.decode(idx)

    # --- End of coordinate functions ---

    def from_cube(self, cube: Cube):
        """Update this CubieCube from a Facelet Cube.

        Args:
        - cube: a cube in Cube (Facelet) representation.
          Cube is the Facelet representation I used for illustrating other algorithms: e.g., BFS.
          Cube is also the representation used to graphically display the cube on the terminal.
        """
        # Corner facelet positions
        corner_facelets = [
            [(0, 2, 2), (1, 0, 0), (2, 0, 2)],  # Corner URF: U-R-F
            [(0, 2, 0), (2, 0, 0), (4, 0, 2)],  # Corner UFL: U-F-L
            [(0, 0, 0), (4, 0, 0), (5, 0, 2)],  # Corner ULB: U-L-B
            [(0, 0, 2), (5, 0, 0), (1, 0, 2)],  # Corner UBR: U-B-R
            [(3, 0, 2), (2, 2, 2), (1, 2, 0)],  # Corner DFR: D-F-R
            [(3, 0, 0), (4, 2, 2), (2, 2, 0)],  # Corner DLF: D-L-F
            [(3, 2, 0), (5, 2, 2), (4, 2, 0)],  # Corner DBL: D-B-L
            [(3, 2, 2), (1, 2, 2), (5, 2, 0)],  # Corner DRB: D-R-B
        ]

        # Edge facelet positions
        edge_facelets = [
            [(0, 1, 2), (1, 0, 1)],  # Edge UR: U-R
            [(0, 2, 1), (2, 0, 1)],  # Edge UF: U-F
            [(0, 1, 0), (4, 0, 1)],  # Edge UL: U-L
            [(0, 0, 1), (5, 0, 1)],  # Edge UB: U-B
            [(3, 1, 2), (1, 2, 1)],  # Edge DR: D-R
            [(3, 0, 1), (2, 2, 1)],  # Edge DF: D-F
            [(3, 1, 0), (4, 2, 1)],  # Edge DL: D-L
            [(3, 2, 1), (5, 2, 1)],  # Edge DB: D-B
            [(2, 1, 2), (1, 1, 0)],  # Edge FR: F-R
            [(2, 1, 0), (4, 1, 2)],  # Edge FL: F-L
            [(5, 1, 2), (4, 1, 0)],  # Edge BL: B-L
            [(5, 1, 0), (1, 1, 2)],  # Edge BR: B-R
        ]

        # Map sets of colors to piece indices
        corner_colors_map = {}
        for i in range(8):
            # corner_facelets[i] contains (face, row, col).
            # The color of that facelet in a solved cube is simply `face`.
            colors = tuple(sorted([f[0] for f in corner_facelets[i]]))
            corner_colors_map[colors] = i

        edge_colors_map = {}
        for i in range(12):
            colors = tuple(sorted([f[0] for f in edge_facelets[i]]))
            edge_colors_map[colors] = i

        # Process Corners
        for i in range(8):
            # Read colors from the cube at the positions for corner i
            obs_colors = []
            for f, r, c in corner_facelets[i]:
                obs_colors.append(cube.faces[f, r, c])

            # Identify the piece
            piece_idx = corner_colors_map.get(tuple(sorted(obs_colors)))
            if piece_idx is None:
                raise ValueError(
                    f"Invalid corner colors at position {Corner(i).name}: {obs_colors}"
                )

            self.corners[i, 0] = piece_idx

            # Identify orientation
            # The primary color of the piece is the one on the U(0) or D(3) face in the solved state.
            # For corner `piece_idx`, the primary color is `corner_facelets[piece_idx][0][0]`.
            primary_color = corner_facelets[piece_idx][0][0]

            # Find where this primary color is in the observed colors
            # obs_colors are [U/D, R/L/F/B, F/B/L/R] relative to the corner position

            if obs_colors[0] == primary_color:
                self.corners[i, 1] = 0
            elif obs_colors[1] == primary_color:
                self.corners[i, 1] = 1
            elif obs_colors[2] == primary_color:
                self.corners[i, 1] = 2
            else:
                found = False
                for idx, color in enumerate(obs_colors):
                    if color == 0 or color == 3:
                        if idx == 0:
                            self.corners[i, 1] = 0
                        elif idx == 1:
                            self.corners[i, 1] = 2
                        elif idx == 2:
                            self.corners[i, 1] = 1
                        found = True
                        break

                if not found:
                    raise ValueError(
                        f"Primary color {primary_color} not found in observed colors {obs_colors}"
                    )

        # Process Edges
        for i in range(12):
            obs_colors = []
            for f, r, c in edge_facelets[i]:
                obs_colors.append(cube.faces[f, r, c])

            piece_idx = edge_colors_map.get(tuple(sorted(obs_colors)))
            if piece_idx is None:
                raise ValueError(
                    f"Invalid edge colors at position {Edge(i).name}: {obs_colors}"
                )

            self.edges[i, 0] = piece_idx

            # Identify orientation
            # Primary color is `edge_facelets[piece_idx][0][0]`
            primary_color = edge_facelets[piece_idx][0][0]

            if obs_colors[0] == primary_color:
                self.edges[i, 1] = 0
            elif obs_colors[1] == primary_color:
                self.edges[i, 1] = 1
            else:
                raise ValueError(
                    f"Primary color {primary_color} not found in observed colors {obs_colors}"
                )

    def to_cube(self) -> Cube:
        """Convert this CubieCube representation to a Facelet Cube representation.

        Returns:
        - cube: a cube in Cube (Facelet) representation.
          Cube is the Facelet representation I used for illustrating other algorithms: e.g., BFS.
          Cube is also the representation used to graphically display the cube on the terminal.
        """

        # Initialize faces array
        faces = np.zeros((6, 3, 3), dtype=np.uint8)

        # Corner facelet positions for each corner in solved state
        # Each corner has 3 facelets, listed in clockwise order when viewed from outside
        # Face order: U=0, R=1, F=2, D=3, L=4, B=5
        corner_facelets = [
            [(0, 2, 2), (1, 0, 0), (2, 0, 2)],  # Corner URF: U-R-F
            [(0, 2, 0), (2, 0, 0), (4, 0, 2)],  # Corner UFL: U-F-L
            [(0, 0, 0), (4, 0, 0), (5, 0, 2)],  # Corner ULB: U-L-B
            [(0, 0, 2), (5, 0, 0), (1, 0, 2)],  # Corner UBR: U-B-R
            [(3, 0, 2), (2, 2, 2), (1, 2, 0)],  # Corner DFR: D-F-R
            [(3, 0, 0), (4, 2, 2), (2, 2, 0)],  # Corner DLF: D-L-F
            [(3, 2, 0), (5, 2, 2), (4, 2, 0)],  # Corner DBL: D-B-L
            [(3, 2, 2), (1, 2, 2), (5, 2, 0)],  # Corner DRB: D-R-B
        ]

        # Each edge has 2 facelets
        edge_facelets = [
            [(0, 1, 2), (1, 0, 1)],  # Edge UR: U-R
            [(0, 2, 1), (2, 0, 1)],  # Edge UF: U-F
            [(0, 1, 0), (4, 0, 1)],  # Edge UL: U-L
            [(0, 0, 1), (5, 0, 1)],  # Edge UB: U-B
            [(3, 1, 2), (1, 2, 1)],  # Edge DR: D-R
            [(3, 0, 1), (2, 2, 1)],  # Edge DF: D-F
            [(3, 1, 0), (4, 2, 1)],  # Edge DL: D-L
            [(3, 2, 1), (5, 2, 1)],  # Edge DB: D-B
            [(2, 1, 2), (1, 1, 0)],  # Edge FR: F-R
            [(2, 1, 0), (4, 1, 2)],  # Edge FL: F-L
            [(5, 1, 2), (4, 1, 0)],  # Edge BL: B-L
            [(5, 1, 0), (1, 1, 2)],  # Edge BR: B-R
        ]

        # Set center facelets (they don't move)
        for face in range(6):
            faces[face, 1, 1] = face

        for i in range(self.num_corners):
            corner_pos = self.corners[i, 0]
            orientation = self.corners[i, 1]

            # Get colors of the piece (from its solved position)
            piece_facelets = corner_facelets[corner_pos]
            colors = [piece_facelets[j][0] for j in range(3)]  # Original face colors

            # Apply orientation
            rotated_colors = colors[-orientation:] + colors[:-orientation]

            # Write to the current position i
            target_facelets = corner_facelets[i]
            for j in range(3):
                face, row, col = target_facelets[j]
                faces[face, row, col] = rotated_colors[j]

        for i in range(self.num_edges):
            edge_pos = self.edges[i, 0]
            orientation = self.edges[i, 1]

            # Get colors of the piece
            piece_facelets = edge_facelets[edge_pos]
            colors = [piece_facelets[j][0] for j in range(2)]

            if orientation == 1:
                colors = colors[::-1]

            # Write to the current position i
            target_facelets = edge_facelets[i]
            for j in range(2):
                face, row, col = target_facelets[j]
                faces[face, row, col] = colors[j]

        cube = Cube(initial=faces, size=3)
        return cube

    def to_string(self) -> str:
        """Convert to facelet string format: U1-U9, R1-R9, F1-F9, D1-D9, L1-L9, B1-B9"""
        cube = self.to_cube()
        s = ""
        color_chars = {x.value: x.name for x in Color}

        for face in range(6):
            for row in range(3):
                for col in range(3):
                    s += color_chars[cube.faces[face, row, col]]
        return s

    def from_string(self, cube_string: str):
        """Load from facelet string format: U1-U9, R1-R9, F1-F9, D1-D9, L1-L9, B1-B9"""
        if len(cube_string) != 54:
            raise ValueError("Cube string must be 54 characters long")

        char_colors = {x.name: x.value for x in Color}
        faces = np.zeros((6, 3, 3), dtype=np.uint8)

        idx = 0
        for face in range(6):
            for row in range(3):
                for col in range(3):
                    faces[face, row, col] = char_colors[cube_string[idx]]
                    idx += 1

        cube = Cube(initial=faces, size=3)
        self.from_cube(cube)


# ------
# Basic moves

# Up-move
U_MOVE = CubieCube(
    corners=np.array(
        [
            [Corner.UBR, 0],
            [Corner.URF, 0],
            [Corner.UFL, 0],
            [Corner.ULB, 0],
            [Corner.DFR, 0],
            [Corner.DLF, 0],
            [Corner.DBL, 0],
            [Corner.DRB, 0],
        ]
    ),
    edges=np.array(
        [
            [Edge.UB, 0],
            [Edge.UR, 0],
            [Edge.UF, 0],
            [Edge.UL, 0],
            [Edge.DR, 0],
            [Edge.DF, 0],
            [Edge.DL, 0],
            [Edge.DB, 0],
            [Edge.FR, 0],
            [Edge.FL, 0],
            [Edge.BL, 0],
            [Edge.BR, 0],
        ]
    ),
)

# Right-move
R_MOVE = CubieCube(
    corners=np.array(
        [
            [Corner.DFR, 2],
            [Corner.UFL, 0],
            [Corner.ULB, 0],
            [Corner.URF, 1],
            [Corner.DRB, 1],
            [Corner.DLF, 0],
            [Corner.DBL, 0],
            [Corner.UBR, 2],
        ]
    ),
    edges=np.array(
        [
            [Edge.FR, 0],
            [Edge.UF, 0],
            [Edge.UL, 0],
            [Edge.UB, 0],
            [Edge.BR, 0],
            [Edge.DF, 0],
            [Edge.DL, 0],
            [Edge.DB, 0],
            [Edge.DR, 0],
            [Edge.FL, 0],
            [Edge.BL, 0],
            [Edge.UR, 0],
        ]
    ),
)

# Front-move
F_MOVE = CubieCube(
    corners=np.array(
        [
            [Corner.UFL, 1],
            [Corner.DLF, 2],
            [Corner.ULB, 0],
            [Corner.UBR, 0],
            [Corner.URF, 2],
            [Corner.DFR, 1],
            [Corner.DBL, 0],
            [Corner.DRB, 0],
        ]
    ),
    edges=np.array(
        [
            [Edge.UR, 0],
            [Edge.FL, 1],
            [Edge.UL, 0],
            [Edge.UB, 0],
            [Edge.DR, 0],
            [Edge.FR, 1],
            [Edge.DL, 0],
            [Edge.DB, 0],
            [Edge.UF, 1],
            [Edge.DF, 1],
            [Edge.BL, 0],
            [Edge.BR, 0],
        ]
    ),
)

# Down-move
D_MOVE = CubieCube(
    corners=np.array(
        [
            [Corner.URF, 0],
            [Corner.UFL, 0],
            [Corner.ULB, 0],
            [Corner.UBR, 0],
            [Corner.DLF, 0],
            [Corner.DBL, 0],
            [Corner.DRB, 0],
            [Corner.DFR, 0],
        ]
    ),
    edges=np.array(
        [
            [Edge.UR, 0],
            [Edge.UF, 0],
            [Edge.UL, 0],
            [Edge.UB, 0],
            [Edge.DF, 0],
            [Edge.DL, 0],
            [Edge.DB, 0],
            [Edge.DR, 0],
            [Edge.FR, 0],
            [Edge.FL, 0],
            [Edge.BL, 0],
            [Edge.BR, 0],
        ]
    ),
)

# Left-move
L_MOVE = CubieCube(
    corners=np.array(
        [
            [Corner.URF, 0],
            [Corner.ULB, 1],
            [Corner.DBL, 2],
            [Corner.UBR, 0],
            [Corner.DFR, 0],
            [Corner.UFL, 2],
            [Corner.DLF, 1],
            [Corner.DRB, 0],
        ]
    ),
    edges=np.array(
        [
            [Edge.UR, 0],
            [Edge.UF, 0],
            [Edge.BL, 0],
            [Edge.UB, 0],
            [Edge.DR, 0],
            [Edge.DF, 0],
            [Edge.FL, 0],
            [Edge.DB, 0],
            [Edge.FR, 0],
            [Edge.UL, 0],
            [Edge.DL, 0],
            [Edge.BR, 0],
        ]
    ),
)

# Back-move
B_MOVE = CubieCube(
    corners=np.array(
        [
            [Corner.URF, 0],
            [Corner.UFL, 0],
            [Corner.UBR, 1],
            [Corner.DRB, 2],
            [Corner.DFR, 0],
            [Corner.DLF, 0],
            [Corner.ULB, 2],
            [Corner.DBL, 1],
        ]
    ),
    edges=np.array(
        [
            [Edge.UR, 0],
            [Edge.UF, 0],
            [Edge.UL, 0],
            [Edge.BR, 1],
            [Edge.DR, 0],
            [Edge.DF, 0],
            [Edge.DL, 0],
            [Edge.BL, 1],
            [Edge.FR, 0],
            [Edge.FL, 0],
            [Edge.UB, 1],
            [Edge.DB, 1],
        ]
    ),
)

BasicMoves: dict[Color, CubieCube] = {
    Color.U: U_MOVE,
    Color.R: R_MOVE,
    Color.F: F_MOVE,
    Color.D: D_MOVE,
    Color.L: L_MOVE,
    Color.B: B_MOVE,
}

# -----------------------------------------------------------------------------------------------------------------
# ---------------------------------------- Basic Symmetries -------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------

# 120° clockwise rotation around the long diagonal URF-DBL
cpROT_URF3 = [
    Corner.URF,
    Corner.DFR,
    Corner.DLF,
    Corner.UFL,
    Corner.UBR,
    Corner.DRB,
    Corner.DBL,
    Corner.ULB,
]
coROT_URF3 = [1, 2, 1, 2, 2, 1, 2, 1]
epROT_URF3 = [
    Edge.UF,
    Edge.FR,
    Edge.DF,
    Edge.FL,
    Edge.UB,
    Edge.BR,
    Edge.DB,
    Edge.BL,
    Edge.UR,
    Edge.DR,
    Edge.DL,
    Edge.UL,
]
eoROT_URF3 = [1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1]

# 180° rotation around the axis through the F and B centers
cpROT_F2 = [
    Corner.DLF,
    Corner.DFR,
    Corner.DRB,
    Corner.DBL,
    Corner.UFL,
    Corner.URF,
    Corner.UBR,
    Corner.ULB,
]
coROT_F2 = [0, 0, 0, 0, 0, 0, 0, 0]
epROT_F2 = [
    Edge.DL,
    Edge.DF,
    Edge.DR,
    Edge.DB,
    Edge.UL,
    Edge.UF,
    Edge.UR,
    Edge.UB,
    Edge.FL,
    Edge.FR,
    Edge.BR,
    Edge.BL,
]
eoROT_F2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# 90° clockwise rotation around the axis through the U and D centers
cpROT_U4 = [
    Corner.UBR,
    Corner.URF,
    Corner.UFL,
    Corner.ULB,
    Corner.DRB,
    Corner.DFR,
    Corner.DLF,
    Corner.DBL,
]
coROT_U4 = [0, 0, 0, 0, 0, 0, 0, 0]
epROT_U4 = [
    Edge.UB,
    Edge.UR,
    Edge.UF,
    Edge.UL,
    Edge.DB,
    Edge.DR,
    Edge.DF,
    Edge.DL,
    Edge.BR,
    Edge.FR,
    Edge.FL,
    Edge.BL,
]
eoROT_U4 = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]

# reflection at the plane through the U, D, F, B centers
cpMIRR_LR2 = [
    Corner.UFL,
    Corner.URF,
    Corner.UBR,
    Corner.ULB,
    Corner.DLF,
    Corner.DFR,
    Corner.DRB,
    Corner.DBL,
]
coMIRR_LR2 = [3, 3, 3, 3, 3, 3, 3, 3]
epMIRR_LR2 = [
    Edge.UL,
    Edge.UF,
    Edge.UR,
    Edge.UB,
    Edge.DL,
    Edge.DF,
    Edge.DR,
    Edge.DB,
    Edge.FL,
    Edge.FR,
    Edge.BR,
    Edge.BL,
]
eoMIRR_LR2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


class BS(IntEnum):
    ROT_URF3 = 0
    ROT_F2 = 1
    ROT_U4 = 2
    MIRR_LR2 = 3


basicSymCube = [CubieCube()] * 4
basicSymCube[BS.ROT_URF3] = CubieCube(
    np.array([[c, o] for c, o in zip(cpROT_URF3, coROT_URF3)]),
    np.array([[e, o] for e, o in zip(epROT_URF3, eoROT_URF3)]),
)
basicSymCube[BS.ROT_F2] = CubieCube(
    np.array([[c, o] for c, o in zip(cpROT_F2, coROT_F2)]),
    np.array([[e, o] for e, o in zip(epROT_F2, eoROT_F2)]),
)
basicSymCube[BS.ROT_U4] = CubieCube(
    np.array([[c, o] for c, o in zip(cpROT_U4, coROT_U4)]),
    np.array([[e, o] for e, o in zip(epROT_U4, eoROT_U4)]),
)
basicSymCube[BS.MIRR_LR2] = CubieCube(
    np.array([[c, o] for c, o in zip(cpMIRR_LR2, coMIRR_LR2)]),
    np.array([[e, o] for e, o in zip(epMIRR_LR2, eoMIRR_LR2)]),
)

# ######################################## Fill SymCube list ###########################################################

# 48 CubieCubes will represent the 48 cube symmetries
symCube = []
cc = CubieCube()  # Identity cube
for urf3 in range(3):
    for f2 in range(2):
        for u4 in range(4):
            for lr2 in range(2):
                symCube.append(CubieCube(cc.corners.copy(), cc.edges.copy()))
                cc.multiply(basicSymCube[BS.MIRR_LR2])
            cc.multiply(basicSymCube[BS.ROT_U4])
        cc.multiply(basicSymCube[BS.ROT_F2])
    cc.multiply(basicSymCube[BS.ROT_URF3])
########################################################################################################################

# ########################################## Fill the inv_idx array ####################################################

# Indices for the inverse symmetries: SymCube[inv_idx[idx]] == SymCube[idx]^(-1)
inv_idx = np.zeros(k.N_SYM, dtype=np.uint8)
for j in range(k.N_SYM):
    for i in range(k.N_SYM):
        cc = CubieCube(symCube[j].corners.copy(), symCube[j].edges.copy())
        cc.corner_multiply(symCube[i])
        if (
            cc.corners[Corner.URF, 0] == Corner.URF
            and cc.corners[Corner.UFL, 0] == Corner.UFL
            and cc.corners[Corner.ULB, 0] == Corner.ULB
        ):
            inv_idx[j] = i
            break
########################################################################################################################


class Symmetries:
    def __init__(self, folder: str = k.FOLDER, show_progress: bool = True):
        self.folder = folder
        self.show_progress = show_progress
        self.console = Console()  # use for pretty printing and progress bars

        if not os.path.exists(self.folder):
            os.mkdir(self.folder)

    def create_tables(self):
        self.create_mult_sym_table()
        self.create_conj_move_table()
        self.create_twist_conj_table()
        self.create_ud_edges_conj_table()
        self.create_flipslice_tables()
        self.create_corner_tables()

    def create_mult_sym_table(self):
        """Generate the group table for the 48 cube symmetries"""
        mult_sym = np.zeros((k.N_SYM, k.N_SYM), dtype=np.uint8)
        for i in range(k.N_SYM):
            for j in range(k.N_SYM):
                cc = CubieCube(symCube[i].corners.copy(), symCube[i].edges.copy())
                cc.multiply(symCube[j])
                for k_sym in range(k.N_SYM):
                    if cc == symCube[k_sym]:  # SymCube[i]*SymCube[j] == SymCube[k]
                        mult_sym[i, j] = k_sym
                        break
        return mult_sym

    def create_conj_move_table(self):
        """Generate the table for the conjugation of a move m by a symmetry s. conj_move[N_MOVE*s + m] = s*m*s^-1"""
        conj_move = np.zeros((k.N_SYM, k.N_MOVE), dtype=np.uint16)
        for s in range(k.N_SYM):
            for m in Color:
                ss = CubieCube(
                    symCube[s].corners.copy(), symCube[s].edges.copy()
                )  # copy cube
                ss.multiply(BasicMoves[m])  # s*m
                ss.multiply(symCube[inv_idx[s]])  # s*m*s^-1
                for m2 in Color:
                    if ss == BasicMoves[m2]:
                        conj_move[s, m] = m2
        return conj_move

    def create_twist_conj_table(self):
        """Generate the phase 1 table for the conjugation of the twist t by a symmetry s. twist_conj[t, s] = s*t*s^-1"""
        fname = "conj_twist.npy"
        fpath = os.path.join(self.folder, fname)
        if not path.isfile(fpath):
            twist_conj = np.zeros((k.N_TWIST, k.N_SYM_D4h), dtype=np.uint16)
            for t in range(k.N_TWIST):
                cc = CubieCube()
                cc.set_twist(t)
                for s in range(k.N_SYM_D4h):
                    ss = CubieCube(
                        symCube[s].corners.copy(), symCube[s].edges.copy()
                    )  # copy cube
                    ss.corner_multiply(cc)  # s*t
                    ss.corner_multiply(symCube[inv_idx[s]])  # s*t*s^-1
                    twist_conj[t, s] = ss.get_twist()
            np.save(fpath, twist_conj)
        else:
            if self.show_progress:
                self.console.print(f"[bold green]Loading {fname} table...[/bold green]")
            # mmap_mode='r' allows for instant loading (memory mapping)
            # See: https://numpy.org/devdocs/reference/generated/numpy.memmap.html
            twist_conj = np.load(fpath, mmap_mode="r")
        return twist_conj

    def create_ud_edges_conj_table(self):
        """Generate the phase 2 table for the conjugation of the URtoDB coordinate by a symmetrie"""
        fname = "conj_ud_edges.npy"
        fpath = os.path.join(self.folder, fname)
        if not path.isfile(fpath):
            # if self.show_progress:
            # self.console.print(f"[bold blue]Creating {fname} table...[/bold blue]")
            ud_edges_conj = np.zeros((k.N_UD_EDGES, k.N_SYM_D4h), dtype=np.uint16)

            iterator = range(k.N_UD_EDGES)
            if self.show_progress:
                iterator = track(
                    iterator,
                    description=f"Generating {fname}...".ljust(
                        k.PROGRESS_BAR_DESC_WIDTH
                    ),
                )

            for t in iterator:
                cc = CubieCube()
                cc.set_ud_edges(t)
                for s in range(k.N_SYM_D4h):
                    ss = CubieCube(
                        symCube[s].corners.copy(), symCube[s].edges.copy()
                    )  # copy cube
                    ss.edge_multiply(cc)  # s*t
                    ss.edge_multiply(symCube[inv_idx[s]])  # s*t*s^-1
                    ud_edges_conj[t, s] = ss.get_ud_edges()
            np.save(fpath, ud_edges_conj)
        else:
            if self.show_progress:
                self.console.print(f"[bold green]Loading {fname} table...[/bold green]")
            ud_edges_conj = np.load(fpath, mmap_mode="r")
        return ud_edges_conj

    def create_flipslice_tables(self):
        """Generate the tables to handle the symmetry reduced flip-slice coordinate in  phase 1"""
        fname1 = "fs_classidx.npy"
        fname2 = "fs_sym.npy"
        fname3 = "fs_rep.npy"
        fpath1 = path.join(self.folder, fname1)
        fpath2 = path.join(self.folder, fname2)
        fpath3 = path.join(self.folder, fname3)

        if not (path.isfile(fpath1) and path.isfile(fpath2) and path.isfile(fpath3)):
            # if self.show_progress:
            # self.console.print(
            # "[bold blue]Creating flipslice sym-tables...[/bold blue]"
            # )
            flipslice_classidx = np.full(
                k.N_FLIP * k.N_SLICE, 65535, dtype=np.uint16
            )  # idx -> classidx
            flipslice_sym = np.zeros(
                k.N_FLIP * k.N_SLICE, dtype=np.uint8
            )  # idx -> symmetry
            flipslice_rep = np.zeros(
                k.N_FLIPSLICE_CLASS, dtype=np.uint32
            )  # classidx -> idx of representant

            classidx = 0
            cc = CubieCube()

            iterator = range(k.N_SLICE)
            if self.show_progress:
                iterator = track(
                    iterator,
                    description="Generating flipslice sym-tables...".ljust(
                        k.PROGRESS_BAR_DESC_WIDTH
                    ),
                )

            for slc in iterator:
                cc.set_slice(slc)
                for flip in range(k.N_FLIP):
                    cc.set_flip(flip)
                    idx = k.N_FLIP * slc + flip

                    if flipslice_classidx[idx] == 65535:
                        flipslice_classidx[idx] = classidx
                        flipslice_sym[idx] = 0
                        flipslice_rep[classidx] = idx
                    else:
                        continue
                    for s in range(
                        k.N_SYM_D4h
                    ):  # conjugate representant by all 16 symmetries
                        ss = CubieCube(
                            symCube[inv_idx[s]].corners.copy(),
                            symCube[inv_idx[s]].edges.copy(),
                        )  # copy cube
                        ss.edge_multiply(cc)
                        ss.edge_multiply(symCube[s])  # s^-1*cc*s
                        idx_new = k.N_FLIP * ss.get_slice() + ss.get_flip()
                        if flipslice_classidx[idx_new] == 65535:
                            flipslice_classidx[idx_new] = classidx
                            flipslice_sym[idx_new] = s
                    classidx += 1
            np.save(fpath1, flipslice_classidx)
            np.save(fpath2, flipslice_sym)
            np.save(fpath3, flipslice_rep)
            return flipslice_classidx, flipslice_sym, flipslice_rep

        else:
            if self.show_progress:
                self.console.print(
                    "[bold green]Loading flipslice sym-tables...[/bold green]"
                )
            flipslice_classidx = np.load(fpath1, mmap_mode="r")
            flipslice_sym = np.load(fpath2, mmap_mode="r")
            flipslice_rep = np.load(fpath3, mmap_mode="r")
            return flipslice_classidx, flipslice_sym, flipslice_rep

    def create_corner_tables(self):
        """Generate the tables to handle the symmetry reduced corner permutation coordinate in phase 2"""
        fname1 = "co_classidx.npy"
        fname2 = "co_sym.npy"
        fname3 = "co_rep.npy"
        fpath1 = path.join(self.folder, fname1)
        fpath2 = path.join(self.folder, fname2)
        fpath3 = path.join(self.folder, fname3)

        if not (path.isfile(fpath1) and path.isfile(fpath2) and path.isfile(fpath3)):
            # if self.show_progress:
            # self.console.print(
            # "[bold blue]Creating corner sym-tables...[/bold blue]"
            # )
            corner_classidx = np.full(
                k.N_CORNERS, 65535, dtype=np.uint16
            )  # idx -> classidx
            corner_sym = np.zeros(k.N_CORNERS, dtype=np.uint8)  # idx -> symmetry
            corner_rep = np.zeros(
                k.N_CORNERS_CLASS, dtype=np.uint16
            )  # classidx -> idx of representant

            classidx = 0
            cc = CubieCube()

            iterator = range(k.N_CORNERS)
            if self.show_progress:
                iterator = track(
                    iterator,
                    description="Generating corner sym-tables...".ljust(
                        k.PROGRESS_BAR_DESC_WIDTH
                    ),
                )

            for cp in iterator:
                cc.set_corners(cp)

                if corner_classidx[cp] == 65535:
                    corner_classidx[cp] = classidx
                    corner_sym[cp] = 0
                    corner_rep[classidx] = cp
                else:
                    continue
                for s in range(
                    k.N_SYM_D4h
                ):  # conjugate representant by all 16 symmetries
                    ss = CubieCube(
                        symCube[inv_idx[s]].corners.copy(),
                        symCube[inv_idx[s]].edges.copy(),
                    )  # copy cube
                    ss.corner_multiply(cc)
                    ss.corner_multiply(symCube[s])  # s^-1*cc*s
                    cp_new = ss.get_corners()
                    if corner_classidx[cp_new] == 65535:
                        corner_classidx[cp_new] = classidx
                        corner_sym[cp_new] = s
                classidx += 1
            np.save(fpath1, corner_classidx)
            np.save(fpath2, corner_sym)
            np.save(fpath3, corner_rep)
            return corner_classidx, corner_sym, corner_rep

        else:
            if self.show_progress:
                self.console.print(
                    "[bold green]Loading corner sym-tables...[/bold green]"
                )
            corner_classidx = np.load(fpath1, mmap_mode="r")
            corner_sym = np.load(fpath2, mmap_mode="r")
            corner_rep = np.load(fpath3, mmap_mode="r")
            return corner_classidx, corner_sym, corner_rep
