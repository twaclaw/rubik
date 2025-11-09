"""
This implementation follows the implementation by Herbert Kociemba.
https://github.com/hkociemba/RubiksCube-TwophaseSolver.git

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
from enum import IntEnum

import numpy as np

from .cube import Cube


class Color(IntEnum):
    U = 0  # Up
    R = 1  # Right
    F = 2  # Front
    D = 3  # Down
    L = 4  # Left
    B = 5  # Back


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
        params:
        - corners: shape (8, 2) -- first column: which corner, second column: orientation
        - edges: shape (12, 2) -- first column: which edge, second column: orientation
        - Orientations can be:
         - 0 -> correctly oriented
         - 1 -> rotated clockwise (for corners) or flipped (for edges)
         - 2 -> rotated counter-clockwise (for corners) or flipped (for edges)
        """
        self.num_corners = 8
        self.num_edges = 12
        self._comb_vectorized = np.vectorize(
            self.c_nk
        )  # TODO: consider using scipy.special.comb

        # used for slice coordinate calculations
        self._slice_edge = np.array([Edge.FR, Edge.FL, Edge.BL, Edge.BR])
        self._other_edge = np.array([Edge.UR, Edge.UF, Edge.UL, Edge.UB, Edge.DR, Edge.DF, Edge.DL, Edge.DB])

        if corners is None:
            self.corners = np.array([[Corner(i), 0] for i in range(8)], dtype=int)
        else:
            self.corners = corners.copy()

        if edges is None:
            self.edges = np.array([[Edge(i), 0] for i in range(12)], dtype=int)
        else:
            self.edges = edges

    @staticmethod
    def rotate_right(a: np.ndarray, left: int, right: int):
        """In-place rotate right of a[left:right]"""
        a[left : right + 1] = np.roll(a[left : right + 1], 1)

    @staticmethod
    def rotate_left(a: np.ndarray, left: int, right: int):
        """In-place rotate left of a[left:right]"""
        a[left : right + 1] = np.roll(a[left : right + 1], -1)

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
        ep_a = self.edges[:, 0]  # permuatation
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

    # Functions defining coordinates for two-phase algorithm

    def get_twist(self) -> int:
        """Compute the twist coordinate (corner orientations) of this CubieCube."""
        return np.polyval(self.corners[:-1, 1], 3).astype(int)

    def set_twist(self, twist: int) -> None:
        """Set the corner orientations of this CubieCube from the given twist coordinate."""
        powers = 3 ** np.arange(self.num_corners - 1)
        div = twist // powers
        twistparity = np.sum(self.corners[:-1, 1]) % 3
        self.corners[:, 1] = np.concat(
            [np.flip(np.mod(div, 3)), [(3 - twistparity) % 3]]
        )

    def get_flip(self) -> int:
        """Compute the flip coordinate (edge orientations) of this CubieCube."""
        return np.polyval(self.edges[:-1, 1], 2).astype(int)

    def set_flip(self, flip: int) -> None:
        """Set the edge orientations of this CubieCube from the given flip coordinate."""
        powers = 2 ** np.arange(self.num_edges - 1)
        div = flip // powers
        flipparity = np.sum(self.edges[:, 1]) % 2
        self.edges[:, 1] = np.concat([np.flip(np.mod(div, 2)), [(2 - flipparity) % 2]])

    def get_slice(self) -> int:
        """Get the location of the UD-slice edges FR,FL,BL and BR ignoring their permutation.
        0<= slice < 495 in phase 1, slice = 0 in phase 2."""
        mask = (self.edges[:, 0] >= Edge.FR) & (self.edges[:, 0] <= Edge.BR)
        j = np.where(mask)[0][::-1] # flipped order
        x = np.arange(len(j))
        return np.sum(self._comb_vectorized(11 - j, x + 1))

    def set_slice(self, idx: int):
        ep = np.full(12, -1, dtype=np.int8) # invalidate all edge positions
        x = 4
        #TODO: check if this can be optimized
        for j in np.arange(self.num_edges):
            comb = self.c_nk(11 - j, x)
            if idx - comb >= 0:
                ep[j] = self._slice_edge[4 - x]
                idx -= comb
                x -= 1

        mask = (ep == -1)
        ep[mask] = self._other_edge[:np.sum(mask)]
        self.edges[:, 0] = ep

    # --- End of coordinate functions ---

    def to_cube(self) -> Cube:
        """Convert this CubieCube representation to a Facelet Cube representation."""

        # Initialize faces array
        faces = np.zeros((6, 3, 3), dtype=np.uint8)

        # Corner facelet positions for each corner in solved state
        # Each corner has 3 facelets, listed in clockwise order when viewed from outside
        # Face order: U=0, R=1, F=2, D=3, L=4, B=5
        corner_facelets = [
            [(0, 2, 2), (1, 0, 2), (2, 0, 2)],  # Corner URF: U-R-F
            [(0, 2, 0), (2, 0, 0), (4, 0, 0)],  # Corner UFL: U-F-L
            [(0, 0, 0), (4, 0, 2), (5, 0, 0)],  # Corner ULB: U-L-B
            [(0, 0, 2), (5, 0, 2), (1, 0, 0)],  # Corner UBR: U-B-R
            [(3, 2, 2), (2, 2, 2), (1, 2, 0)],  # Corner DFR: D-F-R
            [(3, 2, 0), (4, 2, 2), (2, 2, 0)],  # Corner DLF: D-L-F
            [(3, 0, 0), (5, 2, 0), (4, 2, 0)],  # Corner DBL: D-B-L
            [(3, 0, 2), (1, 2, 2), (5, 2, 2)],  # Corner DRB: D-R-B
        ]

        # Each edge has 2 facelets
        edge_facelets = [
            [(0, 1, 2), (1, 0, 1)],  # Edge UR: U-R
            [(0, 2, 1), (2, 0, 1)],  # Edge UF: U-F
            [(0, 1, 0), (4, 0, 1)],  # Edge UL: U-L
            [(0, 0, 1), (5, 0, 1)],  # Edge UB: U-B
            [(3, 1, 2), (1, 2, 1)],  # Edge DR: D-R
            [(3, 2, 1), (2, 2, 1)],  # Edge DF: D-F
            [(3, 1, 0), (4, 2, 1)],  # Edge DL: D-L
            [(3, 0, 1), (5, 2, 1)],  # Edge DB: D-B
            [(2, 1, 2), (1, 1, 2)],  # Edge FR: F-R
            [(2, 1, 0), (4, 1, 0)],  # Edge FL: F-L
            [(5, 1, 0), (4, 1, 2)],  # Edge BL: B-L
            [(5, 1, 2), (1, 1, 0)],  # Edge BR: B-R
        ]

        # Set center facelets (they don't move)
        for face in range(6):
            faces[face, 1, 1] = face

        for i in range(self.num_corners):
            corner_pos = self.corners[i, 0]
            orientation = self.corners[i, 1]

            facelets = corner_facelets[corner_pos]

            colors = [facelets[j][0] for j in range(3)]  # Original face colors
            rotated_colors = colors[orientation:] + colors[:orientation]

            for j in range(3):
                face, row, col = facelets[j]
                faces[face, row, col] = rotated_colors[j]

        for i in range(self.num_edges):
            edge_pos = self.edges[i, 0]
            orientation = self.edges[i, 1]

            facelets = edge_facelets[edge_pos]

            colors = [facelets[j][0] for j in range(2)]
            if orientation == 1:
                colors = colors[::-1]

            for j in range(2):
                face, row, col = facelets[j]
                faces[face, row, col] = colors[j]

        cube = Cube(initial=faces, size=3)
        return cube


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
