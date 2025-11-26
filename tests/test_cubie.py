import numpy as np
import pytest

from rubik.cube import Cube
from rubik.kociemba.cubie import BasicMoves, Color, CubieCube
from rubik.kociemba.defs import Constants as k


class TestCubie:
    def test_cube_conversions(self):
        c1 = Cube(initial="solved")  # create facelet cube
        cc = CubieCube()  # create cubie cube
        cc2 = cc.copy()
        cc.from_cube(c1)  # convert facelet cube to cubie cube
        s1 = cc.to_string()  # convert cubie cube to string
        cc2.from_string(s1)  # create new cubie cube from string
        c2 = cc2.to_cube()  # convert back to facelet cube
        assert (c1.faces == c2.faces).all()

        for i in range(20):
            c1 = Cube(initial="random", number_of_scramble_moves=i + 10)
            cc.from_cube(c1)
            s1 = cc.to_string()
            cc2.from_string(s1)
            c2 = cc2.to_cube()
            assert (c1.faces == c2.faces).all()

    @pytest.mark.parametrize(
        "test_cases",
        [
            [(400, [Color.F])],
            [(400, [Color.B])],
            [(400, [Color.U])],
            [(400, [Color.D])],
            [(400, [Color.R])],
            [(400, [Color.L])],
            [(6, [Color.F, Color.F, Color.R, Color.R])],
            [(6, [Color.B, Color.B, Color.L, Color.L])],
            [(6, [Color.L, Color.L, Color.F, Color.F])],
        ],
    )
    def test_identity_cycles(self, test_cases):
        cubie = CubieCube()
        cube = cubie.to_cube()
        original_state = cube.faces.copy()

        for repetitions, moves in test_cases:
            for _ in range(repetitions):
                for move in moves:
                    cubie.multiply(BasicMoves[move])

        cube = cubie.to_cube()
        assert (cube.faces == original_state).all(), (
            "Cube state does not match original solved state"
        )

    @pytest.mark.parametrize(
        "sequence",
        [
            [Color.F],
            [Color.B],
            [Color.U],
            [Color.D],
            [Color.R],
            [Color.L],
            [Color.F, Color.F, Color.R, Color.R],
            [Color.B, Color.B, Color.L, Color.L],
            [Color.L, Color.L, Color.F, Color.F],
        ],
    )
    def test_inverses(self, sequence):
        cubie = CubieCube()
        cubie_original = cubie.copy()
        moves = [BasicMoves[move] for move in sequence]
        moves_inverse = [move.inverse() for move in reversed(moves)]
        for move in moves:
            cubie.multiply(move)

        for move in moves_inverse:
            cubie.multiply(move)

        assert cubie == cubie_original

    def test_flips(self):
        cube = CubieCube()
        for i in range(2**11):
            cube.set_flip(i)
            assert i == cube.get_flip()

    def test_twists(self):
        cube = CubieCube()
        for i in range(3**7):
            cube.set_twist(i)
            assert i == cube.get_twist()

    def test_slices(self):
        cube = CubieCube()
        for i in range(495):
            cube.set_slice(i)
            assert i == cube.get_slice()

    def test_slice_sorted(self):
        cube = CubieCube()
        for i in range(11880):
            cube.set_slice_sorted(i)
            assert i == cube.get_slice_sorted()

    def test_u_edges(self):
        cube = CubieCube()
        for i in range(11880):
            cube.set_u_edges(i)
            assert i == cube.get_u_edges()

    def test_d_edges(self):
        cube = CubieCube()
        for i in range(11880):
            cube.set_d_edges(i)
            assert i == cube.get_d_edges()

    def test_ud_edges(self):
        cube = CubieCube()
        for i in range(40320):
            cube.set_ud_edges(i)
            assert i == cube.get_ud_edges()

    def test_corners(self):
        cube = CubieCube()
        for i in range(40320):
            cube.set_corners(i)
            c = cube.get_corners()
            assert i == c


class TestSymmetries:
    def test_symmetry_count(self):
        cube = CubieCube()
        syms = cube.symmetries()
        assert len(syms) == 2 * k.N_SYM
        assert sorted(syms) == list(range(2 * k.N_SYM))

    def test_symmetries_scrambled_cube(self):
        c = Cube(initial="random", number_of_scramble_moves=3)
        cc = CubieCube()
        cc.from_cube(c)
        syms = cc.symmetries()
        assert isinstance(syms, list)
        assert len(syms) <= 2 * k.N_SYM
