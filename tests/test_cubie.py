import numpy as np
import pytest

from rubik.cube import Cube, Move
from rubik.cubie_cube import BasicMoves, Color, CubieCube


class TestCubie:
    def test_cube_conversions(self):
        c1 = Cube(initial="solved") # create facelet cube
        cc = CubieCube() # create cubie cube
        cc2 = cc.copy()
        cc.from_cube(c1) # convert facelet cube to cubie cube
        s1 = cc.to_string() # convert cubie cube to string
        cc2.from_string(s1) # create new cubie cube from string
        c2 = cc2.to_cube() # convert back to facelet cube
        assert (c1.faces == c2.faces).all()

        for i in range(20):
            c1 = Cube(initial="random", number_of_scramble_moves=i+10)
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

    def test_aux_functions(self):
        cube = CubieCube()
        N = 20
        a = np.random.randint(0, N, N)
        a0 = a.copy()

        for right in range(N):
            for left in range(right, N):
                cube.rotate_right(a, right, left)
                cube.rotate_left(a, right, left)
                assert np.array_equal(a0, a)
