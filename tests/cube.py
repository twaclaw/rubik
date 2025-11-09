import numpy as np
import pytest

from rubik.cube import Cube, Move
from rubik.cubie_cube import BasicMoves, Color, CubieCube


class TestCube:
    N = 10000

    def test_solved(self):
        cube = Cube(initial="solved")
        assert cube.is_solution()
        cube = Cube(initial="random")
        assert not cube.is_solution()

    @pytest.mark.parametrize(
        "elements",
        [
            [Move.F],
            [Move.U],
            [Move.B],
            [Move.D],
            [Move.R],
            [Move.L],
            [Move.E, Move.M, Move.S],
            [Move.F, Move.B, Move.U, Move.D],
            [Move.B, Move.R, Move.L],
            [Move.F, Move.U, Move.B, Move.D, Move.R, Move.L],
        ],
    )
    def test_moves(self, elements):
        for size in [2, 3, 4, 5]:
            cube = Cube(initial="solved", size=size)
            moves = [Move(np.random.choice(elements)) for _ in range(self.N)]
            moves_inverse = [move.inverse() for move in reversed(moves)]
            cube.moves([move.name for move in moves])
            cube.moves([move.name for move in moves_inverse])
            assert cube.is_solution()

    @pytest.mark.parametrize(
        "test_cases",
        [
            [(400, ["F"])],
            [(400, ["M"])],
            [(400, ["E"])],
            [(400, ["S"])],
            [(6, ["F", "F", "R", "R"])],
            [(1, ["F", "f", "U", "u", "R", "r", "D", "d", "L", "l", "B", "b"])],
            [(6, ["B", "B", "L", "L"])],
            [(6, ["L", "L", "F", "F"])],
            [(1, ["R", "U", "r", "f", "U", "F", "u", "r", "F", "R", "f", "u"])],
            [(6, ["R", "U", "r", "u"])],  # sexy move
        ],
    )
    def test_identity_cycles(self, test_cases):
        for initial in ["solved", "random"]:
            for size in [2, 3, 4, 5]:
                cube = Cube(initial=initial, size=size)
                original_state = cube.faces.copy()

                for repetitions, moves in test_cases:
                    for _ in range(repetitions):
                        cube.moves(moves)

                assert (cube.faces == original_state).all(), (
                    "Cube state does not match original solved state"
                )

    def test_compress_decompress(self):
        for initial in ["solved", "random"]:
            for size in [2, 3, 4, 5]:
                for _ in range(self.N):
                    cube = Cube(initial=initial, size=size)
                    original_state = cube.faces.copy()

                    compressed = cube.compress()
                    cube.decompress(compressed)

                    assert (cube.faces == original_state).all(), (
                        "Cube state does not match original solved state"
                    )

    @pytest.mark.parametrize(
        "move",
        [Move.F, Move.U, Move.B, Move.D, Move.R, Move.L, Move.E, Move.M, Move.S],
    )
    def test_inverses(self, move):
        for size in [2, 3, 4, 5]:
            cube = Cube(initial="random", size=size)
            c0 = cube.faces.copy()
            for _ in range(4 * 100):
                cube.move(move.inverse())
            assert (cube.faces == c0).all()

            c0 = cube.faces.copy()
            for _ in range(self.N):
                cube.move(move)
                cube.move(move.inverse())
                assert (cube.faces == c0).all()


class TestCubie:
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
