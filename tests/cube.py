
import numpy as np
import pytest

from rubik.cube import Cube, Move


class TestCube:
    N = 1000000

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
        [
            Move.F,
            Move.U,
            Move.B,
            Move.D,
            Move.R,
            Move.L,
        ],
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
