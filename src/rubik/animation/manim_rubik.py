"""
This code is a subset modified from this package: https://manim-rubikscube.readthedocs.io/en/stable/
at https://github.com/WampyCakes/manim-rubikscube
"""

import manim as mn
import numpy as np

my_template = mn.TexTemplate()
my_template.add_to_preamble(r"\usepackage{xcolor}")

# --- Utils ---
def normalize_moves(moves: list[str]) -> list[str]:
    return [m.upper() + "'" if m.islower() else m for m in moves]

def htm_str_inverse(moves: str) -> str:
    return [m.swapcase() for m in moves[::-1]]



def get_axis_from_face(face):
    if face == "F" or face == "B":
        return mn.X_AXIS
    elif face == "U" or face == "D":
        return mn.Z_AXIS
    else:
        return mn.Y_AXIS


def get_faces_of_cubie(dim, position):
    dim = dim - 1
    try:
        faces = {
            # Front corners
            (0, 0, 0): [mn.LEFT, mn.DOWN, mn.IN],
            (0, 0, dim): [mn.LEFT, mn.DOWN, mn.OUT],
            (0, dim, 0): [mn.LEFT, mn.UP, mn.IN],
            (0, dim, dim): [mn.LEFT, mn.UP, mn.OUT],
            # Back corners
            (dim, 0, 0): [mn.RIGHT, mn.DOWN, mn.IN],
            (dim, 0, dim): [mn.RIGHT, mn.DOWN, mn.OUT],
            (dim, dim, 0): [mn.RIGHT, mn.UP, mn.IN],
            (dim, dim, dim): [mn.RIGHT, mn.UP, mn.OUT],
        }
        return faces[position]
    except Exception:
        x = position[0]
        y = position[1]
        z = position[2]

        if x == 0:
            if y == 0:
                return [mn.DOWN, mn.LEFT]
            elif y == dim:
                return [mn.UP, mn.LEFT]
            else:
                if z == 0:
                    return [mn.IN, mn.LEFT]
                elif z == dim:
                    return [mn.OUT, mn.LEFT]
                else:
                    return [mn.LEFT]
        elif x == dim:
            if y == 0:
                return [mn.DOWN, mn.RIGHT]
            elif y == dim:
                return [mn.UP, mn.RIGHT]
            else:
                if z == 0:
                    return [mn.IN, mn.RIGHT]
                elif z == dim:
                    return [mn.OUT, mn.RIGHT]
                else:
                    return [mn.RIGHT]
        else:
            if y == 0:
                if z == 0:
                    return [mn.IN, mn.DOWN]
                elif z == dim:
                    return [mn.OUT, mn.DOWN]
                else:
                    return [mn.DOWN]
            elif y == dim:
                if z == 0:
                    return [mn.IN, mn.UP]
                elif z == dim:
                    return [mn.OUT, mn.UP]
                else:
                    return [mn.UP]
            else:
                if z == 0:
                    return [mn.IN]
                elif z == dim:
                    return [mn.OUT]
                else:
                    return []


# --- Classes ---


class Cubie(mn.VGroup):
    def __init__(self, x, y, z, dim, colors, **kwargs):
        super().__init__(**kwargs)
        self.dimensions = dim
        self.colors = colors
        self.position = np.array([x, y, z])
        self.faces = {}
        self.create_faces()

    def get_position(self):
        return self.position

    def get_rounded_center(self):
        return tuple(
            [
                round(float(self.get_x()), 3),
                round(float(self.get_y()), 3),
                round(float(self.get_z()), 3),
            ]
        )

    def create_faces(self):
        faces = np.array(
            get_faces_of_cubie(
                self.dimensions, (self.position[0], self.position[1], self.position[2])
            )
        ).tolist()
        i = 0
        for vect in mn.OUT, mn.DOWN, mn.LEFT, mn.IN, mn.UP, mn.RIGHT:
            face = mn.Square(side_length=2, shade_in_3d=True, stroke_width=3)
            if vect.tolist() in faces:
                face.set_fill(self.colors[i], 1)
            else:
                face.set_fill(mn.BLACK, 1)

            face.flip()
            face.shift(2 * mn.OUT / 2.0)
            face.apply_matrix(mn.utils.space_ops.z_to_vector(vect))

            self.faces[tuple(vect)] = face
            self.add(face)
            i += 1

    def get_face(self, face):
        if face == "F":
            return self.faces[tuple(mn.LEFT)]
        elif face == "B":
            return self.faces[tuple(mn.RIGHT)]
        elif face == "R":
            return self.faces[tuple(mn.DOWN)]
        elif face == "L":
            return self.faces[tuple(mn.UP)]
        elif face == "U":
            return self.faces[tuple(mn.OUT)]
        elif face == "D":
            return self.faces[tuple(mn.IN)]


class RubiksCube(mn.VMobject):
    def __init__(
        self,
        dim=3,
        colors=[mn.WHITE, "#0051BA", "#C41E3A", "#FFD500", "#009E60", "#FF5800"], # U, R, F, D, L, B
        x_offset=2.1,
        y_offset=2.1,
        z_offset=2.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if not (dim >= 2):
            raise Exception("Dimension must be >= 2")

        self.dimensions = dim
        self.colors = colors
        self.x_offset = [[mn.Mobject.shift, [x_offset, 0, 0]]]
        self.y_offset = [[mn.Mobject.shift, [0, y_offset, 0]]]
        self.z_offset = [[mn.Mobject.shift, [0, 0, z_offset]]]
        self.indices = {}

        self.cubies = np.ndarray((dim, dim, dim), dtype=Cubie)
        self.generate_cubies()
        self.move_to(mn.ORIGIN)
        self.set_indices()

    def generate_cubies(self):
        for x in range(self.dimensions):
            for y in range(self.dimensions):
                for z in range(self.dimensions):
                    cubie = Cubie(x, y, z, self.dimensions, self.colors)
                    self.transform_cubie(x, self.x_offset, cubie)
                    self.transform_cubie(y, self.y_offset, cubie)
                    self.transform_cubie(z, self.z_offset, cubie)
                    self.add(cubie)
                    self.cubies[x, y, z] = cubie

    def transform_cubie(self, position, offset, tile):
        offsets_nr = len(offset)
        for i in range(offsets_nr):
            for j in range(int(len(offset[i]) / 2)):
                if position < 0:
                    magnitude = len(range(-i, position, -offsets_nr)) * -1
                    offset[-1 - i][0 + j * 2](
                        tile, magnitude * np.array(offset[-1 - i][1 + j * 2])
                    )
                else:
                    magnitude = len(range(i, position, offsets_nr))
                    offset[i][0 + j * 2](
                        tile, magnitude * np.array(offset[i][1 + j * 2])
                    )

    def get_face(self, face, flatten=True):
        if face == "F":
            face_cubies = self.cubies[0, :, :]
        elif face == "B":
            face_cubies = self.cubies[self.dimensions - 1, :, :]
        elif face == "U":
            face_cubies = self.cubies[:, :, self.dimensions - 1]
        elif face == "D":
            face_cubies = self.cubies[:, :, 0]
        elif face == "L":
            face_cubies = self.cubies[:, self.dimensions - 1, :]
        elif face == "R":
            face_cubies = self.cubies[:, 0, :]
        else:
            return None

        if flatten:
            return face_cubies.flatten()
        else:
            return face_cubies

    def set_indices(self):
        self.indices = {}
        for c in self.cubies.flatten():
            self.indices[c.get_rounded_center()] = c.position
        print("INITIAL INDICES SAMPLE:", list(self.indices.keys())[:5])
        print("ALL INDICES:", list(self.indices.keys()))

    def set_state(self, positions):
        """Set the cube to a specific state.

        For instance:
        Solved state -> UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB
        Scrambled -> DRLUUBFBRBLURRLRUBLRDDFDLFUFUFFDBDBUUBDLRDLRFLBRFBFUDL
        """
        colors = {
            "U": self.colors[0],
            "R": self.colors[1],
            "F": self.colors[2],
            "D": self.colors[3],
            "L": self.colors[4],
            "B": self.colors[5],
        }
        positions = list(positions)

        for cubie in np.rot90(self.get_face("U", False), 2).flatten():
            cubie.get_face("U").set_fill(colors[positions.pop(0)], 1)

        for cubie in np.rot90(np.flip(self.get_face("R", False), (0, 1)), -1).flatten():
            cubie.get_face("R").set_fill(colors[positions.pop(0)], 1)

        for cubie in np.rot90(np.flip(self.get_face("F", False), 0)).flatten():
            cubie.get_face("F").set_fill(colors[positions.pop(0)], 1)

        for cubie in np.rot90(np.flip(self.get_face("D", False), 0), 2).flatten():
            cubie.get_face("D").set_fill(colors[positions.pop(0)], 1)

        for cubie in np.rot90(np.flip(self.get_face("L", False), 0)).flatten():
            cubie.get_face("L").set_fill(colors[positions.pop(0)], 1)

        for cubie in np.rot90(np.flip(self.get_face("B", False), (0, 1)), -1).flatten():
            cubie.get_face("B").set_fill(colors[positions.pop(0)], 1)

    def adjust_indices(self, cubies):
        for c in cubies.flatten():
            center = c.get_rounded_center()
            if center not in self.indices:
                ...
            loc = self.indices[center]
            self.cubies[loc[0], loc[1], loc[2]] = c

    def perform_move(self, move):
        face = move[0]
        axis = get_axis_from_face(face)
        angle = mn.PI / 2 if ("R" in move or "F" in move or "D" in move) else -mn.PI / 2
        angle = angle if "2" not in move else angle * 2
        angle = -angle if "'" in move else angle

        mn.VGroup(*self.get_face(face)).rotate(angle, axis)
        self.adjust_indices(self.get_face(face, False))

    def perform_sequence(self, moves):
        if isinstance(moves, str):
            moves = moves.split()
        for move in moves:
            self.perform_move(move)

    def scale(self, *args, **kwargs):
        super().scale(*args, **kwargs)
        self.set_indices()
        return self

    def shift(self, *args, **kwargs):
        super().shift(*args, **kwargs)
        self.set_indices()
        return self


class CubeMove(mn.Animation):
    def __init__(self, mobject, face, **kwargs):
        self.axis = get_axis_from_face(face[0])
        self.face = face
        self.angle = mn.PI / 2 if ("R" in face or "F" in face or "D" in face) else -mn.PI / 2
        self.angle = self.angle if "2" not in face else self.angle * 2
        self.angle = -self.angle if "'" in face else self.angle
        super().__init__(mobject, **kwargs)

    def create_starting_mobject(self):
        starting_mobject = self.mobject.copy()
        if starting_mobject.indices == {}:
            starting_mobject.set_indices()
        return starting_mobject

    def interpolate_mobject(self, alpha):
        self.mobject.become(self.starting_mobject)

        mn.VGroup(*self.mobject.get_face(self.face[0])).rotate(
            alpha * self.angle, self.axis
        )

    def finish(self):
        super().finish()
        self.mobject.adjust_indices(self.mobject.get_face(self.face[0], False))


class RubiksCubeAnimation(mn.ThreeDScene):
    """Animates the solving of a Rubik's Cube given the phase1 and phase2  sequences of Kociemba's algorithm"""
    def __init__(self, moves_ph1, moves_ph2, initial_state, moves_ph1_htm=None, moves_ph2_htm=None, **kwargs):
        self.moves_ph1 = moves_ph1
        self.moves_ph2 = moves_ph2
        self.moves_ph1_htm = moves_ph1_htm if moves_ph1_htm is not None else moves_ph1
        self.moves_ph2_htm = moves_ph2_htm if moves_ph2_htm is not None else moves_ph2
        self.initial_state = initial_state
        super().__init__(**kwargs)

    def rotate(self, delay: int = 10, rate: float = 0.5):
        self.begin_ambient_camera_rotation(rate=rate)
        self.wait(delay)
        self.stop_ambient_camera_rotation()
        self.move_camera(phi=60 * mn.DEGREES, theta=-135 * mn.DEGREES)


    def construct(self):
        self.cube = RubiksCube().scale(0.6)
        self.cube.move_to(mn.ORIGIN)

        if self.initial_state:
            if isinstance(self.initial_state, str) and len(self.initial_state) == 54 and " " not in self.initial_state:
                self.cube.set_state(self.initial_state)
            else:
                self.cube.perform_sequence(self.initial_state)

        self.add(self.cube)
        self.set_camera_orientation(phi=60 * mn.DEGREES, theta=-135 * mn.DEGREES)

        label1 = mn.Text(f"Initial string: {self.initial_state}", font_size=24).to_corner(mn.UL)
        self.add_fixed_in_frame_mobjects(label1)
        self.rotate(delay=6, rate=0.5)
        self.remove(label1)

        if self.moves_ph1:
            self.animate_sequence(self.moves_ph1, self.moves_ph1_htm, "Applying phase 1 moves: ")

        self.rotate(delay=6, rate=0.5)
        if self.moves_ph2:
            self.animate_sequence(self.moves_ph2, self.moves_ph2_htm, "Applying phase 2 moves: ")

        self.rotate(delay=6, rate=0.5)


    def animate_sequence(self, moves, moves_htm, prefix):
        current_text = ""
        label = mn.Text(prefix, font_size=36).to_corner(mn.UL)
        self.add_fixed_in_frame_mobjects(label)
        self.wait(1)

        for i, move in enumerate(moves):
            if i < len(moves_htm):
                current_text += moves_htm[i] + " "
                self.remove(label)
                label = mn.Text(prefix + current_text, font_size=36).to_corner(mn.UL)
                self.add_fixed_in_frame_mobjects(label)

            self.play(CubeMove(self.cube, move), run_time=0.5)

            self.wait(0.1)

        self.wait(2)
        self.remove(label)


class RubiksCubeStatic(mn.ThreeDScene):
    """Displays static Rubik's Cube states given the phase1 and phase2  sequences of Kociemba's algorithm
    - Initial state
    - State after phase 1
    - State after phase 2 -> solved cube
    """
    def __init__(self, moves_ph1, moves_ph2, initial_state, moves_ph1_htm=None, moves_ph2_htm=None, **kwargs):
        self.moves_ph1 = moves_ph1
        self.moves_ph2 = moves_ph2
        self.moves_ph1_htm = moves_ph1_htm if moves_ph1_htm is not None else moves_ph1
        self.moves_ph2_htm = moves_ph2_htm if moves_ph2_htm is not None else moves_ph2
        self.initial_state = initial_state
        super().__init__(**kwargs)

    def construct(self):
        self.set_camera_orientation(phi=60 * mn.DEGREES, theta=-135 * mn.DEGREES)

        # Cube 1: Initial
        cube1 = RubiksCube().scale(0.4)
        cube1.move_to(mn.LEFT * 4.5)
        self.setup_cube(cube1, self.initial_state)

        # Cube 2: After Phase 1
        cube2 = RubiksCube().scale(0.4)
        cube2.move_to(mn.ORIGIN)
        self.setup_cube(cube2, self.initial_state)
        if self.moves_ph1:
            cube2.perform_sequence(self.moves_ph1)

        # Cube 3: After Phase 2
        cube3 = RubiksCube().scale(0.4)
        cube3.move_to(mn.RIGHT * 4.5)
        self.setup_cube(cube3, self.initial_state)
        if self.moves_ph1:
            cube3.perform_sequence(self.moves_ph1)
        if self.moves_ph2:
            cube3.perform_sequence(self.moves_ph2)

        self.add(cube1, cube2, cube3)

    def setup_cube(self, cube, initial_state):
        if initial_state:
            if isinstance(initial_state, str) and len(initial_state) == 54 and " " not in initial_state:
                cube.set_state(initial_state)
            else:
                cube.perform_sequence(initial_state)


class MeetInTheMiddleAnimation(mn.ThreeDScene):
    """Animates the meet-in-the-middle solving of a Rubik's Cube given two sequences of moves from both ends"""
    def __init__(self, initial_state_1=None, moves_1=None, initial_state_2=None, moves_2=None, **kwargs):
        self.initial_state_1 = initial_state_1
        self.moves_1_htm = moves_1 or []
        self.initial_state_2 = initial_state_2
        self.moves_2_htm = moves_2 or []

        self.moves_1 = normalize_moves(self.moves_1_htm)
        self.moves_2 = normalize_moves(self.moves_2_htm)
        super().__init__(**kwargs)


    def construct(self):
        self.cube1 = RubiksCube().scale(0.5)
        self.cube2 = RubiksCube().scale(0.5)

        self.cube1.move_to((mn.LEFT + mn.UP) * 3)
        self.cube2.move_to((mn.RIGHT + mn.DOWN) * 3)

        self.setup_cube(self.cube1, self.initial_state_1)
        self.setup_cube(self.cube2, self.initial_state_2)

        self.add(self.cube1)
        self.add(self.cube2)

        self.set_camera_orientation(phi=60 * mn.DEGREES, theta=-135 * mn.DEGREES)

        self.wait(1)

        self.animate_sequences(self.moves_1, self.moves_2, delay=0.5)

    def setup_cube(self, cube, initial_state):
        if initial_state:
            if isinstance(initial_state, str) and len(initial_state) == 54 and " " not in initial_state:
                cube.set_state(initial_state)
            else:
                cube.perform_sequence(initial_state)

    def animate_sequences(self, moves1, moves2, delay: float = 1.0):
        len1 = len(moves1)
        len2 = len(moves2)
        max_len = max(len1, len2)

        str1 = ""
        str2 = ""

        text1 = mn.Text(str1, font_size=24).to_corner(mn.DL).shift(mn.UP * 0.5 + mn.RIGHT * 0.5)
        text2 = mn.Text(str2, font_size=24).to_corner(mn.DR).shift(mn.UP * 0.5 + mn.LEFT * 0.5)
        bwd_str = "".join(self.moves_2_htm)
        bwd_str_inv = "".join(htm_str_inverse(self.moves_2_htm))
        fwd_str = "".join(self.moves_1_htm)

        solution_text = mn.Tex(fr"\textbf{{Solution:}} \boldmath${fwd_str} + ({{\color{{yellow}}{bwd_str}}})^{{-1}} = {fwd_str}{{\color{{yellow}}{bwd_str_inv}}}$", font_size=42, tex_template=my_template)
        text3 = mn.Text("Applying solution to original cube:", font_size=24).next_to(solution_text, mn.DOWN)

        self.add_fixed_in_frame_mobjects(text1, text2)

        for i in range(max_len):
            animations = []
            if i < len1:
                animations.append(CubeMove(self.cube1, moves1[i]))
                if i < len(self.moves_1_htm):
                    str1 += self.moves_1_htm[i] + " "
            if i < len2:
                animations.append(CubeMove(self.cube2, moves2[i]))
                if i < len(self.moves_2_htm):
                    str2 += self.moves_2_htm[i] + " "

            if animations:
                self.play(*animations, run_time=0.5)

                self.remove(text1, text2)
                text1 = mn.Text(str1, font_size=36).to_corner(mn.DL).shift(mn.UP * 0.5 + mn.RIGHT * 0.5)
                text2 = mn.Text(str2, font_size=36, color=mn.YELLOW).to_corner(mn.DR).shift(mn.UP * 0.5 + mn.LEFT * 0.5)
                self.add_fixed_in_frame_mobjects(text1, text2)

                self.wait(delay)
        self.wait(2)
        solution_text.to_edge(mn.UP)
        self.add_fixed_in_frame_mobjects(solution_text)
        self.wait(3)

        # Restart cubes to initial states
        self.play(mn.FadeOut(self.cube1), mn.FadeOut(self.cube2))

        self.cube1 = RubiksCube().scale(0.5)
        self.cube1.move_to((mn.LEFT + mn.UP) * 3)
        self.setup_cube(self.cube1, self.initial_state_1)

        self.cube2 = RubiksCube().scale(0.5)
        self.cube2.move_to((mn.RIGHT + mn.DOWN) * 3)
        self.setup_cube(self.cube2, self.initial_state_2)

        self.play(mn.FadeIn(self.cube1), mn.FadeIn(self.cube2))
        text3.next_to(solution_text, mn.DOWN)
        self.add_fixed_in_frame_mobjects(text3)
        self.wait(2)
        # self.remove(solution_text)


        solution_moves = self.moves_1 + normalize_moves(htm_str_inverse(self.moves_2_htm))
        for move in solution_moves:
            self.play(CubeMove(self.cube1, move), run_time=delay)
        self.wait(1)