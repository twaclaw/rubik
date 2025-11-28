"""
This code is a subset modified from this package: https://manim-rubikscube.readthedocs.io/en/stable/
at https://github.com/WampyCakes/manim-rubikscube
"""

import numpy as np
from manim import (
    BLACK,
    DEGREES,
    DOWN,
    IN,
    LEFT,
    ORIGIN,
    OUT,
    PI,
    RIGHT,
    UL,
    UP,
    WHITE,
    X_AXIS,
    Y_AXIS,
    Z_AXIS,
    Animation,
    Mobject,
    Square,
    Text,
    ThreeDScene,
    VGroup,
    VMobject,
)
from manim.utils.space_ops import z_to_vector

# --- Utils ---


def get_axis_from_face(face):
    if face == "F" or face == "B":
        return X_AXIS
    elif face == "U" or face == "D":
        return Z_AXIS
    else:
        return Y_AXIS


def get_faces_of_cubie(dim, position):
    dim = dim - 1
    try:
        faces = {
            # Front corners
            (0, 0, 0): [LEFT, DOWN, IN],
            (0, 0, dim): [LEFT, DOWN, OUT],
            (0, dim, 0): [LEFT, UP, IN],
            (0, dim, dim): [LEFT, UP, OUT],
            # Back corners
            (dim, 0, 0): [RIGHT, DOWN, IN],
            (dim, 0, dim): [RIGHT, DOWN, OUT],
            (dim, dim, 0): [RIGHT, UP, IN],
            (dim, dim, dim): [RIGHT, UP, OUT],
        }
        return faces[position]
    except Exception:
        x = position[0]
        y = position[1]
        z = position[2]

        if x == 0:
            if y == 0:
                return [DOWN, LEFT]
            elif y == dim:
                return [UP, LEFT]
            else:
                if z == 0:
                    return [IN, LEFT]
                elif z == dim:
                    return [OUT, LEFT]
                else:
                    return [LEFT]
        elif x == dim:
            if y == 0:
                return [DOWN, RIGHT]
            elif y == dim:
                return [UP, RIGHT]
            else:
                if z == 0:
                    return [IN, RIGHT]
                elif z == dim:
                    return [OUT, RIGHT]
                else:
                    return [RIGHT]
        else:
            if y == 0:
                if z == 0:
                    return [IN, DOWN]
                elif z == dim:
                    return [OUT, DOWN]
                else:
                    return [DOWN]
            elif y == dim:
                if z == 0:
                    return [IN, UP]
                elif z == dim:
                    return [OUT, UP]
                else:
                    return [UP]
            else:
                if z == 0:
                    return [IN]
                elif z == dim:
                    return [OUT]
                else:
                    return []


# --- Classes ---


class Cubie(VGroup):
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
        for vect in OUT, DOWN, LEFT, IN, UP, RIGHT:
            face = Square(side_length=2, shade_in_3d=True, stroke_width=3)
            if vect.tolist() in faces:
                face.set_fill(self.colors[i], 1)
            else:
                face.set_fill(BLACK, 1)

            face.flip()
            face.shift(2 * OUT / 2.0)
            face.apply_matrix(z_to_vector(vect))

            self.faces[tuple(vect)] = face
            self.add(face)
            i += 1

    def get_face(self, face):
        if face == "F":
            return self.faces[tuple(LEFT)]
        elif face == "B":
            return self.faces[tuple(RIGHT)]
        elif face == "R":
            return self.faces[tuple(DOWN)]
        elif face == "L":
            return self.faces[tuple(UP)]
        elif face == "U":
            return self.faces[tuple(OUT)]
        elif face == "D":
            return self.faces[tuple(IN)]


class RubiksCube(VMobject):
    def __init__(
        self,
        dim=3,
        colors=[WHITE, "#C41E3A", "#009E60", "#FFD500", "#FF5800", "#0051BA"], #TODO: check
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
        self.x_offset = [[Mobject.shift, [x_offset, 0, 0]]]
        self.y_offset = [[Mobject.shift, [0, y_offset, 0]]]
        self.z_offset = [[Mobject.shift, [0, 0, z_offset]]]
        self.indices = {}

        self.cubies = np.ndarray((dim, dim, dim), dtype=Cubie)
        self.generate_cubies()
        self.move_to(ORIGIN)
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
        angle = PI / 2 if ("R" in move or "F" in move or "D" in move) else -PI / 2
        angle = angle if "2" not in move else angle * 2
        angle = -angle if "'" in move else angle

        VGroup(*self.get_face(face)).rotate(angle, axis)
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


class CubeMove(Animation):
    def __init__(self, mobject, face, **kwargs):
        self.axis = get_axis_from_face(face[0])
        self.face = face
        self.angle = PI / 2 if ("R" in face or "F" in face or "D" in face) else -PI / 2
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

        VGroup(*self.mobject.get_face(self.face[0])).rotate(
            alpha * self.angle, self.axis
        )

    def finish(self):
        super().finish()
        self.mobject.adjust_indices(self.mobject.get_face(self.face[0], False))


class RubiksCubeAnimation(ThreeDScene):
    def __init__(self, moves_ph1, moves_ph2, initial_state, **kwargs):
        self.moves_ph1 = moves_ph1
        self.moves_ph2 = moves_ph2
        self.initial_state = initial_state
        super().__init__(**kwargs)

    def rotate(self, delay: int = 10, rate: float = 0.5):
        self.begin_ambient_camera_rotation(rate=rate)
        self.wait(delay)
        self.stop_ambient_camera_rotation()
        self.move_camera(phi=60 * DEGREES, theta=-45 * DEGREES)


    def construct(self):
        self.cube = RubiksCube().scale(0.6)
        self.cube.move_to(ORIGIN)

        if self.initial_state:
            if isinstance(self.initial_state, str) and len(self.initial_state) == 54 and " " not in self.initial_state:
                self.cube.set_state(self.initial_state)
            else:
                self.cube.perform_sequence(self.initial_state)

        self.add(self.cube)
        self.set_camera_orientation(phi=60 * DEGREES, theta=-45 * DEGREES)

        label1 = Text(f"Initial stage: {self.initial_state}", font_size=24).to_corner(UL)
        self.add_fixed_in_frame_mobjects(label1)
        self.rotate(delay=6, rate=0.5)
        self.remove(label1)

        if self.moves_ph1:
            label2 = Text(f"Applying phase 1 moves: {''.join(self.moves_ph1)}", font_size=24).to_corner(UL)
            self.add_fixed_in_frame_mobjects(label2)
            self.wait(3)
            self.animate_sequence(self.moves_ph1)
            self.remove(label2)

        self.rotate(delay=6, rate=0.5)
        if self.moves_ph2:
            label3 = Text(f"Applying phase 2 moves: {''.join(self.moves_ph2)}", font_size=24).to_corner(UL)
            self.add_fixed_in_frame_mobjects(label3)
            self.wait(3)
            self.animate_sequence(self.moves_ph2)
            self.remove(label3)

        self.rotate(delay=6, rate=0.5)


    def animate_sequence(self, moves):
        for move in moves:
            self.play(CubeMove(self.cube, move), run_time=0.5)
            self.wait(0.1)


class RubiksCubeStatic(ThreeDScene):
    def construct(self):
        self.cube = RubiksCube().scale(0.6)
        self.add(self.cube)
        self.set_camera_orientation(phi=60 * DEGREES, theta=-45 * DEGREES)

        # Apply moves instantly without animation
        moves = ["R", "U", "R'", "U'"]
        for move in moves:
            anim = CubeMove(self.cube, move)
            VGroup(*self.cube.get_face(anim.face[0])).rotate(anim.angle, anim.axis)
            self.cube.adjust_indices(self.cube.get_face(anim.face[0], False))

        # Manim will automatically save the last frame as an image if run with -s
