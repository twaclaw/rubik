from dataclasses import dataclass


@dataclass
class Constants:
    N_PERM_4: int = 24
    N_CHOOSE_8_4: int = 70
    N_MOVE: int = 18  # number of possible face moves
    N_TWIST: int = 2187  # 3^7 possible corner orientations in phase 1
    N_FLIP: int = 2048  # 2^11 possible edge orientations in phase 1
    N_SLICE_SORTED: int = 11880  # 12*11*10*9 possible positions of the FR, FL, BL, BR edges in phase 1
    N_SLICE: int =  495 # N_SLICE_SORTED // N_PERM_4  # we ignore the permutation of FR, FL, BL, BR in phase 1
    N_FLIPSLICE_CLASS: int = 64430  # number of equivalence classes for combined flip+slice concerning symmetry group D4h
    N_U_EDGES_PHASE2: int = 1680  # number of different positions of the edges UR, UF, UL and UB in phase 2
    N_CORNERS: int = 40320  # 8! corner permutations in phase 2
    N_CORNERS_CLASS: int = 2768  # number of equivalence classes concerning symmetry group D4h
    N_UD_EDGES: int = 40320  # 8! permutations of the edges in the U-face and D-face in phase 2
    N_SYM: int = 48  # number of cube symmetries of full group Oh
    N_SYM_D4h: int = 16  # Number of symmetries of subgroup D4h
    FOLDER: str = "twophase"  # Folder name for generated tables
    PROGRESS_BAR_DESC_WIDTH: int = 40  # Width of the description in the progress bar

