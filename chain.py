"""Turns an edit (name + parameters) into a matrix, and a chain of edits into one matrix."""
from math_engine.transforms import (about_point, compose, flip_horizontal,
                                    flip_vertical, rotation, scaling, shear)

STEP_NAMES = ["Rotate", "Flip horizontal", "Flip vertical", "Shear", "Scale"]

_BUILDERS = {
    "Rotate": lambda p: rotation(p["angle"]),
    "Flip horizontal": lambda p: flip_horizontal(),
    "Flip vertical": lambda p: flip_vertical(),
    "Shear": lambda p: shear(p.get("kx", 0.0), p.get("ky", 0.0)),
    "Scale": lambda p: scaling(p["sx"], p["sy"]),
}


def step_matrix(name, params, width, height):
    """3x3 matrix of one edit, applied about the centre of the image."""
    cx, cy = (width - 1) / 2, (height - 1) / 2
    return about_point(_BUILDERS[name](params), cx, cy)


def chain_matrix(chain, width, height):
    """Combined matrix of [(name, params), ...] listed in the order applied."""
    return compose([step_matrix(n, p, width, height) for n, p in chain])
