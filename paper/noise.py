# /// script
# require-python = ">=3.14"
# dependencies = [
#     "numpy",
#     "cheartpy",
#     "aorta_personalization"
# ]
# ///

from typing import TYPE_CHECKING, Unpack

from inverse import main_reverse
from meshes import BENT_CYLINDER_QUAD_MESH, BULGE_CYLINDER_QUAD_MESH, STRAIGHT_CYLINDER_QUAD_MESH
from problems import (
    PROBS_NOISE_BENT,
    PROBS_NOISE_BULGE,
    PROBS_NOISE_STRAIGHT,
)

if TYPE_CHECKING:
    from forward import MainSimKwargs


def main_cli(**kwargs: Unpack[MainSimKwargs]) -> None:
    for p in PROBS_NOISE_STRAIGHT:
        main_reverse(p, STRAIGHT_CYLINDER_QUAD_MESH, **kwargs)
    for p in PROBS_NOISE_BENT:
        main_reverse(p, BENT_CYLINDER_QUAD_MESH, **kwargs)
    for p in PROBS_NOISE_BULGE:
        main_reverse(p, BULGE_CYLINDER_QUAD_MESH, **kwargs)


def main_select(**kwargs: Unpack[MainSimKwargs]) -> None:
    # for p in PROBS_NOISE_STRAIGHT:
    #     main_reverse(p, STRAIGHT_CYLINDER_QUAD_MESH, **kwargs)
    # for p in PROBS_NOISE_BENT:
    #     main_reverse(p, BENT_CYLINDER_QUAD_MESH, **kwargs)
    for p in PROBS_NOISE_BULGE:
        main_reverse(p, BENT_CYLINDER_QUAD_MESH, **kwargs)


def main_test(**kwargs: Unpack[MainSimKwargs]) -> None:
    main_reverse(PROBS_NOISE_BENT[0], BENT_CYLINDER_QUAD_MESH, **kwargs)


if __name__ == "__main__":
    # main_cli(cores=16, prog_bar=True)
    main_select(cores=16, prog_bar=True)
    # main_test(cores=16, prog_bar=True)
