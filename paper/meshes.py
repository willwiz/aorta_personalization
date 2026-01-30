# /// script
# require-python = ">=3.14"
# dependencies = [
#     "aorta_personalization",
# ]
# ///

from pathlib import Path

from aorta_personalization.mesh._types import BCPatchTag
from aorta_personalization.mesh.types import CylinderDims, ElementTypes, Geometries, MeshInfo

_DEFAULT_CYLINDER = CylinderDims(shape=(9.0, 12.0, 200.0), nelem=(2, 16, 64))

STRAIGHT_CYLINDER_QUAD_MESH = MeshInfo(
    GEO=Geometries["STRAIGHT_CYLINDER"],
    DIR=Path("mesh_straight_cylinder"),
    SPEC=_DEFAULT_CYLINDER,
    DISP="cyl_quad",
    PRES="cyl_lin",
    ELEM=ElementTypes["HEX"],
    ORDER=2,
    FIELD="CenterLineField-0.D",
    NORMAL="CenterNormalField-0.D",
    INLET=BCPatchTag("inlet", 1),
    OUTLET=BCPatchTag("outlet", 2),
    INNER=BCPatchTag("inner", 3),
    OUTER=BCPatchTag("outer", 4),
    ENDS=[1, 2],
)


BENT_CYLINDER_QUAD_MESH = MeshInfo(
    GEO=Geometries["BENT_CYLINDER"],
    DIR=Path("mesh_bent_cylinder"),
    SPEC=_DEFAULT_CYLINDER,
    DISP="cyl_quad",
    PRES="cyl_lin",
    ELEM=ElementTypes["HEX"],
    ORDER=2,
    FIELD="CenterLineField-0.D",
    NORMAL="CenterNormalField-0.D",
    INLET=BCPatchTag("inlet", 1),
    OUTLET=BCPatchTag("outlet", 2),
    INNER=BCPatchTag("inner", 3),
    OUTER=BCPatchTag("outer", 4),
    ENDS=[1, 2],
)
BULGE_CYLINDER_QUAD_MESH = MeshInfo(
    GEO=Geometries["BENT_CYLINDER"],
    DIR=Path("mesh_bulge_cylinder"),
    SPEC=_DEFAULT_CYLINDER,
    DISP="cyl_quad",
    PRES="cyl_lin",
    ELEM=ElementTypes["HEX"],
    ORDER=2,
    FIELD="CenterLineField-0.D",
    NORMAL="CenterNormalField-0.D",
    INLET=BCPatchTag("inlet", 1),
    OUTLET=BCPatchTag("outlet", 2),
    INNER=BCPatchTag("inner", 3),
    OUTER=BCPatchTag("outer", 4),
    ENDS=[1, 2],
)
