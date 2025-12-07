from aorta_personalization.mesh._types import MeshInfo
from aorta_personalization.solid.types import SolidProbVars
from cheartpy.fe.trait import IBCPatch, IVariable


def get_boundary_conditions_list(
    mesh: MeshInfo, svars: SolidProbVars, motion: IVariable | None = None
) -> list[IBCPatch]: ...
