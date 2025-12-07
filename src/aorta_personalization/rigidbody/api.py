from aorta_personalization.mesh.types import Geometries
from aorta_personalization.solid.types import SolidProbVars
from aorta_personalization.topology.types import ProblemTopologies
from cheartpy.fe.physics.fs_coupling.struct import FSCouplingProblem
from cheartpy.fe.trait import IVariable


def create_rigid_body_constraints(
    tops: ProblemTopologies,
    space: IVariable,
    svars: SolidProbVars,
    geo: Geometries,
    motion_prob: FSCouplingProblem | None,
) -> dict[str, FSCouplingProblem]: ...
