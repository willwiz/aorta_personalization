import numpy as np
from aorta_personalization.mesh.types import MeshInfo
from aorta_personalization.topology.types import ProblemTopologies
from cheartpy.cl.struct import CLPartition, CLTopology
from cheartpy.fe.trait import ITopInterface, IVariable


def create_centerline_topology_list[F: np.floating, I: np.integer](
    mesh: MeshInfo, tops: ProblemTopologies, part: CLPartition[F, I], field: IVariable
) -> tuple[CLTopology | None, list[ITopInterface]]: ...
