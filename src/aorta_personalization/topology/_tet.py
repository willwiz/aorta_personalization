from typing import Unpack

from aorta_personalization.mesh.types import MeshInfo
from pytools.result import Err, Ok

from ._types import ProblemInterfaces, ProblemTopologies, TopologyKwargs


def create_tetrahedron_topologies(
    mesh: MeshInfo,
    **kwargs: Unpack[TopologyKwargs],
) -> Ok[tuple[ProblemTopologies, ProblemInterfaces]] | Err: ...
