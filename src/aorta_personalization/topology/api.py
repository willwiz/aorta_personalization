__all__ = ["create_topology_list"]


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from aorta_personalization.mesh.types import MeshInfo
    from cheartpy.fe.trait import ITopInterface

    from ._types import ProblemTopologies


def create_topology_list(
    mesh: MeshInfo,
) -> tuple[ProblemTopologies, Sequence[ITopInterface]]: ...
