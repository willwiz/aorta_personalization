from typing import TYPE_CHECKING, NamedTuple

from aorta_personalization.mesh.types import ProblemTopologies
from cheartpy.fe.api import create_basis, create_top_interface, create_topology

from ._types import ElementTypes, MeshInfo

if TYPE_CHECKING:
    from collections.abc import Sequence

    from cheartpy.fe.aliases import CHEART_ELEMENT_TYPE, CHEART_QUADRATURE_TYPE
    from cheartpy.fe.trait import ITopInterface


class _IntegrationScheme(NamedTuple):
    elem: CHEART_ELEMENT_TYPE
    surf: CHEART_ELEMENT_TYPE
    method: CHEART_QUADRATURE_TYPE
    np: int


_ALLOWED_ORDERS = (1, 2)


_MESH_INTEGRATION_SCHEMES: dict[ElementTypes, _IntegrationScheme] = {
    ElementTypes.HEX: _IntegrationScheme("HEXAHEDRAL_ELEMENT", "QUADRILATERAL_ELEMENT", "GL", 9),
    ElementTypes.TET: _IntegrationScheme("TETRAHEDRAL_ELEMENT", "TRIANGLE_ELEMENT", "KL", 4),
}


def create_topology_list(
    mesh: MeshInfo,
) -> tuple[ProblemTopologies, Sequence[ITopInterface]]:
    (body_elem, surf_elem, method, n_gp) = _MESH_INTEGRATION_SCHEMES[mesh.ELEM]
    basis_vol = {i: create_basis(body_elem, "NL", method, i, n_gp) for i in _ALLOWED_ORDERS}
    basis_surf = {i: create_basis(surf_elem, "NL", method, i, n_gp) for i in _ALLOWED_ORDERS}
    vol_lin = create_topology("TPLin", basis_vol[1], mesh.DIR / mesh.PRES)
    if mesh.ORDER == 1:
        vol_body = vol_lin
    else:
        vol_body = create_topology("TPQuad", basis_vol[mesh.ORDER], mesh.DIR / mesh.DISP)
    surfaces = (mesh.INNER, mesh.OUTER, mesh.INLET, mesh.OUTLET)
    surf = {
        lbl: create_topology(
            f"TPSurf{lbl}", basis_surf[mesh.ORDER], mesh.DIR / f"{mesh.DISP}_{lbl}"
        )
        for lbl, _ in surfaces
    }
    for lbl, i in surfaces:
        surf[lbl].create_in_boundary(vol_body, i)
    interfaces = [
        create_top_interface(
            "ManyToOne", [surf[lbl]], vol_body, mesh.DIR / f"interface-{lbl}.IN", idx
        )
        for lbl, idx in surfaces
    ]
    if mesh.ORDER > 1:
        interfaces = [*interfaces, create_top_interface("OneToOne", [vol_lin, vol_body])]
    return (
        ProblemTopologies(
            vol_body,
            vol_lin,
            vol_body,
            inner=surf[mesh.INNER.name],
            outer=surf[mesh.OUTER.name],
            inlet=surf[mesh.INLET.name],
            outlet=surf[mesh.OUTLET.name],
        ),
        interfaces,
    )
