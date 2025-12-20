from typing import TYPE_CHECKING, NamedTuple, overload

import numpy as np
from cheartpy.cl.cl_expressions import ll_str
from cheartpy.cl.struct import CLPartition, CLStructure
from cheartpy.fe.api import (
    create_basis,
    create_expr,
    create_top_interface,
    create_topology,
    create_variable,
)
from pytools.result import Err, Ok

if TYPE_CHECKING:
    from aorta_personalization.mesh.types import MeshInfo, ProblemTopologies
    from cheartpy.fe.trait import ITopInterface, IVariable


class _CLTopRVar[T: (CLStructure, None)](NamedTuple):
    struct: T
    interfaces: list[ITopInterface]


@overload
def create_centerline_topology_list[F: np.floating, I: np.integer](
    mesh: MeshInfo, tops: ProblemTopologies, part: None, field: IVariable
) -> Ok[_CLTopRVar[None]] | Err: ...
@overload
def create_centerline_topology_list[F: np.floating, I: np.integer](
    mesh: MeshInfo, tops: ProblemTopologies, part: CLPartition[F, I], field: IVariable
) -> Ok[_CLTopRVar[CLStructure]] | Err: ...
def create_centerline_topology_list[F: np.floating, I: np.integer](
    mesh: MeshInfo, tops: ProblemTopologies, part: CLPartition[F, I] | None, field: IVariable
) -> Ok[_CLTopRVar[CLStructure]] | Ok[_CLTopRVar[None]] | Err:
    if part is None:
        return Ok(_CLTopRVar(None, []))
    basis = tops.inner.get_basis()
    if basis is None:
        msg = "Centerline basis not found in topology"
        return Err(ValueError(msg))
    cl_top = create_topology(
        f"TP{part.prefix}Az{basis.order}",
        basis,
        (mesh.DIR / f"{part.prefix}Az{basis.order}"),
    )
    lm_basis = create_basis(basis.elem, basis.basis.kind, basis.quadrature.kind, 0, basis.gp)
    lm_top = create_topology(
        f"TP{part.prefix}Az{'L'}", lm_basis, mesh.DIR / f"{part.prefix}Az{'L'}"
    )
    lm_top.discontinuous = True
    interfaces: list[ITopInterface] = [
        create_top_interface("OneToOne", [cl_top, lm_top]),
        create_top_interface(
            "ManyToOne",
            [cl_top],
            tops.U,
            (mesh.DIR / f"interface-{part.prefix}Az.IN"),
            part.in_surf,
        ),
    ]
    support_var = create_variable(
        f"{part.prefix}Support",
        lm_top,
        3,
        data=mesh.DIR / f"{part.prefix}Az{'L'}V_Support.INIT",
        freq=-1,
    )
    elem = create_variable(
        f"{part.prefix}Elem",
        lm_top,
        part.nn,
        data=(mesh.DIR / f"{part.prefix}Az{'L'}V_Elem.INIT"),
        freq=-1,
    )
    basis = create_expr(f"{part}_basis", [ll_str(field, support_var)])
    basis.add_deps(field, support_var)
    b_vec = create_expr(f"{part}v", [ll_str(field, v) for v in part.support])
    b_vec.add_deps(field, support_var)
    struct = CLStructure(
        part.prefix, part.in_surf, part.nn, part.ne, cl_top, lm_top, support_var, elem, basis, b_vec
    )
    return Ok(_CLTopRVar(struct, interfaces))
