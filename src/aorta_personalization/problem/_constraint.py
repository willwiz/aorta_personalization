from typing import TYPE_CHECKING, Literal, NamedTuple, TypedDict, Unpack

from aorta_personalization.mesh.types import Geometries, ProblemTopologies
from cheartpy.fe.physics.fs_coupling import ROT_CONS_CHOICE, create_rotation_constraint

if TYPE_CHECKING:
    from collections.abc import Mapping

    from aorta_personalization.solid.types import SolidProbVars
    from cheartpy.fe.physics.fs_coupling import FSCouplingProblem
    from cheartpy.fe.trait import IVariable
    from cheartpy.fe.trait._basic import ICheartTopology


class _Config(NamedTuple):
    pfx: str
    top: ICheartTopology
    cons: ROT_CONS_CHOICE


class _RBodyConsKwargs(TypedDict, total=False):
    inlet: str
    outlet: str
    motion: FSCouplingProblem | None


CARD_DIRECTIONS: set[Literal["x", "y", "z"]] = {"x", "y", "z"}


def _create_rigid_body_constraints_from_config(
    c: _Config, space: IVariable, disp: IVariable, sfx: str
) -> FSCouplingProblem:
    return create_rotation_constraint(
        f"P{c.pfx}{sfx}", c.top, c.cons, space=space, disp=disp, freq=-1
    )


def create_rigid_body_constraints(
    tops: ProblemTopologies,
    space: IVariable,
    svars: SolidProbVars,
    geo: Geometries,
    **kwargs: Unpack[_RBodyConsKwargs],
) -> dict[str, FSCouplingProblem]:
    _inlet = kwargs.get("inlet", "Inlet")
    _outlet = kwargs.get("outlet", "Outlet")
    _no_motion = kwargs.get("motion") is None
    surf_orientation: Mapping[str, set[Literal["x", "y", "z"]]] = {
        _inlet: {"x"},
        _outlet: {"z" if geo is Geometries.BENT_CYLINDER else "x"},
    }
    cons: Mapping[str, ROT_CONS_CHOICE] = {
        s: {"T": (CARD_DIRECTIONS - orient) if _no_motion else set(), "R": orient}
        for (s, orient) in surf_orientation.items()
    }
    pars: Mapping[str, _Config | None] = {
        _inlet: _Config(pfx=f"{svars.U}", top=tops.inlet, cons=cons[_inlet])
        if tops.inlet
        else None,
        _outlet: _Config(pfx=f"{svars.U}", top=tops.outlet, cons=cons[_outlet])
        if tops.outlet
        else None,
    }
    return {
        s: _create_rigid_body_constraints_from_config(v, space, svars.U, s)
        for s, v in pars.items()
        if v
    }
