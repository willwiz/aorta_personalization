from typing import TYPE_CHECKING, TypedDict, Unpack

from cheartpy.fe.api import create_variable

from ._types import SolidProbVars

if TYPE_CHECKING:
    from aorta_personalization.mesh.types import ProblemTopologies
    from cheartpy.fe.trait import ICheartTopology
    from pytools.arrays import T3


class _CreateSolidVarsKwargs(TypedDict, total=False):
    freq: int
    pfx: T3[str]
    sfx: str


def create_solid_vars(
    top: ProblemTopologies,
    space_top: ICheartTopology,
    **kwargs: Unpack[_CreateSolidVarsKwargs],
) -> SolidProbVars:
    # unpack inputs
    _freq = kwargs.get("freq", 1)
    _px, _pu, _pp = kwargs.get("pfx", ("Space", "Disp", "Pres"))
    _sfx = kwargs.get("sfx", "")
    # code
    space = create_variable(f"{_px}{_sfx}", space_top, 3, data=top.U.mesh, freq=_freq)
    disp = create_variable(f"{_pu}{_sfx}", space_top, 3, freq=_freq)
    pres = create_variable(f"{_pp}{_sfx}", top.P, 1, freq=_freq)
    return SolidProbVars(space, disp, pres)
