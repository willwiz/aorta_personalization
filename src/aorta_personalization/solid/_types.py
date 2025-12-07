import dataclasses as dc
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cheartpy.fe.trait.basic import IVariable


@dc.dataclass(slots=True)
class SolidProbVars:
    X: IVariable
    U: IVariable
    P: IVariable
