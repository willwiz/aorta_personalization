import dataclasses as dc
from collections.abc import Sequence
from typing import Literal, TypedDict

from cheartpy.fe.trait.basic import ICheartTopology

# symbols follow continuum-mechanics notation


class TopologyKwargs(TypedDict, total=False):
    scheme: Literal["p1", "p2p1"]


@dc.dataclass(slots=True, frozen=True)
class ProblemTopologies:
    U: ICheartTopology
    P: ICheartTopology
    X: ICheartTopology
    inner: ICheartTopology
    outer: ICheartTopology
    inlet: ICheartTopology | None
    outlet: ICheartTopology | None


type ProblemInterfaces = Sequence[ICheartTopology]
