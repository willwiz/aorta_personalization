import dataclasses as dc


@dc.dataclass(slots=True)
class ProblemVariableNames:
    noise: str
    xi: str
    x0: str
    xt: str
    u0: str
    ut: str
    p0: str
    pt: str
    dm: str
    data: str
