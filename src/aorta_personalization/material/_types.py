import dataclasses as dc
from typing import Literal

STIFF_MODES = Literal["const", "grad", "sine", "circ"]


@dc.dataclass(slots=True, frozen=True)
class MaterialProperty:
    mode: STIFF_MODES
    lower: float
    upper: float
