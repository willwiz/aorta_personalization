from typing import TYPE_CHECKING, TypedDict, Unpack

from cheartpy.io.api import chread_d, chwrite_d_utf

if TYPE_CHECKING:
    from pathlib import Path

    from aorta_personalization.problem.types import Labels


def check_for_vars(root: Path, *vs: str, max_idx: int = 100) -> list[str]:
    return [v for v in vs if (root / f"{v}-{max_idx}.D").exists()]


class _AddWriteVarKwargs(TypedDict, total=False):
    disp_i: str
    disp_t: str
    disp: str


def addwrite_var(lbl: Labels, i: int, **prefix: Unpack[_AddWriteVarKwargs]) -> None:
    disp_i = prefix.get("disp_i", "U0")
    disp_t = prefix.get("disp_t", "Ut")
    disp = prefix.get("disp", "Disp")
    cur = chread_d(lbl.D / f"{disp_t}-{i}.D")
    ref = chread_d(lbl.D / f"{disp_i}-{i}.D")
    chwrite_d_utf((lbl.D / f"{disp}-{i}.D"), cur - ref)
