from concurrent import futures
from typing import TYPE_CHECKING, TypedDict, Unpack

from cheartpy.io.api import chread_d, chwrite_d_utf
from cheartpy.search.api import get_var_index
from pytools.parallel import PEXEC_ARGS, parallel_exec
from pytools.result import Err, Ok

if TYPE_CHECKING:
    from pathlib import Path


class _MakeLongitudinalFieldKwargs(TypedDict, total=False):
    var: str
    prefix: str
    n_t: int
    cores: int


def make_longitudinal_field(
    root: Path, **kwargs: Unpack[_MakeLongitudinalFieldKwargs]
) -> Ok[None] | Err:
    field_name = kwargs.get("var", "CLField")
    prefix = kwargs.get("prefix", "CLz")
    cores = kwargs.get("cores", 1)
    n_t = kwargs.get("n_t", 100)
    if not (root / f"{field_name}-{n_t}.D").is_file():
        return Err(FileNotFoundError(f"{field_name}-{n_t}.D not found in {root}"))
    cl = chread_d(root / f"{field_name}-{n_t}.D")
    match get_var_index([f.name for f in root.glob(rf"{field_name}-*.D")], field_name):
        case Ok(idx):
            pass
        case Err(e):
            return Err(e)
    with futures.ProcessPoolExecutor(cores) as executor:
        args: PEXEC_ARGS = [([(root / f"{prefix}-{i}.D"), cl[:, [0]]], {}) for i in idx]
        parallel_exec(executor, chwrite_d_utf, args, prog_bar=True)
    return Ok(None)
