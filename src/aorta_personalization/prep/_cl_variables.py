from pathlib import Path
from typing import TYPE_CHECKING, TypedDict, Unpack

import numpy as np
from cheartpy.cl.api import ll_interp
from cheartpy.io.api import chread_d, chwrite_d_utf
from cheartpy.search.api import get_var_index
from pytools.result import Err, Ok

if TYPE_CHECKING:
    from cheartpy.cl.struct import CLPartition
    from pytools.arrays import A1, A2


def expand_cl_variable_to_main_topology[F: np.floating, I: np.integer](
    part: CLPartition[F, I],
    cl: A1[F],
    prefix: str,
    step: int,
    *,
    root_dir: Path,
) -> None:
    lms = chread_d(root_dir / f"{part.prefix}{prefix}-{step}.D", dtype=cl.dtype)
    if lms.shape[0] == 1 and lms.shape[1] == part.nn:
        lms = lms.reshape(-1, 1)
    res = ll_interp(part, lms, cl)
    chwrite_d_utf((root_dir / f"{prefix}-{step}.D"), res)


class _CLVarExpandKwargs(TypedDict, total=False):
    root_dir: Path


def expand_cl_variables_to_main_topology[F: np.floating, I: np.integer](
    part: CLPartition[F, I] | None, cl: A2[F], *variables: str, **kwargs: Unpack[_CLVarExpandKwargs]
) -> Ok[list[str]] | Err:
    if part is None:
        return Ok([])
    root_dir: Path = kwargs.get("root_dir", Path())
    files = (f.name for f in root_dir.glob(f"{part.prefix}{variables[0]}-*.D"))
    match get_var_index(files, f"{part.prefix}{variables[0]}"):
        case Ok(items):
            pass
        case Err(e):
            return Err(e)
    if len(items) == 0:
        msg = f"No data files found for variable(s) {variables} with prefix {part.prefix}"
        return Err(FileNotFoundError(msg))
    for v in variables:
        for i in items:
            expand_cl_variable_to_main_topology(part, cl[:, 0], v, i, root_dir=root_dir)
    return Ok(list(variables))
