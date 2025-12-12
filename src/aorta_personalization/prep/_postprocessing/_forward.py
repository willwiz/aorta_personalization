from concurrent import futures
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict, Unpack

import numpy as np
from cheartpy.io.api import chread_d, chwrite_d_utf
from cheartpy.search.api import get_var_index
from pytools.parallel import PEXEC_ARGS, parallel_exec
from pytools.result import Err, Ok

if TYPE_CHECKING:
    from pytools.arrays import A2, DType


class _UPSKW(TypedDict, total=False):
    home: Path
    prefix: str


class _PostProcessPhysicalSpaceKwargs(TypedDict, total=False):
    home: Path
    space: str
    stiff: str
    cores: int
    prog_bar: bool


def update_physical_space[F: np.floating](
    ref: A2[F] | str, disp: str, i: int, *, dtype: DType[F] = np.float64, **kwargs: Unpack[_UPSKW]
) -> None:
    data_dir = kwargs.get("home", Path())
    space = kwargs.get("prefix", "Space")
    match ref:
        case np.ndarray():
            dtype = ref.dtype
        case str():
            ref = chread_d(ref, dtype=dtype)
    cur = chread_d((data_dir / f"{disp}-{i}.D"), dtype=dtype)
    chwrite_d_utf((data_dir / f"{space}-{i}.D"), cur + ref)


def stripe_modulus_from_stiff_var(i: int, **kwargs: Unpack[_UPSKW]) -> None:
    root = kwargs.get("home", Path())
    prefix = kwargs.get("prefix", "Stiff")
    stiff = chread_d(root / f"{prefix}-{i}.D")
    chwrite_d_utf(root / f"{prefix}-{i}.D", stiff[:, [0]])


def postprocess_physical_space(
    ref_space: Path, disp: str, **kwargs: Unpack[_PostProcessPhysicalSpaceKwargs]
) -> Ok[None] | Err:
    """Post-process the physical space data after simulation.

    Returns
    -------
    None

    """
    _bar = kwargs.get("prog_bar", False)
    x_i = chread_d(ref_space)
    home = kwargs.get("home", Path())
    files = (f.name for f in home.glob(f"{disp}-*.D"))
    if not any(files):
        msg = f"No variable files found for displacement variable '{disp}' in directory '{home}'."
        return Err(ValueError(msg))
    match get_var_index(files, disp):
        case Ok(items):
            pass
        case Err(e):
            return Err(e)
    phys_args: PEXEC_ARGS = [
        ([x_i, disp, i], {"home": home, "space": kwargs.get("space", "Space")}) for i in items
    ]
    stiff_args: PEXEC_ARGS = [
        ([i], {"home": home, "prefix": kwargs.get("stiff", "Stiff")}) for i in items
    ]
    with futures.ProcessPoolExecutor(max_workers=kwargs.get("cores", 1)) as exe:
        parallel_exec(exe, update_physical_space, phys_args, prog_bar=_bar)
        parallel_exec(exe, stripe_modulus_from_stiff_var, stiff_args, prog_bar=_bar)
    return Ok(None)
