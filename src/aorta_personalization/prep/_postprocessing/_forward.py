from pathlib import Path
from typing import TypedDict, Unpack

import numpy as np
from cheartpy.io.api import chread_d, chwrite_d_utf
from cheartpy.search.api import get_var_index
from pytools.arrays import A2, DType
from pytools.result import Err, Ok


class _UPSKW(TypedDict, total=False):
    home: Path
    prefix: str


class _PostProcessPhysicalSpaceKwargs(TypedDict, total=False):
    home: Path
    space: str
    stiff: str
    cores: int


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
    prefix = kwargs.get("prefix", "Stiff")
    stiff = chread_d(f"{prefix}-{i}.D")
    chwrite_d_utf(f"{prefix}-{i}.D", stiff[:, [0]])


def postprocess_physical_space(
    ref_space: Path, disp: str, **kwargs: Unpack[_PostProcessPhysicalSpaceKwargs]
) -> Ok[None] | Err:
    """Post-process the physical space data after simulation.

    Returns
    -------
    None

    """
    x_i = chread_d(ref_space)
    home = kwargs.get("home", Path())
    files = (f.name for f in home.glob(disp))
    match get_var_index(files, disp):
        case Ok(items):
            pass
        case Err(e):
            return Err(e)
    for i in items:
        update_physical_space(x_i, disp, i, home=home, prefix=kwargs.get("space", "Space"))
        stripe_modulus_from_stiff_var(i, home=home, prefix=kwargs.get("stiff", "Stiff"))
    return Ok(None)
