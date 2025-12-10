from pathlib import Path
from typing import TYPE_CHECKING

from aorta_personalization.mesh._types import MeshTuple
from cheartpy.io.api import chread_d, chwrite_d_utf
from cheartpy.mesh.api import import_cheart_mesh
from pytools.path import clear_dir
from pytools.result import Err, Ok

if TYPE_CHECKING:
    import numpy as np
    from aorta_personalization.mesh._types import MeshInfo
    from pytools.logging.trait import ILogger


_MRI_DIR = Path("DATA_AORTA")


def setup_aorta_mesh(
    mesh: MeshInfo, step: int = 2, *, log: ILogger
) -> MeshTuple[np.float64, np.intc]:
    log.debug(f"Reading tracked MRI data from {_MRI_DIR}")
    match import_cheart_mesh(_MRI_DIR / "model_Tracked_forward"):
        case Ok(disp_mesh):
            mesh.DIR.mkdir(exist_ok=True)
        case Err(e):
            raise e
    log.debug(f"Updating space from {_MRI_DIR / f'TrackedSpace-{step}.D'}")
    disp_mesh.space.v = chread_d(_MRI_DIR / f"TrackedSpace-{step}.D")
    log.debug(f"Reading CL field from {_MRI_DIR / 'CenterLineField-0.D'}")
    cl = chread_d(_MRI_DIR / "CenterLineField-0.D")
    log.debug(f"Reading normals from {_MRI_DIR / 'CenterNormalField-0.D'}")
    normal = chread_d(_MRI_DIR / "CenterNormalField-0.D")
    log.debug(f"Mesh will be saved to {mesh.DIR}")
    clear_dir(mesh.DIR)
    disp_mesh.save(mesh.DIR / mesh.DISP)
    chwrite_d_utf((mesh.DIR / mesh.FIELD), cl)
    chwrite_d_utf((mesh.DIR / mesh.NORMAL), normal)
    return MeshTuple(disp_mesh, cl)
