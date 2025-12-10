from typing import Required, TypedDict, Unpack

import numpy as np
from aorta_personalization.mesh.types import MeshInfo
from aorta_personalization.problem.types import ProblemParameters
from cheartpy.cl.struct import CLPartition
from pytools.arrays import A2
from pytools.logging.trait import ILogger


class _MakeReferenceDataKwargs(TypedDict, total=False):
    log: Required[ILogger]
    pedantic: bool
    cores: int


def make_reference_data_for_inverse_estimation[F: np.floating, I: np.integer](
    pb: ProblemParameters,
    mesh: MeshInfo,
    cl: A2[F],
    cl_part: CLPartition[F, I] | None,
    dl_part: CLPartition[F, I],
    **kwargs: Unpack[_MakeReferenceDataKwargs],
) -> None: ...
