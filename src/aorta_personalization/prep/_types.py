from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    from aorta_personalization.mesh.types import MeshInfo
    from aorta_personalization.problem.types import ProblemParameters
    from cheartpy.cl.struct import CLPartition
    from cheartpy.fe.p_file import PFile
    from pytools.result import Err, Ok


class PFileGenerator[F: np.floating, I: np.integer](Protocol):
    def __call__(
        self, prob: ProblemParameters, mesh: MeshInfo, *part: CLPartition[F, I] | None
    ) -> Ok[PFile] | Err: ...
