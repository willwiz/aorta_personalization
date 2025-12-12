from pathlib import Path
from typing import TYPE_CHECKING, Required, TypedDict, Unpack

import numpy as np
from cheartpy.fe.cmd import run_prep, run_problem
from cheartpy.paraview.api import cheart2vtu_api
from pytools.path import clear_dir

if TYPE_CHECKING:
    from aorta_personalization.mesh.types import MeshInfo
    from aorta_personalization.problem.types import ProblemParameters
    from cheartpy.cl.struct import CLPartition
    from pytools.logging.api import ILogger

    from ._types import PFileGenerator


class _RunnerKwargs(TypedDict, total=False):
    log: Required[ILogger]
    pedantic: bool
    cores: int


def run_simulation[F: np.floating, I: np.integer](
    pfile_call: PFileGenerator[F, I],
    pb: ProblemParameters,
    mesh: MeshInfo,
    *parts: CLPartition[F, I] | None,
    **kwargs: Unpack[_RunnerKwargs],
) -> None:
    log = kwargs.get("log")
    pedantic = kwargs.get("pedantic", False)
    cores = kwargs.get("cores", 16)
    prob_name, prob_log = [pb.P.N + ext for ext in [".P", ".log"]]
    log.info(f"Starting {prob_name}")
    log.info("Making P-file")
    pfile = pfile_call(pb, mesh, *parts).unwrap()
    with Path(prob_name).open("w") as f:
        pfile.write(f)
    log.info(f"{prob_name} is written to file")
    clear_dir(mesh.DIR, "PART", "IN")
    run_prep(prob_name)
    log.info(f"Running Cheart ({prob_log}):")
    log.info(f"Results are saved to {pfile.output_dir}:")
    err = run_problem(prob_name, pedantic=pedantic, cores=cores, log=prob_log)
    log.info(f"Simulation exited with error {err}")
    if err > 0:
        raise RuntimeError


def cheart2vtu_cmdline_args(
    mesh: MeshInfo, pb: ProblemParameters, cores: int, out_dir: str
) -> list[str]:
    return [
        "find",
        "--mesh", str(mesh.DIR / mesh.DISP),
        "-f", str(pb.P.D),
        "-o", out_dir,
        "-c", str(cores),
    ]  # fmt: skip


def run_vtu(
    mesh: MeshInfo,
    pb: ProblemParameters,
    *vs: str,
    space: str | None = None,
    cores: int = 4,
) -> None:
    cheart2vtu_api(
        prefix="res",
        mesh=str(mesh.DIR / mesh.DISP),
        space=space,
        input_dir=str(pb.P.D),
        output_dir=str(pb.P.D),
        cores=cores,
        variables=vs,
    )
