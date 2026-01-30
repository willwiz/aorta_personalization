# /// script
# require-python = ">=3.14"
# dependencies = [
#     "numpy",
#     "pytools",
#     "cheartpy",
# ]
# ///
# pyright: reportUnknownMemberType=false
from pathlib import Path
from typing import TYPE_CHECKING, Literal, NamedTuple, Required, TypedDict, Unpack

import numpy as np
from cheartpy.cl.mesh import (
    create_cl_partition,
)
from cheartpy.io.api import chread_d
from meshes import BENT_CYLINDER_QUAD_MESH, STRAIGHT_CYLINDER_QUAD_MESH
from pytools.logging import ILogger, get_logger
from pytools.plotting.api import close_figure, create_figure, style_kwargs, update_figure_setting
from pytools.plotting.trait import PlotKwargs
from pytools.result import Err, Ok, all_ok

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from aorta_personalization.mesh._types import MeshInfo
    from pytools.arrays import A1, A2, DType


_3D = 3

_COLOR_MAP = {"U0": "r", "Ut": "g", "Stiff": "k"}

REF_MAP: Mapping[Literal["U0", "Ut", "Stiff"], str] = {
    "U0": "Disp-50.D",
    "Ut": "Disp-100.D",
    "Stiff": "Stiff-100.D",
}

TAG_MAP: Mapping[Literal["U0", "Ut", "Stiff"], str] = {
    "U0": "$U_0$",
    "Ut": "$U$",
    "Stiff": "$k$",
}


def calculate_norm[F: np.floating](val: A2[F], ref: A2[F]) -> Ok[float] | Err:
    log = get_logger()
    if val.shape != ref.shape:
        return Err(
            ValueError(
                f"Shape mismatch when calculating norm: "
                f"val.shape={val.shape}, ref.shape={ref.shape}"
            )
        )
    res = val - ref
    log.info(
        f"val: median = {np.median(np.abs(val)):<12g}, max = {np.max(np.abs(val)):<12g}",
        f"ref: median = {np.median(np.abs(ref)):<12g}, max = {np.max(np.abs(ref)):<12g}",
        f"res: median = {np.median(np.abs(res)):<12g}, max = {np.max(np.abs(res)):<12g}",
    )
    return Ok(np.sqrt(np.einsum("ij,ij->", res, res)))


def get_weighted_values[F: np.floating](cl: A2[F], var: A1[F], support: A1[F]) -> Ok[float] | Err:
    if len(support) != _3D:
        msg = f"Support must have length 3, got {len(support)}"
        return Err(ValueError(msg))
    left, c, right = support
    weights = np.maximum(
        np.minimum((cl[:, 0] - left) / (c - left), (right - cl[:, 0]) / (right - c)), 0.0
    )
    if len(var) != len(weights):
        msg = f"Variable and weights must have the same length, got {len(var)} and {len(weights)}"
        return Err(ValueError(msg))
    n_dim = len(var.shape)
    weight_dim = (weights.shape[0],) + (1,) * (n_dim - 1)
    weights = weights.reshape(weight_dim)
    return Ok(np.sum(var * weights, axis=0) / np.sum(weights))


def plot_convergence[F: np.floating](
    *err_vals: A1[F], ticks: Sequence[str], fout: Path, **kwargs: Unpack[PlotKwargs]
) -> None:
    kwargs = (
        PlotKwargs(padleft=0.3, padbottom=0.2, padtop=0.01, padright=0.015, markersize=2) | kwargs
    )
    fig, ax = create_figure(figsize=kwargs.get("figsize", (3, 2)), dpi=kwargs.get("dpi", 600))
    update_figure_setting(fig, **kwargs)
    style = style_kwargs(**kwargs)
    for v in err_vals:
        ax.plot(v, "o-", **style)
    # if 0.1 * np.max(np.concatenate(err_vals)) / np.min(np.concatenate(err_vals)) < 1.0:
    #     base = 2
    # else:
    #     base = 10
    base = 10
    ax.set_yscale("log", base=base)
    ax.set_xticks(range(len(ticks)))
    ax.set_xticklabels(ticks)
    ax.tick_params(axis="both", which="major", labelsize=6, reset=True)
    curve_labels = kwargs.get("curve_labels")
    if curve_labels is not None:
        ax.legend(curve_labels, fontsize=6, loc="upper right", frameon=False)
    fig.savefig(fout, transparent=kwargs.get("transparency", True))
    close_figure(fig)


class _PointData[F: np.floating](NamedTuple):
    x: A2[F]
    y: A2[F]


def plot_points[F: np.floating](
    *points: _PointData[F], fout: Path, **kwargs: Unpack[PlotKwargs]
) -> None:
    kwargs = (
        PlotKwargs(padleft=0.2, padbottom=0.2, padtop=0.01, padright=0.015, markersize=2) | kwargs
    )
    fig, ax = create_figure(figsize=kwargs.get("figsize", (1.5, 1)), dpi=kwargs.get("dpi", 600))
    style = style_kwargs(**kwargs)
    update_figure_setting(fig, **kwargs)
    for x, y in points:
        ax.plot(x, y, "o", **style)
    ax.tick_params(axis="both", which="major", labelsize=6)
    fig.savefig(fout, transparent=kwargs.get("transparency", True))
    close_figure(fig)


class _SimSet(TypedDict, total=False):
    shape: Required[Literal["straight", "bent", "bulge"]]
    mode: Required[Literal["const", "grad", "sine", "circ"]]
    freq: Literal[1, 2]
    mag: Literal["0.5", "1.0"]


def import_d_file[F: np.floating](file: Path, *, dtype: DType[F] = np.float64) -> Ok[A2[F]] | Err:
    if file.is_file():
        return Ok(chread_d(file, dtype=dtype))
    msg = f"{file} not found."
    return Err(FileNotFoundError(msg))


def import_d_files[F: np.floating](
    *files: Path, dtype: DType[F] = np.float64
) -> Ok[Sequence[A2[F]]] | Err:
    match all_ok([import_d_file(file, dtype=dtype) for file in files]):
        case Ok(values):
            return Ok(values)
        case Err(e):
            return Err(e)


class _CLMesh[F: np.floating](NamedTuple):
    cl: A2[F]
    nn: int
    normal: A2[F]
    support: A2[F]


def import_cl_mesh[F: np.floating](
    mesh: MeshInfo, ne: int, *, dtype: DType[F] = np.float64
) -> Ok[_CLMesh[F]] | Err:
    if not (file := mesh.DIR / "CenterLineField-0.D").is_file():
        msg = f"{file} not found."
        return Err(FileNotFoundError(msg))
    cl = chread_d(file, dtype=dtype)
    if not (file := mesh.DIR / "CenterNormalField-0.D").is_file():
        msg = f"{file} not found."
        return Err(FileNotFoundError(msg))
    normal = chread_d(mesh.DIR / "CenterNormalField-0.D", dtype=dtype)
    part = create_cl_partition(("DL", 4), ne=ne, ftype=dtype)
    return Ok(_CLMesh(cl=cl, nn=cl.shape[0], normal=normal, support=part.support))


def calculate_weighted_norm[F: np.floating](
    val: A2[F], ref: A2[F], mesh: MeshInfo, ne: int
) -> Ok[float] | Err:
    match import_cl_mesh(mesh, ne):
        case Ok(cl):
            pass
        case Err(e):
            return Err(e)
    if val.shape == cl.normal.shape:
        diff: A1[F] = np.einsum("ij,ij->i", val - ref, cl.normal)
    else:
        diff = (val - ref).reshape(-1).astype(val.dtype)
        # return Ok(np.sqrt(np.sum(diff * diff) / len(diff)))
    # print(f"{diff.shape=}, {diff.mean()=}, {diff.std()=}")
    # print(f"{diff.min()=}, {diff.max()=}, {ref.mean()=}")
    match all_ok([get_weighted_values(cl.cl, diff, support) for support in cl.support]):
        case Ok(w_v):
            eps = np.asarray(w_v, dtype=cl.cl.dtype)
            return Ok(np.sqrt(np.sum(eps * eps)) / len(eps))
        case Err(e):
            return Err(e)


def l2_convergence_figure(
    dataset: _SimSet,
    _mesh: MeshInfo,
    *vs: Literal["U0", "Ut", "Stiff"],
    root: Path,
    log: ILogger,
    **kwargs: Unpack[PlotKwargs],
) -> Ok[None] | Err:
    kwargs = PlotKwargs(color=[_COLOR_MAP[v] for v in vs]) | kwargs
    refinements = [2, 4, 8, 16]
    ref_files = {
        v: [
            Path("forward") / f"forward_{dataset['shape']}_{dataset['mode']}_{i}" / REF_MAP[v]
            for i in refinements
        ]
        for v in vs
    }
    log.info(ref_files)
    match all_ok({v: import_d_files(*ref_files[v]) for v in vs}):
        case Ok(ref):
            pass
        case Err(e):
            return Err(e)
    cur_files = {
        v: [
            Path("inverse") / f"inverse_{dataset['shape']}_{dataset['mode']}_{i}" / f"{v}-100.D"
            for i in refinements
        ]
        for v in vs
    }
    log.disp(cur_files)
    match all_ok({v: import_d_files(*cur_files[v]) for v in vs}):
        case Ok(cur):
            pass
        case Err(e):
            return Err(e)
    log.info({k: [np.max(np.abs(c)) for c in v] for k, v in cur.items()})
    match all_ok(
        {v: all_ok([calculate_norm(c, r) for c, r in zip(cur[v], ref[v], strict=True)]) for v in vs}
    ):
        case Ok(err_norms):
            err_norms = {k: np.asarray(v, dtype=np.float64) for k, v in err_norms.items()}
        case Err(e):
            return Err(e)
    labels = [TAG_MAP[v] for v in vs] if len(vs) > 1 else None
    kwargs = kwargs | PlotKwargs(curve_labels=labels) if labels is not None else kwargs
    plot_convergence(
        *err_norms.values(),
        ticks=[str(i) for i in refinements],
        fout=root / f"convergence_{dataset['shape']}_{dataset['mode']}_{'_'.join(vs)}.png",
        **kwargs,
    )
    return Ok(None)


def mean_convergence_figure(
    dataset: _SimSet,
    mesh: MeshInfo,
    *vs: Literal["U0", "Ut", "Stiff"],
    root: Path,
    log: ILogger,
    **kwargs: Unpack[PlotKwargs],
) -> Ok[None] | Err:
    kwargs = PlotKwargs(color=[_COLOR_MAP[v] for v in vs]) | kwargs
    refinements = [2, 4, 8, 16]
    ref_files = {
        v: [
            Path("forward") / f"forward_{dataset['shape']}_{dataset['mode']}_{16}" / REF_MAP[v]
            for _i in refinements
        ]
        for v in vs
    }
    cur_files = {
        v: [
            Path("inverse") / f"inverse_{dataset['shape']}_{dataset['mode']}_{i}" / f"{v}-100.D"
            for i in refinements
        ]
        for v in vs
    }
    log.info(ref_files, cur_files)
    match all_ok({v: import_d_files(*ref_files[v]) for v in vs}):
        case Ok(ref):
            pass
        case Err(e):
            return Err(e)
    match all_ok({v: import_d_files(*cur_files[v]) for v in vs}):
        case Ok(cur):
            pass
        case Err(e):
            return Err(e)
    log.info({k: [np.max(np.abs(c)) for c in v] for k, v in cur.items()})
    match all_ok(
        {
            v: all_ok(
                [
                    calculate_weighted_norm(c, r, mesh, n)
                    for n, c, r in zip(refinements, cur[v], ref[v], strict=True)
                ]
            )
            for v in vs
        }
    ):
        case Ok(err_norms):
            err_norms = {
                k: np.asarray(v, dtype=np.float64)
                / np.sqrt(np.asarray(refinements, dtype=np.float64))
                for k, v in err_norms.items()
            }
            log.info(err_norms)
        case Err(e):
            return Err(e)
    labels = [TAG_MAP[v] for v in vs] if len(vs) > 1 else None
    kwargs = kwargs | PlotKwargs(curve_labels=labels) if labels is not None else kwargs
    plot_convergence(
        *err_norms.values(),
        ticks=[str(i) for i in refinements],
        fout=root / f"convergence_{dataset['shape']}_{dataset['mode']}_{'_'.join(vs)}.png",
        **kwargs,
    )
    return Ok(None)


def compute_err_on_clnodes[F: np.floating](
    v: Literal["U0", "Ut", "Stiff"], dataset: _SimSet, cl: _CLMesh[F], normal: A2[F] | None = None
) -> Ok[dict[int, A2[F]]] | Err:
    freq = dataset.get("freq")
    if freq is None:
        return Err(ValueError("Frequency 'freq' must be specified in dataset"))
    mag = dataset.get("mag")
    if mag is None:
        return Err(ValueError("Magnitude 'mag' must be specified in dataset"))
    match import_d_file(
        Path("forward") / f"forward_{dataset['shape']}_{dataset['mode']}_8" / REF_MAP[v],
        dtype=cl.cl.dtype,
    ):
        case Ok(ref):
            files = {
                k: Path("noise")
                / f"noise_{dataset['shape']}_{dataset['mode']}_{8}_{freq}_{mag}_{k}"
                / f"{v}-100.D"
                for k in range(10)
            }
        case Err(e):
            return Err(e)
    match all_ok({k: import_d_file(f, dtype=cl.cl.dtype) for k, f in files.items()}):
        case Ok(disp):
            data: dict[int, A2[F]] = {k: (d - ref).astype(cl.cl.dtype) for k, d in disp.items()}
        case Err(e):
            return Err(e)
    if normal is not None:
        diff: dict[int, A1[F]] = {k: np.einsum("ij,ij->i", d, normal) for k, d in data.items()}
        denom: A1[F] = np.einsum("ij,ij->i", ref, normal)
    else:
        diff = {k: d.reshape(-1).astype(cl.cl.dtype) for k, d in data.items()}
        denom = ref.reshape(-1).astype(cl.cl.dtype)
    match all_ok(
        {
            k: all_ok([get_weighted_values(cl.cl, d, support) for support in cl.support])
            for k, d in diff.items()
        }
    ):
        case Ok(dct):
            errors = {k: np.asarray(v, dtype=cl.cl.dtype) for k, v in dct.items()}
        case Err(e):
            return Err(e)
    match all_ok([get_weighted_values(cl.cl, denom, support) for support in cl.support]):
        case Ok(dct):
            return Ok(
                {
                    k: (np.asarray(v) / np.abs(np.asarray(dct))).astype(cl.cl.dtype)
                    for k, v in errors.items()
                }
            )
        case Err(e):
            return Err(e)


def noise_figure(
    dataset: _SimSet, mesh: MeshInfo, root: Path, **kwargs: Unpack[PlotKwargs]
) -> Ok[None] | Err:
    freq = dataset.get("freq")
    if freq is None:
        return Err(ValueError("Frequency 'freq' must be specified in dataset"))
    mag = dataset.get("mag")
    if mag is None:
        return Err(ValueError("Magnitude 'mag' must be specified in dataset"))
    match import_cl_mesh(mesh, 8):
        case Ok(cl):
            pass
        case Err(e):
            return Err(e)
    match compute_err_on_clnodes("Ut", dataset, cl, cl.normal):
        case Ok(disp_err):
            pass
        case Err(e):
            return Err(e)
    match compute_err_on_clnodes("Stiff", dataset, cl):
        case Ok(stiff_err):
            pass
        case Err(e):
            return Err(e)
    plot_data = [
        _PointData(x=d, y=s) for (d, s) in zip(disp_err.values(), stiff_err.values(), strict=True)
    ]
    plot_points(
        *plot_data,
        fout=root / f"noise_{dataset['shape']}_{dataset['mode']}_f{freq}_m{mag}.png",
        **kwargs,
    )
    return Ok(None)


class FigureDef(TypedDict, total=True):
    type: Literal["l2_convergence", "mean_convergence", "noise"]
    dataset: _SimSet
    mesh: MeshInfo
    variables: Sequence[Literal["U0", "Ut", "Stiff"]]


_FREQS: Sequence[Literal[1, 2]] = [1, 2]
_MAGS: Sequence[Literal["0.5", "1.0"]] = ["0.5", "1.0"]
_MODES: Sequence[Literal["circ", "const", "grad", "sine"]] = ["grad", "sine"]
_SHAPES: Sequence[Literal["straight", "bent", "bulge"]] = ["straight", "bent", "bulge"]
FIG1: Sequence[FigureDef] = [
    *[
        FigureDef(
            type="l2_convergence",
            dataset=_SimSet(shape="straight", mode=mode),
            mesh=STRAIGHT_CYLINDER_QUAD_MESH,
            variables=["U0", "Ut"],
        )
        for mode in _MODES
    ],
    *[
        FigureDef(
            type="l2_convergence",
            dataset=_SimSet(shape="straight", mode=mode),
            mesh=STRAIGHT_CYLINDER_QUAD_MESH,
            variables=["Stiff"],
        )
        for mode in _MODES
    ],
    *[
        FigureDef(
            type="noise",
            dataset=_SimSet(shape="straight", mode="grad", freq=freq, mag=mag),
            mesh=STRAIGHT_CYLINDER_QUAD_MESH,
            variables=[],
        )
        for freq in _FREQS
        for mag in _MAGS
    ],
    *[
        FigureDef(
            type="noise",
            dataset=_SimSet(shape="straight", mode="sine", freq=freq, mag=mag),
            mesh=STRAIGHT_CYLINDER_QUAD_MESH,
            variables=[],
        )
        for freq in _FREQS
        for mag in _MAGS
    ],
]


FIG2: Sequence[FigureDef] = [
    FigureDef(
        type="l2_convergence",
        dataset=_SimSet(shape="bent", mode="grad"),
        mesh=BENT_CYLINDER_QUAD_MESH,
        variables=["U0", "Ut"],
    ),
    FigureDef(
        type="l2_convergence",
        dataset=_SimSet(shape="bent", mode="grad"),
        mesh=BENT_CYLINDER_QUAD_MESH,
        variables=["Stiff"],
    ),
    FigureDef(
        type="l2_convergence",
        dataset=_SimSet(shape="bulge", mode="sine"),
        mesh=BENT_CYLINDER_QUAD_MESH,
        variables=["U0", "Ut"],
    ),
    FigureDef(
        type="l2_convergence",
        dataset=_SimSet(shape="bulge", mode="sine"),
        mesh=BENT_CYLINDER_QUAD_MESH,
        variables=["Stiff"],
    ),
    # *[
    #     FigureDef(
    #         type="noise",
    #         dataset=_SimSet(shape="bent", mode="grad", freq=freq, mag=mag),
    #         mesh=BENT_CYLINDER_QUAD_MESH,
    #         variables=[],
    #     )
    #     for freq in _FREQS
    #     for mag in _MAGS
    # ],
    # *[
    #     FigureDef(
    #         type="noise",
    #         dataset=_SimSet(shape="bent", mode="sine", freq=freq, mag=mag),
    #         mesh=BENT_CYLINDER_QUAD_MESH,
    #         variables=[],
    #     )
    #     for freq in _FREQS
    #     for mag in _MAGS
    # ],
    # *[
    #     FigureDef(
    #         type="noise",
    #         dataset=_SimSet(shape="bulge", mode="grad", freq=freq, mag=mag),
    #         mesh=BENT_CYLINDER_QUAD_MESH,
    #         variables=[],
    #     )
    #     for freq in _FREQS
    #     for mag in _MAGS
    # ],
    *[
        FigureDef(
            type="noise",
            dataset=_SimSet(shape="bulge", mode="sine", freq=freq, mag=mag),
            mesh=BENT_CYLINDER_QUAD_MESH,
            variables=[],
        )
        for freq in _FREQS
        for mag in _MAGS
    ],
]


FIG3: Sequence[FigureDef] = [
    FigureDef(
        type="mean_convergence",
        dataset=_SimSet(shape="bulge", mode="circ"),
        mesh=BENT_CYLINDER_QUAD_MESH,
        variables=["Stiff"],
    ),
    FigureDef(
        type="mean_convergence",
        dataset=_SimSet(shape="bulge", mode="circ"),
        mesh=BENT_CYLINDER_QUAD_MESH,
        variables=["U0", "Ut"],
    ),
    *[
        FigureDef(
            type="noise",
            dataset=_SimSet(shape="bulge", mode="circ", freq=freq, mag=mag),
            mesh=BENT_CYLINDER_QUAD_MESH,
            variables=[],
        )
        for freq in _FREQS
        for mag in _MAGS
    ],
]


def make_figure(fig: FigureDef, root: Path, log: ILogger, **kwargs: Unpack[PlotKwargs]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    match fig["type"]:
        case "l2_convergence":
            log.brief(f"Generating figure: {fig['dataset']}, {fig['variables']}")
            match l2_convergence_figure(
                fig["dataset"], fig["mesh"], *fig["variables"], root=root, log=log, **kwargs
            ):
                case Ok():
                    return
                case Err(e):
                    log.error(f"Error generating convergence figure: {e}")
        case "mean_convergence":
            log.brief(f"Generating figure: {fig['dataset']}, {fig['variables']}")
            match mean_convergence_figure(
                fig["dataset"], fig["mesh"], *fig["variables"], root=root, log=log, **kwargs
            ):
                case Ok():
                    return
                case Err(e):
                    log.error(f"Error generating convergence figure: {e}")
        case "noise":
            log.brief(f"Generating figure: {fig['dataset']}")
            match noise_figure(fig["dataset"], fig["mesh"], root=root, **kwargs):
                case Ok():
                    return
                case Err(e):
                    log.error(f"Error generating noise figure: {e}")
    raise SystemExit(1)


def main(figs: Sequence[FigureDef], root: Path, log: ILogger, **kwargs: Unpack[PlotKwargs]) -> None:
    for fig in figs:
        make_figure(fig, root=root, log=log, **kwargs)


if __name__ == "__main__":
    fig_home = Path("figures")
    log = get_logger(level="INFO")
    main(FIG1, root=fig_home / "fig_straight", log=log)
    main(FIG2, root=fig_home / "fig_bent", log=log)
    main(FIG3, root=fig_home / "fig_circ", log=log)
