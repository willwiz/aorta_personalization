from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from cheartpy.mesh.struct import CheartMesh
    from pytools.arrays import A2


def create_fiber_field[F: np.floating](cl: A2[F], normal: A2[F], *, warp: bool = False) -> A2[F]:
    r = normal
    z = np.zeros_like(normal)
    if warp:
        q = 0.5 * np.pi * cl[:, 0]
        z[:, 0] = np.cos(q)
        z[:, 2] = -np.sin(q)
    else:
        z[:, 0] = 1.0
    c = np.cross(r, z)
    return np.column_stack((z, c, r)).astype(cl.dtype)


def warp_in_y[F: np.floating](x: A2[F]) -> A2[F]:
    c = np.zeros_like(x)
    radius = 2.0 * x[:, 0].max() / np.pi
    q = 0.5 * np.pi * (1.0 - x[:, 0] / x[:, 0].max())
    r = radius + x[:, 2]
    c[:, 0] = r * np.cos(q)
    c[:, 1] = x[:, 1]
    c[:, 2] = r * np.sin(q)
    return c


def define_centerline_field[F: np.floating, I: np.integer](mesh: CheartMesh[F, I]) -> A2[F]:
    center_line = mesh.space.v[:, [0]] / mesh.space.v[:, 0].max()
    x = mesh.space.v[:, 2]
    circval = (x - x.min()) / (x.max() - x.min())
    return np.hstack((center_line, circval[:, None]))


def create_center_pos[F: np.floating, I: np.integer](mesh: CheartMesh[F, I], cl: A2[F]) -> A2[F]:
    center = np.zeros_like(mesh.space.v)
    center[:, 0] = mesh.space.v[:, 0].max() * cl[:, 0]
    return center
