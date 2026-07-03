"""
Hydro Domain — Python wrapper around the C hydro_domain_t.

Usage
-----
>>> from hydro import Domain
>>> domain = Domain(vertices, triangles)
>>> domain.set_quantity('elevation', bed_values)
>>> domain.set_stage(stage_values)
>>> domain.evolve(finaltime=10.0, yieldstep=1.0, output_sww='output.sww')
>>> stage = domain.get_stage()
>>> domain.close()
"""

import ctypes
import os.path
from typing import Optional

import numpy as np

from . import _core

# ---------------------------------------------------------------------------
# Constants (mirror hydro/config.h)
# ---------------------------------------------------------------------------
G = 9.8
EPSILON = 1.0e-6
MINIMUM_ALLOWED_HEIGHT = 1.0e-5
MAXIMUM_ALLOWED_SPEED = 1000.0
DEFAULT_MANNING = 0.03
EVOLVE_MAX_TIMESTEP = 1000.0
EVOLVE_MIN_TIMESTEP = 1.0e-6

EULER = 1
RK2 = 2
RK3 = 3


class Domain:
    """Shallow-water simulation domain.

    Parameters
    ----------
    vertices : ndarray (N, 2) or (2*N,)
        UNIQUE vertex coordinates in metres.
    triangles : ndarray (M, 3) or (3*M,), int
        Triangle vertex indices (CCW ordering) into the UNIQUE vertex array.
    boundary_tags : ndarray (B,) or None, int
        Per-edge boundary tag: positive integer = boundary, 0 = interior.
    boundary_edges : ndarray (B,) or None, int
        Flat edge indices (0..3*M-1) of boundary edges.
    """

    _QUANTITY_NAMES = {"elevation", "stage", "xmomentum", "ymomentum", "friction"}

    def __init__(
        self,
        vertices: np.ndarray,
        triangles: np.ndarray,
        boundary_tags: Optional[np.ndarray] = None,
        boundary_edges: Optional[np.ndarray] = None,
    ):
        vertices = np.asarray(vertices, dtype=np.float64)
        triangles = np.asarray(triangles, dtype=np.int64)

        if vertices.ndim == 2:
            pass
        else:
            vertices = vertices.reshape(-1, 2)
        if triangles.ndim == 2:
            pass
        else:
            triangles = triangles.reshape(-1, 3)

        n_unique = len(vertices)
        n_tri = len(triangles)

        # The C domain stores vertex_coordinates in EXPANDED (per-triangle)
        # format: 3*n_tri entries.  Build from unique vertices.
        expanded_vc = vertices[triangles.ravel()]  # (n_tri*3, 2)

        self.n_tri = n_tri

        self._handle = _core._lib.hydro_domain_create(
            _core.hydro_int(n_unique),
            _core.hydro_int(n_tri),
        )
        if not self._handle:
            raise MemoryError("hydro_domain_create failed")

        # Boundary defaults
        if boundary_tags is None or boundary_edges is None:
            bt = np.zeros(3 * n_tri, dtype=np.int64)
            be = np.arange(3 * n_tri, dtype=np.int64)
        else:
            bt = np.asarray(boundary_tags, dtype=np.int64)
            be = np.asarray(boundary_edges, dtype=np.int64)

        self._set_geometry(
            np.asarray(expanded_vc, dtype=np.float64).ravel(),
            np.asarray(triangles, dtype=np.int64).ravel(),
            bt, be,
        )

        self.set_parameter("CFL", 1.0)
        self.set_parameter("spatial_order", 1)
        self.set_parameter("timestepping_method", 1)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _set_geometry(self, vc, tri, bt, be):
        _core._lib.hydro_domain_set_geometry(self._handle, vc, tri, bt, be)
        _core._lib.hydro_mesh_build_neighbour_structure(self._handle)
        _core._lib.hydro_mesh_build_boundary_structure(self._handle)

    @property
    def handle(self):
        return self._handle

    # ------------------------------------------------------------------
    # Quantities
    # ------------------------------------------------------------------

    def set_quantity(self, name: str, values: np.ndarray):
        if name not in self._QUANTITY_NAMES:
            raise ValueError(f"Unknown quantity '{name}'")
        arr = np.asarray(values, dtype=np.float64).ravel()
        if len(arr) != self.n_tri:
            raise ValueError(f"Expected {self.n_tri} values, got {len(arr)}")
        _core._lib.hydro_domain_set_quantity(
            self._handle, name.encode(), arr
        )

    def get_quantity(self, name: str) -> np.ndarray:
        out = np.empty(self.n_tri, dtype=np.float64)
        _core._lib.hydro_domain_get_quantity(
            self._handle, name.encode(), out
        )
        return out

    def set_parameter(self, name: str, value: float):
        _core._lib.hydro_domain_set_parameter(
            self._handle, name.encode(), float(value)
        )

    def set_boundary(self, tag: int, bc_type: int, stage: float = 0.0,
                     discharge: float = 0.0):
        params = _core.BCParams()
        params.stage = stage
        params.wh0 = discharge
        _core._lib.hydro_domain_set_boundary(
            self._handle, _core.hydro_int(tag), bc_type,
            ctypes.byref(params),
        )

    def set_boundary_stage(self, tag: int, stage: float, xmom=0.0, ymom=0.0):
        _core._lib.hydro_boundary_update_stage_time(
            self._handle, _core.hydro_int(tag), stage, xmom, ymom,
        )

    def get_time(self) -> float:
        return _core._lib.hydro_domain_get_time(self._handle)

    # ------------------------------------------------------------------
    # Evolve
    # ------------------------------------------------------------------

    def evolve(self, finaltime: float = 10.0, yieldstep: float = 1.0,
               output_sww: Optional[str] = None):
        _core._lib.hydro_quantity_update_derived(self._handle)
        path_bytes = output_sww.encode() if output_sww else None
        ret = _core._lib.hydro_domain_evolve(
            self._handle, float(finaltime), float(yieldstep), path_bytes,
        )
        if ret != 0:
            raise RuntimeError(f"hydro_domain_evolve failed (code {ret})")

    def close(self):
        if self._handle:
            _core._lib.hydro_domain_destroy(self._handle)
            self._handle = None

    def __del__(self):
        self.close()

    def __repr__(self):
        return f"HydroDomain(n_tri={self.n_tri})"

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def set_elevation(self, v): self.set_quantity("elevation", v)
    def set_stage(self, v):     self.set_quantity("stage", v)
    def set_xmomentum(self, v): self.set_quantity("xmomentum", v)
    def set_ymomentum(self, v): self.set_quantity("ymomentum", v)
    def set_friction(self, v):  self.set_quantity("friction", v)

    def get_stage(self):        return self.get_quantity("stage")
    def get_elevation(self):    return self.get_quantity("elevation")
    def get_xmomentum(self):    return self.get_quantity("xmomentum")
    def get_ymomentum(self):    return self.get_quantity("ymomentum")

    def get_height(self):
        return self.get_stage() - self.get_elevation()
