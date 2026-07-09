"""
Hydro Core — Shallow Water Equation Solver
===========================================

A standalone C library for finite-volume simulation of the shallow water
equations on unstructured triangular meshes, with pybind11 Python bindings.

Quick start
-----------
>>> import numpy as np
>>> from hydro import Domain
>>>
>>> # 4-triangle mesh (6 nodes on a 3x2 grid)
>>> vertices = np.array([[0,0],[1,0],[2,0],[0,1],[1,1],[2,1]], dtype=np.float64)
>>> triangles = np.array([[0,1,4],[0,4,3],[1,2,5],[1,5,4]], dtype=np.int64)
>>>
>>> domain = Domain(vertices, triangles)
>>> domain.set_elevation(np.zeros(4))
>>> domain.set_stage(np.array([2.0, 2.0, 1.0, 1.0]))  # dam break
>>> domain.set_xmomentum(np.zeros(4))
>>> domain.set_ymomentum(np.zeros(4))
>>> domain.set_friction(np.full(4, 0.03))
>>>
>>> domain.set_parameter('CFL', 1.0)
>>> domain.set_parameter('spatial_order', 1)
>>> domain.set_parameter('timestepping_method', 1)
>>>
>>> domain.set_name('dam_break')  # SWW → ./dam_break.sww
>>> domain.evolve(finaltime=0.1, yieldstep=0.05)
>>> print(domain.get_stage())
>>> domain.close()
"""

try:
    from hydro._core import (  # type: ignore[import]
        Domain,
        # Constants
        G,
        EPSILON,
        MINIMUM_ALLOWED_HEIGHT,
        MINIMUM_STORABLE_HEIGHT,
        MAXIMUM_ALLOWED_SPEED,
        DEFAULT_MANNING,
        EVOLVE_MAX_TIMESTEP,
        EVOLVE_MIN_TIMESTEP,
        # Timestepping methods
        EULER,
        RK2,
        RK3,
        # Boundary condition types
        HYDRO_BC_NONE,
        HYDRO_BC_REFLECTIVE,
        HYDRO_BC_DIRICHLET,
        HYDRO_BC_TRANSMISSIVE,
        HYDRO_BC_TIME,
        HYDRO_BC_DIRICHLET_DISCHARGE,
        HYDRO_BC_TRANSMISSIVE_STAGE,
        HYDRO_BC_TIME_SERIES,
    )
except ModuleNotFoundError:
    import sys
    print(
        "╔══════════════════════════════════════════════════════════════╗\n"
        "║  hydro._core extension not built.                           ║\n"
        "║                                                            ║\n"
        "║  Build it with:                                            ║\n"
        "║    ./scripts/build_pybind11.sh install                     ║\n"
        "║                                                            ║\n"
        "║  Or install the package:                                   ║\n"
        "║    pip install -e .                                        ║\n"
        "╚══════════════════════════════════════════════════════════════╝",
        file=sys.stderr,
    )
    raise

__version__ = "0.1.0"

# High-level Python wrappers (depend on _core being available)
from hydro.time_series_boundary import TimeSeriesInflowBoundary
from hydro.quantity_centroids import QuantityCentroids
from hydro.mesh_utils import (
    build_boundary_info,
    load_csv_quantity_to_centroids,
    find_boundary_edges_near_region,
    map_region_to_boundary,
)
