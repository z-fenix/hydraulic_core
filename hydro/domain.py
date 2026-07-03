"""
Hydro Domain — re-exports from the native pybind11 _core extension.

Usage
-----
>>> from hydro import Domain
>>> domain = Domain(vertices, triangles)
>>> domain.set_elevation(bed_values)
>>> domain.set_stage(stage_values)
>>> domain.set_name('output')  # SWW → output_dir/output.sww
>>> domain.evolve(finaltime=10.0, yieldstep=1.0)
>>> stage = domain.get_stage()
>>> domain.close()
"""

from hydro._core import (
    Domain,
    # Constants
    G,
    EPSILON,
    MINIMUM_ALLOWED_HEIGHT,
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
)
