"""
Hydro Core — ctypes bindings to libhydro

Loads the shared library and declares all public C function signatures
as Python-callable wrappers.  NumPy arrays are passed as ctypes pointers
via ``np.ctypeslib.ndpointer`` or manual ``ctypes.c_void_p`` casting.

Usage
-----
>>> from hydro import _core
>>> dom = _core.hydro_domain_create(100, 180)
>>> _core.hydro_domain_destroy(dom)
"""

import ctypes
import os
import sys
from ctypes import (
    c_int,
    c_int64,
    c_double,
    c_char_p,
    c_void_p,
    POINTER,
    byref,
)

import numpy as np
from numpy.ctypeslib import ndpointer

# ---------------------------------------------------------------------------
# Type aliases matching hydro/types.h
# ---------------------------------------------------------------------------
hydro_int = c_int64                # typedef int64_t hydro_int
hydro_uint = ctypes.c_uint64       # typedef uint64_t hydro_uint

# ---------------------------------------------------------------------------
# Find and load the shared library
# ---------------------------------------------------------------------------

def _find_libhydro():
    """Locate libhydro.so relative to the build directory or LD_LIBRARY_PATH."""
    # 1. Check environment variable
    path = os.environ.get("HYDRO_LIB_PATH")
    if path and os.path.exists(path):
        return path

    # 2. Look in the standard build location
    candidates = [
        # In-build-tree (meson/CMake)
        os.path.join(os.path.dirname(__file__), "..", "build", "libhydro.so"),
        os.path.join(os.path.dirname(__file__), "..", "builddir", "libhydro.so"),
        # Installed alongside the package
        os.path.join(os.path.dirname(__file__), "libhydro.so"),
        # System-wide
        "libhydro.so",
    ]

    for p in candidates:
        if os.path.exists(p):
            return p
        # Also try via ld
    try:
        lib = ctypes.CDLL("libhydro.so")
        return "libhydro.so"
    except OSError:
        pass

    # Fall back to build/ relative to CWD
    cwd_build = os.path.join(os.getcwd(), "build", "libhydro.so")
    if os.path.exists(cwd_build):
        return cwd_build

    raise OSError(
        "Cannot find libhydro.so.  Set HYDRO_LIB_PATH or build the library:\n"
        "  cd /path/to/hydro_core && mkdir build && cd build && cmake .. && make"
    )


_lib_path = _find_libhydro()
_lib = ctypes.CDLL(_lib_path)

# ===================================================================
# domain.c
# ===================================================================

_lib.hydro_domain_create.argtypes = [hydro_int, hydro_int]
_lib.hydro_domain_create.restype = c_void_p

_lib.hydro_domain_destroy.argtypes = [c_void_p]
_lib.hydro_domain_destroy.restype = None

_lib.hydro_domain_set_geometry.argtypes = [
    c_void_p,
    ndpointer(np.float64, flags="C_CONTIGUOUS"),
    ndpointer(np.int64, flags="C_CONTIGUOUS"),
    ndpointer(np.int64, flags="C_CONTIGUOUS"),
    ndpointer(np.int64, flags="C_CONTIGUOUS"),
]
_lib.hydro_domain_set_geometry.restype = None

_lib.hydro_domain_set_boundary_tag_map.argtypes = [
    c_void_p,
    ndpointer(np.int64, flags="C_CONTIGUOUS"),
    ndpointer(np.int64, flags="C_CONTIGUOUS"),
    hydro_int,
]
_lib.hydro_domain_set_boundary_tag_map.restype = None

_lib.hydro_domain_set_quantity.argtypes = [
    c_void_p, c_char_p,
    ndpointer(np.float64, flags="C_CONTIGUOUS"),
]
_lib.hydro_domain_set_quantity.restype = None

_lib.hydro_domain_set_parameter.argtypes = [c_void_p, c_char_p, c_double]
_lib.hydro_domain_set_parameter.restype = None

_lib.hydro_domain_get_quantity.argtypes = [
    c_void_p, c_char_p,
    ndpointer(np.float64, flags="C_CONTIGUOUS"),
]
_lib.hydro_domain_get_quantity.restype = None

_lib.hydro_domain_get_time.argtypes = [c_void_p]
_lib.hydro_domain_get_time.restype = c_double

# ===================================================================
# mesh.c
# ===================================================================

_lib.hydro_mesh_build_neighbour_structure.argtypes = [c_void_p]
_lib.hydro_mesh_build_neighbour_structure.restype = c_int

_lib.hydro_mesh_build_boundary_structure.argtypes = [c_void_p]
_lib.hydro_mesh_build_boundary_structure.restype = c_int

# ===================================================================
# quantity.c
# ===================================================================

_lib.hydro_quantity_update.argtypes = [c_void_p, c_double]
_lib.hydro_quantity_update.restype = None

_lib.hydro_quantity_extrapolate_first_order.argtypes = [c_void_p]
_lib.hydro_quantity_extrapolate_first_order.restype = None

_lib.hydro_quantity_extrapolate_second_order.argtypes = [c_void_p]
_lib.hydro_quantity_extrapolate_second_order.restype = None

_lib.hydro_quantity_extrapolate_second_order_edge.argtypes = [c_void_p]
_lib.hydro_quantity_extrapolate_second_order_edge.restype = None

_lib.hydro_quantity_distribute_edges_to_vertices.argtypes = [c_void_p]
_lib.hydro_quantity_distribute_edges_to_vertices.restype = None

_lib.hydro_quantity_backup.argtypes = [c_void_p]
_lib.hydro_quantity_backup.restype = None

_lib.hydro_quantity_saxpy.argtypes = [c_void_p, c_double, c_double, c_double]
_lib.hydro_quantity_saxpy.restype = None

_lib.hydro_quantity_update_derived.argtypes = [c_void_p]
_lib.hydro_quantity_update_derived.restype = None

# ===================================================================
# fluxes.c
# ===================================================================

_lib.hydro_compute_fluxes_central.argtypes = [c_void_p, c_double]
_lib.hydro_compute_fluxes_central.restype = c_double

_lib.hydro_protect.argtypes = [c_void_p]
_lib.hydro_protect.restype = c_int

_lib.hydro_fix_negative_cells.argtypes = [c_void_p]
_lib.hydro_fix_negative_cells.restype = c_int

# Boundary condition types
HYDRO_BC_NONE          = 0
HYDRO_BC_REFLECTIVE    = 1
HYDRO_BC_DIRICHLET     = 2
HYDRO_BC_TRANSMISSIVE  = 3
HYDRO_BC_TIME          = 4
HYDRO_BC_DIRICHLET_DISCHARGE = 5
HYDRO_BC_TRANSMISSIVE_STAGE   = 6

# BC parameters struct
class BCParams(ctypes.Structure):
    """Corresponds to C hydro_bc_params_t."""
    _fields_ = [
        ("stage", c_double),
        ("wh0",   c_double),
    ]


# ===================================================================
# boundaries.c
# ===================================================================

_lib.hydro_boundary_update.argtypes = [c_void_p]
_lib.hydro_boundary_update.restype = None

_lib.hydro_domain_set_boundary.argtypes = [c_void_p, hydro_int, c_int, c_void_p]
_lib.hydro_domain_set_boundary.restype = None

_lib.hydro_boundary_update_stage_time.argtypes = [
    c_void_p, hydro_int, c_double, c_double, c_double,
]
_lib.hydro_boundary_update_stage_time.restype = None

_lib.hydro_boundary_evaluate_reflective_segment.argtypes = [
    c_void_p, hydro_int,
    ndpointer(np.int64, flags="C_CONTIGUOUS"),
    ndpointer(np.int64, flags="C_CONTIGUOUS"),
    ndpointer(np.int64, flags="C_CONTIGUOUS"),
]
_lib.hydro_boundary_evaluate_reflective_segment.restype = None

_lib.hydro_boundary_evaluate_dirichlet_segment.argtypes = [
    c_void_p, hydro_int,
    ndpointer(np.int64, flags="C_CONTIGUOUS"),
    ndpointer(np.int64, flags="C_CONTIGUOUS"),
    ndpointer(np.int64, flags="C_CONTIGUOUS"),
    c_double,
]
_lib.hydro_boundary_evaluate_dirichlet_segment.restype = None

_lib.hydro_boundary_evaluate_transmissive_segment.argtypes = [
    c_void_p, hydro_int,
    ndpointer(np.int64, flags="C_CONTIGUOUS"),
    ndpointer(np.int64, flags="C_CONTIGUOUS"),
    ndpointer(np.int64, flags="C_CONTIGUOUS"),
    c_double, c_int,
]
_lib.hydro_boundary_evaluate_transmissive_segment.restype = None

_lib.hydro_boundary_evaluate_discharge_segment.argtypes = [
    c_void_p, hydro_int,
    ndpointer(np.int64, flags="C_CONTIGUOUS"),
    ndpointer(np.int64, flags="C_CONTIGUOUS"),
    ndpointer(np.int64, flags="C_CONTIGUOUS"),
    c_double, c_double,
]
_lib.hydro_boundary_evaluate_discharge_segment.restype = None

# ===================================================================
# friction.c
# ===================================================================

_lib.hydro_manning_friction_flat_semi_implicit.argtypes = [c_void_p]
_lib.hydro_manning_friction_flat_semi_implicit.restype = None

_lib.hydro_manning_friction_sloped_semi_implicit.argtypes = [c_void_p]
_lib.hydro_manning_friction_sloped_semi_implicit.restype = None

# ===================================================================
# forcing.c
# ===================================================================

_lib.hydro_wind_stress_apply.argtypes = [
    c_void_p,
    ndpointer(np.float64, flags="C_CONTIGUOUS"), c_double,
    ndpointer(np.float64, flags="C_CONTIGUOUS"), c_double,
    hydro_int, c_double,
]
_lib.hydro_wind_stress_apply.restype = None

_lib.hydro_barometric_pressure_apply.argtypes = [
    c_void_p, ndpointer(np.float64, flags="C_CONTIGUOUS"), hydro_int, c_double,
]
_lib.hydro_barometric_pressure_apply.restype = None

_lib.hydro_rainfall_apply.argtypes = [c_void_p, c_double]
_lib.hydro_rainfall_apply.restype = None

_lib.hydro_rate_apply.argtypes = [
    c_void_p,
    ndpointer(np.float64, flags="C_CONTIGUOUS"),
    ndpointer(np.int64, flags="C_CONTIGUOUS"),
    hydro_int,
]
_lib.hydro_rate_apply.restype = None

_lib.hydro_gradient_triangle.argtypes = [
    c_double, c_double, c_double, c_double, c_double, c_double,
    c_double, c_double, c_double,
    POINTER(c_double), POINTER(c_double),
]
_lib.hydro_gradient_triangle.restype = None

# ===================================================================
# operators.c
# ===================================================================

_lib.hydro_manning_friction_explicit.argtypes = [c_void_p]
_lib.hydro_manning_friction_explicit.restype = None

_lib.hydro_bed_shear_erosion_apply.argtypes = [
    c_void_p, c_double, c_double,
    ndpointer(np.int64, flags="C_CONTIGUOUS"), hydro_int,
]
_lib.hydro_bed_shear_erosion_apply.restype = None

_lib.hydro_set_stage.argtypes = [
    c_void_p,
    ndpointer(np.float64, flags="C_CONTIGUOUS"),
    ndpointer(np.int64, flags="C_CONTIGUOUS"), hydro_int,
]
_lib.hydro_set_stage.restype = None

_lib.hydro_set_elevation.argtypes = [
    c_void_p,
    ndpointer(np.float64, flags="C_CONTIGUOUS"),
    ndpointer(np.int64, flags="C_CONTIGUOUS"), hydro_int,
]
_lib.hydro_set_elevation.restype = None

_lib.hydro_kinematic_viscosity_apply.argtypes = [
    c_void_p, ndpointer(np.float64, flags="C_CONTIGUOUS"), c_double,
]
_lib.hydro_kinematic_viscosity_apply.restype = None

# ===================================================================
# structures.c
# ===================================================================

_lib.hydro_boyd_box_discharge.argtypes = [
    c_double, c_double, c_double, c_double, c_double,
    c_double, c_double, c_double, c_int,
    c_double, c_double, c_double, c_double,
    POINTER(c_double), POINTER(c_double), POINTER(c_double),
]
_lib.hydro_boyd_box_discharge.restype = c_int

_lib.hydro_boyd_pipe_discharge.argtypes = [
    c_double, c_double, c_double, c_double,
    c_double, c_double, c_double, c_int,
    c_double, c_double, c_double, c_double,
    POINTER(c_double), POINTER(c_double), POINTER(c_double),
]
_lib.hydro_boyd_pipe_discharge.restype = c_int

_lib.hydro_weir_orifice_trapezoid_discharge.argtypes = [
    c_double, c_double, c_double, c_double, c_double, c_double,
    c_double, c_double, c_double, c_int,
    c_double, c_double, c_double, c_double,
    POINTER(c_double), POINTER(c_double), POINTER(c_double),
]
_lib.hydro_weir_orifice_trapezoid_discharge.restype = c_int

_lib.hydro_inlet_distribute_volume.argtypes = [
    c_double,
    ndpointer(np.int64, flags="C_CONTIGUOUS"),
    ndpointer(np.float64, flags="C_CONTIGUOUS"),
    hydro_int,
    ndpointer(np.float64, flags="C_CONTIGUOUS"),
    ndpointer(np.float64, flags="C_CONTIGUOUS"),
    ndpointer(np.float64, flags="C_CONTIGUOUS"),
    ndpointer(np.float64, flags="C_CONTIGUOUS"),
]
_lib.hydro_inlet_distribute_volume.restype = c_double

# ===================================================================
# sww.c
# ===================================================================

_lib.hydro_sww_create.argtypes = [c_char_p, c_void_p, c_double]
_lib.hydro_sww_create.restype = c_void_p

_lib.hydro_sww_store_timestep.argtypes = [c_void_p, c_void_p, c_double]
_lib.hydro_sww_store_timestep.restype = c_int

_lib.hydro_sww_close.argtypes = [c_void_p]
_lib.hydro_sww_close.restype = c_int

# ===================================================================
# geometry.c
# ===================================================================

_lib.hydro_polygon_area.argtypes = [
    ndpointer(np.float64, flags="C_CONTIGUOUS"), hydro_int,
]
_lib.hydro_polygon_area.restype = c_double

_lib.hydro_is_inside_triangle.argtypes = [
    ndpointer(np.float64, flags="C_CONTIGUOUS"),
    ndpointer(np.float64, flags="C_CONTIGUOUS"),
    c_int,
]
_lib.hydro_is_inside_triangle.restype = c_int

_lib.hydro_is_inside_polygon.argtypes = [
    ndpointer(np.float64, flags="C_CONTIGUOUS"),
    ndpointer(np.float64, flags="C_CONTIGUOUS"),
    hydro_int, c_int,
]
_lib.hydro_is_inside_polygon.restype = c_int

_lib.hydro_separate_points_by_polygon.argtypes = [
    ndpointer(np.float64, flags="C_CONTIGUOUS"), hydro_int,
    ndpointer(np.float64, flags="C_CONTIGUOUS"), hydro_int,
    c_int,
    ndpointer(np.int64, flags="C_CONTIGUOUS"),
]
_lib.hydro_separate_points_by_polygon.restype = hydro_int

_lib.hydro_polygon_aabb.argtypes = [
    ndpointer(np.float64, flags="C_CONTIGUOUS"), hydro_int,
    POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double),
]
_lib.hydro_polygon_aabb.restype = None

_lib.hydro_point_on_line.argtypes = [
    ndpointer(np.float64, flags="C_CONTIGUOUS"),
    ndpointer(np.float64, flags="C_CONTIGUOUS"),
    c_double, c_double,
]
_lib.hydro_point_on_line.restype = c_int

_lib.hydro_line_intersection.argtypes = [
    ndpointer(np.float64, flags="C_CONTIGUOUS"),
    ndpointer(np.float64, flags="C_CONTIGUOUS"),
    ndpointer(np.float64, flags="C_CONTIGUOUS"),
]
_lib.hydro_line_intersection.restype = c_int

# ===================================================================
# coordinate_transforms.c
# ===================================================================

_lib.hydro_geo_ref_init.argtypes = [c_void_p]
_lib.hydro_geo_ref_init.restype = None

_lib.hydro_geo_ref_set_utm.argtypes = [c_void_p, hydro_int, c_int]
_lib.hydro_geo_ref_set_utm.restype = None

_lib.hydro_geo_ref_get_epsg.argtypes = [c_void_p]
_lib.hydro_geo_ref_get_epsg.restype = c_int

_lib.hydro_geo_ref_to_absolute.argtypes = [
    c_void_p,
    ndpointer(np.float64, flags="C_CONTIGUOUS"), hydro_int,
]
_lib.hydro_geo_ref_to_absolute.restype = None

_lib.hydro_geo_ref_to_relative.argtypes = [
    c_void_p,
    ndpointer(np.float64, flags="C_CONTIGUOUS"), hydro_int,
]
_lib.hydro_geo_ref_to_relative.restype = None

_lib.hydro_redfearn_latlon_to_utm.argtypes = [
    c_double, c_double,
    POINTER(hydro_int), POINTER(c_double), POINTER(c_double),
    c_double, c_double,
]
_lib.hydro_redfearn_latlon_to_utm.restype = None

_lib.hydro_redfearn_utm_to_latlon.argtypes = [
    hydro_int, c_double, c_double, c_int,
    POINTER(c_double), POINTER(c_double),
]
_lib.hydro_redfearn_utm_to_latlon.restype = None

_lib.hydro_dms_to_decimal_degrees.argtypes = [c_int, c_int, c_double]
_lib.hydro_dms_to_decimal_degrees.restype = c_double

_lib.hydro_decimal_degrees_to_dms.argtypes = [
    c_double, POINTER(c_int), POINTER(c_int), POINTER(c_double),
]
_lib.hydro_decimal_degrees_to_dms.restype = None

# ===================================================================
# fit_interpolate.c
# ===================================================================

_lib.hydro_find_containing_triangle.argtypes = [
    ndpointer(np.float64, flags="C_CONTIGUOUS"),
    ndpointer(np.float64, flags="C_CONTIGUOUS"),
    ndpointer(np.int64, flags="C_CONTIGUOUS"),
    hydro_int, hydro_int,
]
_lib.hydro_find_containing_triangle.restype = c_void_p  # struct, not trivial

_lib.hydro_interpolate_at_point.argtypes = [
    ndpointer(np.float64, flags="C_CONTIGUOUS"),
    ndpointer(np.float64, flags="C_CONTIGUOUS"),
    ndpointer(np.int64, flags="C_CONTIGUOUS"),
    hydro_int,
    ndpointer(np.float64, flags="C_CONTIGUOUS"),
    c_double,
]
_lib.hydro_interpolate_at_point.restype = c_double

_lib.hydro_interpolate_batch.argtypes = [
    ndpointer(np.float64, flags="C_CONTIGUOUS"),
    ndpointer(np.float64, flags="C_CONTIGUOUS"),
    ndpointer(np.int64, flags="C_CONTIGUOUS"),
    hydro_int,
    ndpointer(np.float64, flags="C_CONTIGUOUS"),
    hydro_int,
    ndpointer(np.float64, flags="C_CONTIGUOUS"),
    c_double,
]
_lib.hydro_interpolate_batch.restype = None

_lib.hydro_interpolate_regular_grid.argtypes = [
    ndpointer(np.float64, flags="C_CONTIGUOUS"), hydro_int,
    ndpointer(np.float64, flags="C_CONTIGUOUS"), hydro_int,
    ndpointer(np.float64, flags="C_CONTIGUOUS"),
    ndpointer(np.float64, flags="C_CONTIGUOUS"), hydro_int,
    ndpointer(np.float64, flags="C_CONTIGUOUS"),
    c_int, c_double,
]
_lib.hydro_interpolate_regular_grid.restype = None

# ===================================================================
# timestepping.c
# ===================================================================

_lib.hydro_domain_evolve.argtypes = [c_void_p, c_double, c_double, c_char_p]
_lib.hydro_domain_evolve.restype = c_int

_lib.hydro_evolve_one_euler_step.argtypes = [c_void_p, c_double, c_double]
_lib.hydro_evolve_one_euler_step.restype = None

_lib.hydro_evolve_one_rk2_step.argtypes = [c_void_p, c_double, c_double]
_lib.hydro_evolve_one_rk2_step.restype = None

_lib.hydro_update_timestep.argtypes = [c_void_p, c_double, c_double]
_lib.hydro_update_timestep.restype = None


# ===================================================================
# Convenience wrappers for Python import
# ===================================================================

def hydro_polygon_area(polygon):
    """Compute polygon area. polygon: ndarray (N, 2) or flat."""
    p = np.asarray(polygon, dtype=np.float64).ravel()
    return _lib.hydro_polygon_area(p, len(p) // 2)


def hydro_is_inside_polygon(point, polygon, closed=True):
    p = np.asarray(point, dtype=np.float64).ravel()
    pg = np.asarray(polygon, dtype=np.float64).ravel()
    return bool(_lib.hydro_is_inside_polygon(p, pg, len(pg) // 2, int(closed)))


def hydro_is_inside_triangle(point, triangle, closed=True):
    p = np.asarray(point, dtype=np.float64).ravel()
    t = np.asarray(triangle, dtype=np.float64).ravel()
    return bool(_lib.hydro_is_inside_triangle(p, t, int(closed)))


def hydro_separate_points_by_polygon(points, polygon, closed=True):
    pts = np.asarray(points, dtype=np.float64).ravel()
    pg = np.asarray(polygon, dtype=np.float64).ravel()
    M = len(pts) // 2
    N = len(pg) // 2
    indices = np.empty(M, dtype=np.int64)
    count = _lib.hydro_separate_points_by_polygon(pts, M, pg, N, int(closed), indices)
    return indices, count


def hydro_point_on_line(point, line, rtol=1e-5, atol=1e-8):
    p = np.asarray(point, dtype=np.float64).ravel()
    l = np.asarray(line, dtype=np.float64).ravel()
    return bool(_lib.hydro_point_on_line(p, l, rtol, atol))


# ===================================================================
# Helper: wrap hydro_find_containing_triangle return struct
# ===================================================================

class InterpResult(ctypes.Structure):
    """Mirrors hydro_interp_result_t — search result from the C library."""
    _fields_ = [
        ("triangle_index", hydro_int),
        ("sigma", c_double * 3),
        ("vertices", hydro_int * 3),
    ]


def find_containing_triangle(point, vertex_coords, triangles, start_triangle=-1):
    """Python-friendly wrapper around hydro_find_containing_triangle.

    Returns a dict {triangle_index, sigma, vertices} or None if not found.
    """
    lib = _lib
    # The C function return type needs special handling — it returns by value.
    # We use ctypes to call it properly.
    lib.hydro_find_containing_triangle.restype = InterpResult
    result = lib.hydro_find_containing_triangle(
        point, vertex_coords, triangles,
        len(triangles) // 3, start_triangle,
    )
    if result.triangle_index < 0:
        return None
    return {
        "triangle_index": int(result.triangle_index),
        "sigma": [result.sigma[0], result.sigma[1], result.sigma[2]],
        "vertices": [int(result.vertices[0]), int(result.vertices[1]), int(result.vertices[2])],
    }
