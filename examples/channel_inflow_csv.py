#!/usr/bin/env python3
"""
Channel with time-varying inflow from CSV file.

Demonstrates the HYDRO_BC_TIME_SERIES boundary condition.
Water flows in from the left with a triangular hydrograph,
while the other three sides are reflective walls.

Ported from ANUGA pattern in scenario/setup_boundary_conditions.py
"""

import os
import sys
import time
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

from hydro import Domain, HYDRO_BC_TIME_SERIES, HYDRO_BC_REFLECTIVE
from hydro.mesh_builder import rectangular_cross_domain
from hydro.time_series_boundary import TimeSeriesInflowBoundary

# ---------------------------------------------------------------------------
# Build mesh
# ---------------------------------------------------------------------------
length = 10.0
width = 5.0
dx = dy = 1.0

vertices, triangles, btags, bedges = rectangular_cross_domain(
    int(length / dx), int(width / dy), len1=length, len2=width,
)
n_tri = len(triangles)
print(f"Mesh: {len(vertices)} nodes, {n_tri} triangles, {len(btags)} boundary edges")

# ---------------------------------------------------------------------------
# Initial conditions
# ---------------------------------------------------------------------------
cent_x = vertices[triangles].mean(axis=1)[:, 0]
elevation = -cent_x / 10.0  # linear bed slope

# ---------------------------------------------------------------------------
# Create domain
# ---------------------------------------------------------------------------
domain = Domain(vertices, triangles, btags, bedges)
domain.set_name('channel_inflow_csv')
domain.set_output_dir(os.path.join(os.path.dirname(__file__)))

domain.set_elevation(elevation)
domain.set_friction(np.full(n_tri, 0.01))
domain.set_stage(elevation.copy())          # dry bed
domain.set_xmomentum(np.zeros(n_tri))
domain.set_ymomentum(np.zeros(n_tri))

domain.set_parameter("CFL", 0.9)
domain.set_parameter("spatial_order", 1)
domain.set_parameter("timestepping_method", 1)

# ---------------------------------------------------------------------------
# Boundary conditions
# ---------------------------------------------------------------------------
domain.set_boundary(1, HYDRO_BC_TIME_SERIES)   # left — inflow from CSV
domain.set_boundary(2, HYDRO_BC_REFLECTIVE)    # right — wall
domain.set_boundary(3, HYDRO_BC_REFLECTIVE)    # top — wall
domain.set_boundary(4, HYDRO_BC_REFLECTIVE)    # bottom — wall

# ---------------------------------------------------------------------------
# Create time-series boundary from inline data (or CSV file)
# ---------------------------------------------------------------------------
# Option A: inline arrays
times = np.array([0.0, 2.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0])
q_values = np.array([
    0.0, 1.5, 3.0, 2.5, 1.0, 0.5, 0.3, 0.1, 0.0
])

# Option B: from CSV file (uncomment to use)
# with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
#     f.write("time,Q\n")
#     for t, Q in zip(times, q_values):
#         f.write(f"{t},{Q}\n")
#     csv_path = f.name
# bc = TimeSeriesInflowBoundary(domain, tag=1, csv_file=csv_path)
# os.unlink(csv_path)

bc = TimeSeriesInflowBoundary(
    domain, tag=1, times=times, q_values=q_values,
)

# ---------------------------------------------------------------------------
# Evolve
# ---------------------------------------------------------------------------
t0 = time.time()
finaltime = 40.0
yieldstep = 0.2

print(f"\nEvolution: t=0 → {finaltime} s (yieldstep={yieldstep})")
domain.evolve(finaltime=finaltime, yieldstep=yieldstep)
elapsed = time.time() - t0

stage = domain.get_stage()
height = stage - elevation

output_path = os.path.join(os.path.dirname(__file__),  'channel_inflow_csv.sww')
print(f"\nEvolved in {elapsed:.1f} s")
print(f"Stage   : [{stage.min():.4f}, {stage.max():.4f}]")
print(f"Height  : [{height.min():.4f}, {height.max():.4f}]")
wet = height > 0.001
volume = np.sum(height[wet] * (0.5 * dx * dy))
print(f"Wet     : {wet.sum()} / {n_tri} triangles")
print(f"Volume  : {volume:.4f} m^3")
print(f"SWW     : {output_path}")

bc.close()
domain.close()
