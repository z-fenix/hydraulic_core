#!/usr/bin/env python3
"""
Water flowing down a channel with changing boundary conditions.

Starts with reflective right wall.  Once water reaches the downstream end
(stage > 0 at x=10), the right boundary is changed to an outflow
(Dirichlet stage = -5).

Ported from: anuga_core/examples/simple_examples/channel2.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from hydro import Domain, EULER
from hydro import HYDRO_BC_DIRICHLET, HYDRO_BC_REFLECTIVE
from hydro.mesh_builder import rectangular_cross_domain

# ---------------------------------------------------------------------------
# Build mesh
# ---------------------------------------------------------------------------
length = 10.0
width = 5.0
dx = dy = 1.0

vertices, triangles, bedges, btags = rectangular_cross_domain(
    int(length / dx), int(width / dy), len1=length, len2=width,
)
n_tri = len(triangles)
print(f"Mesh: {len(vertices)} nodes, {n_tri} triangles")

# ---------------------------------------------------------------------------
# Initial conditions
# ---------------------------------------------------------------------------
cent_x = vertices[triangles].mean(axis=1)[:, 0]
elevation = -cent_x / 10.0

# ---------------------------------------------------------------------------
# Create domain
# ---------------------------------------------------------------------------
domain = Domain(vertices, triangles, btags, bedges)

domain.set_elevation(elevation)
domain.set_friction(np.full(n_tri, 0.01))
domain.set_stage(elevation.copy())          # dry bed
domain.set_xmomentum(np.zeros(n_tri))
domain.set_ymomentum(np.zeros(n_tri))

domain.set_parameter("CFL", 1.0)
domain.set_parameter("spatial_order", 1)
domain.set_parameter("timestepping_method", EULER)

# ---------------------------------------------------------------------------
# Boundary conditions — start with reflective right wall
# ---------------------------------------------------------------------------
domain.set_boundary(1, HYDRO_BC_DIRICHLET, stage=0.4)     # left — inflow
domain.set_boundary(2, HYDRO_BC_REFLECTIVE)               # right — wall (initially)
domain.set_boundary(3, HYDRO_BC_REFLECTIVE)               # top — wall
domain.set_boundary(4, HYDRO_BC_REFLECTIVE)               # bottom — wall

# ---------------------------------------------------------------------------
# Evolve with dynamic boundary switching
# ---------------------------------------------------------------------------
output_path = os.path.join(os.path.dirname(__file__), "..", "channel2_output.sww")
outflow_enabled = False

t0 = time.time()

# Run in sub-intervals so we can check stage and switch the boundary
current_time = 0.0
finaltime = 40.0
yieldstep = 0.2

while current_time < finaltime:
    target = min(current_time + yieldstep, finaltime)
    domain.evolve(finaltime=target, yieldstep=target - current_time)

    stage = domain.get_stage()
    current_time = target

    # Check if water has reached the downstream end
    # (any triangle near x=10 with positive stage)
    right_mask = cent_x > 9.0
    if right_mask.any() and np.max(stage[right_mask]) > 0.0:
        if not outflow_enabled:
            outflow_enabled = True
            print(f"t={current_time:.1f}: Stage > 0 at outlet — switching to outflow")
            domain.set_boundary(2, HYDRO_BC_DIRICHLET, stage=-5.0)

elapsed = time.time() - t0

stage = domain.get_stage()
height = stage - elevation

print(f"\nEvolved in {elapsed:.1f} s")
print(f"Stage   : [{stage.min():.4f}, {stage.max():.4f}]")
print(f"Height  : [{height.min():.4f}, {height.max():.4f}]")
print(f"Volume  : {np.sum(height * 0.25):.4f} m^3")
print(f"SWW     : {output_path}")

domain.close()
