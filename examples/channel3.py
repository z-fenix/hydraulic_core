#!/usr/bin/env python3
"""
Water flow down a channel with complex topography — step, constriction, and pole.

A longer (40 m) channel at higher resolution (dx = dy = 0.1 m).
The right boundary switches from reflective to outflow once water
passes the constriction.

Ported from: anuga_core/examples/simple_examples/channel3.py
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
length = 40.0
width = 5.0
dx = dy = 0.1

vertices, triangles, bedges, btags = rectangular_cross_domain(
    int(length / dx), int(width / dy), len1=length, len2=width,
)
n_tri = len(triangles)
print(f"Mesh: {len(vertices)} nodes, {n_tri} triangles")

# ---------------------------------------------------------------------------
# Complex topography
# ---------------------------------------------------------------------------
cent_x = vertices[triangles].mean(axis=1)[:, 0]
cent_y = vertices[triangles].mean(axis=1)[:, 1]

elevation = -cent_x / 10.0

# Step
mask_step = (10 < cent_x) & (cent_x < 12)
elevation[mask_step] += 0.4 - 0.05 * cent_y[mask_step]

# Constriction
mask_con = (27 < cent_x) & (cent_x < 29) & (cent_y > 3)
elevation[mask_con] += 2.0

# Pole
mask_pole = (cent_x - 34) ** 2 + (cent_y - 2) ** 2 < 0.4 ** 2
elevation[mask_pole] += 2.0

print(f"Elevation range: [{elevation.min():.3f}, {elevation.max():.3f}]")

# ---------------------------------------------------------------------------
# Create domain
# ---------------------------------------------------------------------------
domain = Domain(vertices, triangles, btags, bedges)
domain.set_name('channel3')
domain.set_output_dir(os.path.join(os.path.dirname(__file__), '..'))

domain.set_elevation(elevation)
domain.set_friction(np.full(n_tri, 0.01))
domain.set_stage(elevation.copy())          # dry bed
domain.set_xmomentum(np.zeros(n_tri))
domain.set_ymomentum(np.zeros(n_tri))

domain.set_parameter("CFL", 1.0)
domain.set_parameter("spatial_order", 1)
domain.set_parameter("timestepping_method", EULER)

# ---------------------------------------------------------------------------
# Boundary conditions
# ---------------------------------------------------------------------------
domain.set_boundary(1, HYDRO_BC_DIRICHLET, stage=0.4)     # left — inflow
domain.set_boundary(2, HYDRO_BC_REFLECTIVE)               # right — wall (initially)
domain.set_boundary(3, HYDRO_BC_REFLECTIVE)               # top — wall
domain.set_boundary(4, HYDRO_BC_REFLECTIVE)               # bottom — wall

# ---------------------------------------------------------------------------
# Evolve with dynamic boundary switching
# ---------------------------------------------------------------------------
outflow_enabled = False
yieldstep = 0.1
finaltime = 25.0

t0 = time.time()
current_time = 0.0

while current_time < finaltime:
    target = min(current_time + yieldstep, finaltime)
    domain.evolve(finaltime=target, yieldstep=target - current_time)
    current_time = target

    # Check stage near the outlet (x ≈ 37, y ≈ 2.5)
    out_mask = (cent_x > 36.0) & (cent_x < 38.0) & (cent_y > 2.0) & (cent_y < 3.0)
    if out_mask.any():
        stage_pt = np.max(domain.get_stage()[out_mask])
        if stage_pt > -3.3 and not outflow_enabled:
            outflow_enabled = True
            print(f"t={current_time:.1f}: Stage > -3.3 at outlet, stage_pt={stage_pt:.3f} — switching to outflow")
            domain.set_boundary(2, HYDRO_BC_DIRICHLET, stage=-5.0)

elapsed = time.time() - t0

stage = domain.get_stage()
height = stage - elevation

print(f"\nEvolved in {elapsed:.1f} s")
print(f"Stage   : [{stage.min():.4f}, {stage.max():.4f}]")
print(f"Height  : [{height.min():.4f}, {height.max():.4f}]")
wet = height > 0.001
volume = np.sum(height * (0.5 * dx * dy))  # triangle area = 0.5 * dx * dy
print(f"Wet     : {wet.sum()} / {n_tri} triangles")
print(f"Volume  : {volume:.4f} m^3")

domain.close()
