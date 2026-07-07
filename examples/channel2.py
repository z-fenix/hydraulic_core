#!/usr/bin/env python3
"""
Water flowing down a channel with changing boundary conditions.

Starts with reflective right wall for the first 5 seconds to let water
build up, then switches to an outflow boundary (Dirichlet stage = -5).

NOTE: Dynamic boundary switching during evolve requires a generator-style
API (as in ANUGA's ``for t in domain.evolve(...)``).  The current hydro_core
C evolve is a single blocking call, so we approximate the behavior by
evolving in two phases.

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
domain.set_name('channel2')
domain.set_output_dir(os.path.join(os.path.dirname(__file__)))

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
domain.set_boundary(1, HYDRO_BC_DIRICHLET, depth=0.4)     # left — inflow
domain.set_boundary(2, HYDRO_BC_REFLECTIVE)               # right — wall
domain.set_boundary(3, HYDRO_BC_REFLECTIVE)               # top — wall
domain.set_boundary(4, HYDRO_BC_REFLECTIVE)               # bottom — wall

# ---------------------------------------------------------------------------
# Phase 1: Inflow with reflective right wall (water builds up)
# ---------------------------------------------------------------------------
output_path = os.path.join(os.path.dirname(__file__), "channel2.sww")

t0 = time.time()

print("Phase 1: Reflective right wall (0 → 20 s)")
domain.evolve(finaltime=20.0, yieldstep=0.2)

stage = domain.get_stage()
right_mask = cent_x > 9.0
print(f"  Stage near outlet: max={stage[right_mask].max():.4f}")

# ---------------------------------------------------------------------------
# Phase 2: Switch to outflow boundary (append to same SWW)
# ---------------------------------------------------------------------------
print("Phase 2: Outflow right wall (20 → 40 s)")
domain.set_boundary(2, HYDRO_BC_DIRICHLET, depth=-5.0)
domain.evolve(finaltime=40.0, yieldstep=0.2)

elapsed = time.time() - t0

stage = domain.get_stage()
height = stage - elevation

print(f"\nEvolved in {elapsed:.1f} s")
print(f"Stage   : [{stage.min():.4f}, {stage.max():.4f}]")
print(f"Height  : [{height.min():.4f}, {height.max():.4f}]")
print(f"Volume  : {np.sum(height * 0.25):.4f} m^3")
print(f"SWW     : {output_path}")

domain.close()
