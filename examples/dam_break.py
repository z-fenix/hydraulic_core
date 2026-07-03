#!/usr/bin/env python3
"""
Dam-break simulation using Hydro Core.

Simulates the instantaneous failure of a dam separating a 2 m deep
reservoir from a 1 m deep tailwater, on a 4-triangle rectangular mesh.
Outputs an SWW file for post-processing.
"""

import os
import sys

# Add project root to path so "import hydro" works from this directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from hydro import Domain, G, EULER, RK2, RK3


def main():
    print("=== Hydro Core — Dam Break Example ===\n")

    # ---- 1. Build the mesh ------------------------------------------------
    # 6 nodes on a 3 × 2 grid, 4 triangles (two squares, each split diagonally)

    vertices = np.array(
        [[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1]],
        dtype=np.float64,
    )
    triangles = np.array(
        [[0, 1, 4], [0, 4, 3], [1, 2, 5], [1, 5, 4]],
        dtype=np.int64,
    )

    # ---- 2. Create domain ------------------------------------------------
    domain = Domain(vertices, triangles)
    print(f"Domain created: {domain}")

    # ---- 3. Set initial conditions ---------------------------------------
    # Flat bed
    domain.set_elevation(np.zeros(domain.n_tri))

    # Dam break: 2 m on the left (tri 0,1), 1 m on the right (tri 2,3)
    domain.set_stage(np.array([2.0, 2.0, 1.0, 1.0], dtype=np.float64))

    # Still water initially
    domain.set_xmomentum(np.zeros(domain.n_tri))
    domain.set_ymomentum(np.zeros(domain.n_tri))

    # Manning's n
    domain.set_friction(np.full(domain.n_tri, 0.03))

    # ---- 4. Set simulation parameters ------------------------------------
    domain.set_parameter("CFL", 1.0)
    domain.set_parameter("spatial_order", 1)  # 1 = first-order
    domain.set_parameter("timestepping_method", EULER)
    domain.set_parameter("evolve_max_timestep", 0.1)

    # ---- 5. Evolve -------------------------------------------------------
    output_path = os.path.join(
        os.path.dirname(__file__), "..", "dam_break_output.sww"
    )
    print(f"\nEvolving to t=0.2 with yieldstep=0.05...")
    domain.evolve(finaltime=0.2, yieldstep=0.05, output_sww=output_path)

    # ---- 6. Inspect results ----------------------------------------------
    stage = domain.get_stage()
    xmom = domain.get_xmomentum()
    elev = domain.get_elevation()
    h = stage - elev

    print(f"\nFinal time: {domain.get_time():.4f} s")
    print(f"Stage: {stage}")
    print(f"Height: {h}")
    print(f"x-momentum: {xmom}")
    print(f"\nSWW output: {output_path}")

    # ---- 7. Cleanup ------------------------------------------------------
    domain.close()
    print("Done.")


if __name__ == "__main__":
    main()
