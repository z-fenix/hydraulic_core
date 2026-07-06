#!/usr/bin/env python3
"""
Merewether flood simulation — 2007 Pasha Bulka storm surge event.

Ported from ANUGA validation_tests/case_studies/merewether.
Uses real 1 m DEM topography (ASC grid), UTM zone 56S.
Houses and variable friction are omitted for simplicity.

Usage
-----
    python examples/merewether/run_merewether.py [resolution_m]

    resolution_m : mesh cell size in metres (default 8, min 1)
    e.g.  8 → ~8 000 triangles (fast, ~30 s / 60 s sim)
          4 → ~33 000 triangles
          2 → ~130 000 triangles

Requires:
    - topography1.zip in the ANUGA merewether data directory
    - hydro_core built (./scripts/build_pybind11.sh install)
"""

import os
import sys
import time
import zipfile

import numpy as np

# Add hydro_core to path
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _PROJ)

from hydro import Domain, EULER
from hydro import HYDRO_BC_REFLECTIVE, HYDRO_BC_TRANSMISSIVE
from hydro.mesh_builder import rectangular_cross_domain

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_DATA = "/home/zhang/work/std/cfd/anuga_core/validation_tests/case_studies/merewether"

# ---------------------------------------------------------------------------
# 1. Load topography from ASC grid (inside topography1.zip)
# ---------------------------------------------------------------------------
print("Loading topography ...")
with zipfile.ZipFile(os.path.join(_DATA, "topography1.zip")) as zf:
    raw = zf.read("topography1.asc").decode("ascii").splitlines()

# Parse 6-line ASC header
hdr = {}
data_start = 0
for i, line in enumerate(raw):
    line = line.strip()
    if not line:
        continue
    parts = line.split()
    key = parts[0].lower()
    if key == "ncols":
        hdr["ncols"] = int(parts[1])
    elif key == "nrows":
        hdr["nrows"] = int(parts[1])
    elif key == "xllcorner":
        hdr["xllcorner"] = float(parts[1])
    elif key == "yllcorner":
        hdr["yllcorner"] = float(parts[1])
    elif key == "cellsize":
        hdr["cellsize"] = float(parts[1])
    elif key == "nodata_value":
        hdr["nodata_value"] = int(parts[1])
    else:
        data_start = i
        break

ncols   = hdr["ncols"]
nrows   = hdr["nrows"]
x0      = hdr["xllcorner"]
y0      = hdr["yllcorner"]
cellsize = hdr["cellsize"]
nodata  = hdr.get("nodata_value", -9999)

length_x = ncols * cellsize   # ~321 m
length_y = nrows * cellsize   # ~416 m

# Read grid values (loadtxt returns 2-D array for multi-line input)
topo = np.loadtxt(raw[data_start:], dtype=np.float64)
topo[topo == nodata] = np.nan

# ASC coordinate centres (rows run north→south, so y decreases)
x_asc = x0 + cellsize * (np.arange(ncols) + 0.5)
y_asc = y0 + cellsize * (np.arange(nrows - 1, -1, -1) + 0.5)

print(f"  DEM  : {ncols}×{nrows} @ {cellsize:.2f} m")
print(f"  Extent: [{x0:.0f}, {x0+length_x:.0f}] × [{y0:.0f}, {y0+length_y:.0f}]")
print(f"  Z     : [{np.nanmin(topo):.1f}, {np.nanmax(topo):.1f}] m")

# ---------------------------------------------------------------------------
# 2. Mesh resolution
# ---------------------------------------------------------------------------
ds = int(sys.argv[1]) if len(sys.argv) > 1 else 8
ds = max(ds, 1)
nx = ncols // ds
ny = nrows // ds
n_cells = nx * ny
n_tri   = n_cells * 4
print(f"\nMesh : {nx}×{ny} cells → {n_tri} triangles @ ~{ds} m")

# ---------------------------------------------------------------------------
# 3. Build mesh
# ---------------------------------------------------------------------------
vertices, triangles, bedges, btags = rectangular_cross_domain(
    nx, ny, len1=length_x, len2=length_y,
)
# Shift to UTM coordinates
vertices[:, 0] += x0
vertices[:, 1] += y0
print(f"  Nodes: {len(vertices)}, Edges: {len(bedges)}")

# ---------------------------------------------------------------------------
# 4. Interpolate topography → centroids (nearest-neighbour, no-scipy)
# ---------------------------------------------------------------------------
cent_x = vertices[triangles].mean(axis=1)[:, 0]
cent_y = vertices[triangles].mean(axis=1)[:, 1]

# Map centroid → ASC grid indices
col = ((cent_x - x0) / cellsize).astype(int)
row = nrows - 1 - ((cent_y - y0) / cellsize).astype(int)
col = np.clip(col, 0, ncols - 1)
row = np.clip(row, 0, nrows - 1)

elevation = topo[row, col].copy()
nan_mask = np.isnan(elevation)
if nan_mask.any():
    elevation[nan_mask] = 50.0  # outside DEM → high ground (stays dry)
    print(f"  Filled {nan_mask.sum()} cells outside DEM")

print(f"  Z mesh: [{elevation.min():.1f}, {elevation.max():.1f}]")

# ---------------------------------------------------------------------------
# 5. Create domain
# ---------------------------------------------------------------------------
domain = Domain(vertices, triangles, btags, bedges)
domain.set_name("merewether")
domain.set_output_dir(_HERE)  # examples/

# UTM zone 56S
domain.set_parameter("zone", -56)
domain.set_parameter("xllcorner", x0)
domain.set_parameter("yllcorner", y0)

domain.set_elevation(elevation)
domain.set_friction(np.full(n_tri, 0.03))
domain.set_xmomentum(np.zeros(n_tri))
domain.set_ymomentum(np.zeros(n_tri))

domain.set_parameter("CFL", 0.9)
domain.set_parameter("spatial_order", 1)
domain.set_parameter("timestepping_method", EULER)
domain.set_parameter("evolve_max_timestep", 10.0)

# ---------------------------------------------------------------------------
# 6. Boundary conditions
#
# Tags: 1=left, 2=right, 3=top, 4=bottom
# ANUGA: bottom/left=reflective, top/right=transmissive (outflow to sea)
# ---------------------------------------------------------------------------
domain.set_boundary(1, HYDRO_BC_REFLECTIVE)       # left   — closed
domain.set_boundary(2, HYDRO_BC_TRANSMISSIVE)     # right  — outflow (sea)
domain.set_boundary(3, HYDRO_BC_TRANSMISSIVE)     # top    — outflow
domain.set_boundary(4, HYDRO_BC_REFLECTIVE)       # bottom — closed

# ---------------------------------------------------------------------------
# 7. Inlet — Pasha Bulka overtopping
#
# Original: Inlet_operator at (382265, 6354280), radius=10 m, Q=19.7 m³/s
# We set initial stage in the inlet region to represent the storm surge.
# ---------------------------------------------------------------------------
INLET_CX, INLET_CY = 382265.0, 6354280.0
INLET_RADIUS = 15.0

inlet = (cent_x - INLET_CX)**2 + (cent_y - INLET_CY)**2 < INLET_RADIUS**2
print(f"\nInlet: {inlet.sum()} triangles @ ({INLET_CX}, {INLET_CY})")

stage = elevation.copy()
stage[inlet] = np.maximum(elevation[inlet] + 3.0, 5.0)
domain.set_stage(stage)

# ---------------------------------------------------------------------------
# 8. Evolve
# ---------------------------------------------------------------------------
finaltime = 600.0   # 10 min (original: 1000 s)
yieldstep = 30.0

print(f"\nEvolving t=0 → {finaltime} s ...")
t0 = time.time()
domain.evolve(finaltime=finaltime, yieldstep=yieldstep)
elapsed = time.time() - t0

stage = domain.get_stage()
height = stage - elevation
wet = height > 0.01

print(f"\nElapsed : {elapsed:.0f} s")
print(f"Stage   : [{stage.min():.1f}, {stage.max():.1f}]")
print(f"Height  : [{height[wet].min():.2f}, {height.max():.2f}]" if wet.any()
      else "Height  : all dry")
print(f"Wet     : {wet.sum()} / {n_tri}")
print(f"SWW     : {os.path.join(os.path.dirname(_HERE), 'merewether.sww')}")

domain.close()
