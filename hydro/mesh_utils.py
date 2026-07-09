#!/usr/bin/env python3
"""
Mesh utilities for hydro_core — boundary classification, quantity loading,
and region-to-boundary mapping.

These helpers bridge the gap between the YAML configuration format used by
hydraulic_simulate and the low-level numpy arrays expected by hydro_core.Domain.
"""

from __future__ import annotations

import os
from os.path import join
from typing import Sequence

import numpy as np
from scipy.spatial import KDTree


# ---------------------------------------------------------------------------
# Boundary edge classification
# ---------------------------------------------------------------------------

def find_boundary_edges(vertices: np.ndarray, triangles: np.ndarray):
    """Find edges that belong to exactly one triangle (boundary edges).

    Parameters
    ----------
    vertices : (N, 2) float64
    triangles : (M, 3) int64

    Returns
    -------
    boundary_edges : list of (v_lo, v_hi) sorted vertex-pair tuples
    """
    edge_to_tri: dict[tuple[int, int], list[int]] = {}
    for tri_id, (v0, v1, v2) in enumerate(triangles):
        for edge in [
            (min(v0, v1), max(v0, v1)),
            (min(v1, v2), max(v1, v2)),
            (min(v2, v0), max(v2, v0)),
        ]:
            edge_to_tri.setdefault(edge, []).append(tri_id)

    return [edge for edge, tri_ids in edge_to_tri.items() if len(tri_ids) == 1]


def _flat_edge_index(tri_id: int, local_edge: int) -> int:
    """Convert (triangle_id, local_edge) to flat edge index.

    Flat edge index = 3 * tri_id + local_edge
    where local_edge: 0=(v1,v2), 1=(v2,v0), 2=(v0,v1)  [C convention]
    """
    return 3 * tri_id + local_edge


def build_boundary_info(
    vertices: np.ndarray,
    triangles: np.ndarray,
    boundary_file: str,
    xllcorner: float = 0.0,
    yllcorner: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build boundary edge arrays from mesh connectivity and boundary tag file.

    Reads node-level tags from mesh_boundary.csv (in/out1/out2/bc),
    assigns hydro_core numeric tags to each boundary edge based on
    endpoint tags, then splits walls into tags 3/4 by y-position.

    Parameters
    ----------
    vertices : (N, 2) float64 — already shifted to local coords
    triangles : (M, 3) int64
    boundary_file : str — path to mesh_boundary.csv with columns: eid, x, y, tag
    xllcorner, yllcorner : float — coordinate shift (global -> local)

    Returns
    -------
    vertices, triangles (unchanged)
    boundary_tags : (B,) int64  (1=inflow, 2=outflow, 3/4=wall)
    boundary_edges : (B,) int64  (flat edge indices)
    """
    boundary_edges = find_boundary_edges(vertices, triangles)

    # Load node-level tags
    raw = np.loadtxt(boundary_file, delimiter=',', dtype=str, skiprows=1)
    node_indices = raw[:, 0].astype(int)
    tag_strings = raw[:, 3]
    node_tags = dict(zip(node_indices, tag_strings))

    # Build edge map: sorted vertex pair -> (tri_id, local_edge)
    # Edge numbering MUST match hydro_core/mesh.c convention:
    #   edge 0 = (v1, v2)  [opposite vertex 0]
    #   edge 1 = (v2, v0)  [opposite vertex 1]
    #   edge 2 = (v0, v1)  [opposite vertex 2]
    edge_map: dict[tuple[int, int], tuple[int, int]] = {}
    for tri_id, (v0, v1, v2) in enumerate(triangles):
        edge_pairs = [
            ((min(v1, v2), max(v1, v2)), 0),
            ((min(v2, v0), max(v2, v0)), 1),
            ((min(v0, v1), max(v0, v1)), 2),
        ]
        for (v_lo, v_hi), local_edge in edge_pairs:
            edge_map[(v_lo, v_hi)] = (tri_id, local_edge)

    # Tag map: node tag string -> hydro_core numeric tag
    TAG_MAP: dict[str, int | None] = {
        'in': 1,
        'out1': 2,
        'out2': 2,
        'bc': None,  # wall — assigned dynamically (3/4)
    }

    # Classify each boundary edge
    edge_entries: list[tuple[int, np.ndarray, int]] = []
    for v_lo, v_hi in boundary_edges:
        tri_id, local_edge = edge_map[(v_lo, v_hi)]
        flat_idx = _flat_edge_index(tri_id, local_edge)
        v0, v1, v2 = triangles[tri_id]
        verts = [vertices[v0], vertices[v1], vertices[v2]]
        # C edge i connects vertices (i+1)%3 and (i+2)%3
        midpoint = (verts[(local_edge + 1) % 3] + verts[(local_edge + 2) % 3]) / 2.0

        t1 = node_tags.get(v_lo, '')
        t2 = node_tags.get(v_hi, '')

        # Priority: if either endpoint is in/out1/out2, use that tag
        if t1 in TAG_MAP and TAG_MAP[t1] is not None:
            tag = TAG_MAP[t1]
        elif t2 in TAG_MAP and TAG_MAP[t2] is not None:
            tag = TAG_MAP[t2]
        else:
            tag = 0  # wall — to be assigned 3 or 4

        edge_entries.append((flat_idx, midpoint, tag))

    if not edge_entries:
        return (
            vertices, triangles,
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
        )

    flat_indices = np.array([e[0] for e in edge_entries], dtype=np.int64)
    midpoints = np.array([e[1] for e in edge_entries])
    tags = np.array([e[2] for e in edge_entries], dtype=np.int64)

    # Assign wall tags 3/4 based on y-position (top/bottom split)
    wall_mask = tags == 0
    if wall_mask.any():
        y_vals = midpoints[wall_mask, 1]
        y_median = np.median(y_vals)
        tags[wall_mask] = np.where(y_vals > y_median, 3, 4)

    return vertices, triangles, tags, flat_indices


# ---------------------------------------------------------------------------
# Quantity loading from CSV files
# ---------------------------------------------------------------------------

def load_csv_quantity_to_centroids(
    csv_path: str,
    vertices: np.ndarray,
    triangles: np.ndarray,
    xllcorner: float = 0.0,
    yllcorner: float = 0.0,
    col_value: int = 2,
) -> np.ndarray:
    """Load scalar values from a CSV file and assign to triangle centroids.

    The CSV file should have columns: x, y, value (or similar).
    Uses KDTree nearest-neighbor interpolation.

    Parameters
    ----------
    csv_path : str — path to CSV file
    vertices : (N, 2) float64
    triangles : (M, 3) int64
    xllcorner, yllcorner : float — coordinate shift if CSV is in global coords
    col_value : int — column index for the value (default 2 = 3rd column)

    Returns
    -------
    values : (M,) float64 — value at each triangle centroid
    """
    # Compute triangle centroids in local coordinates
    cent_coords = vertices[triangles]  # (M, 3, 2)
    centroids_local = cent_coords.mean(axis=1)  # (M, 2)

    # Convert to global coordinates for CSV lookup
    centroids_global = centroids_local + np.array([xllcorner, yllcorner])

    # Read CSV
    csv_data = np.loadtxt(csv_path, delimiter=',', skiprows=1)
    csv_points = csv_data[:, :2]
    csv_values = csv_data[:, col_value]

    # KDTree nearest-neighbor lookup
    tree = KDTree(csv_points)
    _, indices = tree.query(centroids_global)

    return csv_values[indices]


# ---------------------------------------------------------------------------
# Region-to-boundary mapping (for inlets / rainfall)
# ---------------------------------------------------------------------------

def find_boundary_edges_near_region(
    vertices: np.ndarray,
    triangles: np.ndarray,
    region_coords: Sequence[list[float] | tuple[float, float]],
    tolerance: float = 5.0,
) -> list[int]:
    """Find boundary edges near a given region (line or point).

    Parameters
    ----------
    vertices : (N, 2) float64
    triangles : (M, 3) int64
    region_coords : list of [x, y] points defining the region
    tolerance : float — distance in metres to consider "near"

    Returns
    -------
    flat_edge_indices : list of flat edge indices near the region
    """
    region_arr = np.array(region_coords, dtype=np.float64)

    # Build boundary edge midpoints
    boundary_edges = find_boundary_edges(vertices, triangles)
    edge_map: dict[tuple[int, int], tuple[int, int]] = {}
    for tri_id, (v0, v1, v2) in enumerate(triangles):
        edge_pairs = [
            ((min(v1, v2), max(v1, v2)), 0),
            ((min(v2, v0), max(v2, v0)), 1),
            ((min(v0, v1), max(v0, v1)), 2),
        ]
        for (v_lo, v_hi), local_edge in edge_pairs:
            edge_map[(v_lo, v_hi)] = (tri_id, local_edge)

    # Compute midpoints and flat indices
    midpoints = []
    flat_indices = []
    for v_lo, v_hi in boundary_edges:
        tri_id, local_edge = edge_map[(v_lo, v_hi)]
        v0, v1, v2 = triangles[tri_id]
        verts = [vertices[v0], vertices[v1], vertices[v2]]
        midpoint = (verts[(local_edge + 1) % 3] + verts[(local_edge + 2) % 3]) / 2.0
        midpoints.append(midpoint)
        flat_indices.append(_flat_edge_index(tri_id, local_edge))

    midpoints_arr = np.array(midpoints)

    # Find edges within tolerance of any region point
    near_mask = np.any(
        np.linalg.norm(midpoints_arr[:, np.newaxis] - region_arr[np.newaxis, :], axis=2)
        <= tolerance,
        axis=1,
    )

    return [flat_indices[i] for i in np.where(near_mask)[0]]


def map_region_to_boundary(
    vertices: np.ndarray,
    triangles: np.ndarray,
    region_coords: Sequence[list[float] | tuple[float, float]],
    boundary_tags: np.ndarray,
    boundary_edges: np.ndarray,
    tolerance: float = 5.0,
) -> list[int]:
    """Map a region (inlet/rainfall) to boundary edge indices.

    Returns the positions in the boundary_tags/edges arrays that correspond
    to edges near the given region.

    Parameters
    ----------
    vertices, triangles, region_coords, tolerance — see find_boundary_edges_near_region
    boundary_tags : (B,) int64 — already-classified boundary tags
    boundary_edges : (B,) int64 — flat edge indices

    Returns
    -------
    positions : list of indices into boundary_tags/boundary_edges
    """
    flat_near = set(find_boundary_edges_near_region(vertices, triangles, region_coords, tolerance))

    # Match flat indices to boundary array positions
    edge_to_pos = {edge: i for i, edge in enumerate(boundary_edges)}
    return [edge_to_pos[f] for f in flat_near if f in edge_to_pos]
