"""
Rectangular cross-domain mesh builder.

Produces the same mesh topology as anuga.rectangular_cross_domain():
each rectangular cell is split into 4 CCW triangles around its centroid.
"""

import numpy as np


def rectangular_cross_domain(nx: int, ny: int, len1: float = 1.0, len2: float = 1.0):
    """Build a rectangular cross mesh.

    Parameters
    ----------
    nx : int
        Number of cells in the x-direction.
    ny : int
        Number of cells in the y-direction.
    len1 : float
        Total domain length in x (metres).
    len2 : float
        Total domain width in y (metres).

    Returns
    -------
    vertices : ndarray (N, 2) float64
        Unique vertex coordinates.
    triangles : ndarray (M, 3) int64
        Triangle vertex indices (CCW order).
    boundary_edges : ndarray (B,) int64
        Flat edge indices of boundary edges.
    boundary_tags : ndarray (B,) int64
        Tag for each boundary edge: 1=left, 2=right, 3=top, 4=bottom.
    """
    dx = len1 / nx
    dy = len2 / ny

    # Grid nodes
    nodes = []
    node_idx = {}
    for iy in range(ny + 1):
        for ix in range(nx + 1):
            node_idx[(ix, iy)] = len(nodes)
            nodes.append([ix * dx, iy * dy])

    # Centroids (one per cell)
    centroids = []
    cent_idx = {}
    for cy in range(ny):
        for cx in range(nx):
            cent_idx[(cx, cy)] = len(nodes) + len(centroids)
            centroids.append([(cx + 0.5) * dx, (cy + 0.5) * dy])

    all_verts = np.array(nodes + centroids, dtype=np.float64)

    # CCW triangles: [corner1, centroid, corner2]
    triangles = []
    for cy in range(ny):
        for cx in range(nx):
            nw = node_idx[(cx, cy)]
            ne = node_idx[(cx + 1, cy)]
            sw = node_idx[(cx, cy + 1)]
            se = node_idx[(cx + 1, cy + 1)]
            c = cent_idx[(cx, cy)]
            triangles.append([nw, c, sw])
            triangles.append([ne, c, nw])
            triangles.append([se, c, ne])
            triangles.append([sw, c, se])

    tri = np.array(triangles, dtype=np.int64)

    # Boundary edges
    boundary_edges = []
    boundary_tags = []
    for ti, t in enumerate(tri):
        for ei in range(3):
            v1 = t[ei]
            v2 = t[(ei + 1) % 3]
            x1, y1 = all_verts[v1]
            x2, y2 = all_verts[v2]
            # Map to C edge numbering: edge 0=(v1,v2), 1=(v2,v0), 2=(v0,v1)
            c_ei = 2 if ei == 0 else (0 if ei == 1 else 1)
            flat_edge = 3 * ti + c_ei

            if abs(x1) < 1e-10 and abs(x2) < 1e-10:
                boundary_edges.append(flat_edge)
                boundary_tags.append(1)  # left
            elif abs(x1 - len1) < 1e-10 and abs(x2 - len1) < 1e-10:
                boundary_edges.append(flat_edge)
                boundary_tags.append(2)  # right
            elif abs(y1 - len2) < 1e-10 and abs(y2 - len2) < 1e-10:
                boundary_edges.append(flat_edge)
                boundary_tags.append(3)  # top
            elif abs(y1) < 1e-10 and abs(y2) < 1e-10:
                boundary_edges.append(flat_edge)
                boundary_tags.append(4)  # bottom

    bedges = np.array(boundary_edges, dtype=np.int64)
    btags = np.array(boundary_tags, dtype=np.int64)

    return all_verts, tri, bedges, btags
