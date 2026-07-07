#ifndef HYDRO_MESH_H
#define HYDRO_MESH_H

#include "types.h"
#include "domain.h"

#ifdef __cplusplus
extern "C" {

#endif

/**
 * Compute mesh geometry from vertex coordinates and triangle connectivity.
 *
 * Sets: vertex_coordinates (flattened per-triangle), edge_coordinates,
 *       centroid_coordinates, normals, edgelengths, areas, radii.
 *
 * Assumes CCW vertex ordering for each triangle.
 *
 * @param domain  The domain (mesh arrays must be pre-allocated)
 * @return 0 on success
 */
int hydro_mesh_compute_geometry(hydro_domain_t* domain);

/**
 * Build the neighbour structure for the mesh.
 *
 * Populates: neighbours, neighbour_edges, surrogate_neighbours,
 *            number_of_boundaries, boundary_length.
 *
 * Uses a hash table on edge node pairs to find shared edges between triangles.
 *
 * @param domain  The domain (triangles must be set)
 * @return 0 on success
 */
int hydro_mesh_build_neighbour_structure(hydro_domain_t* domain);

/**
 * Build the boundary mapping from the neighbour structure.
 *
 * Populates: boundary_tags, boundary_edges arrays.
 * Requires neighbours and boundary_tags_map to be set.
 *
 * @param domain  The domain
 * @return 0 on success
 */
int hydro_mesh_build_boundary_structure(hydro_domain_t* domain);

#ifdef __cplusplus
}
#endif

#endif /* HYDRO_MESH_H */
