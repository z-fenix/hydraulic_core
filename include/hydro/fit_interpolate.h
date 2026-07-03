#ifndef HYDRO_FIT_INTERPOLATE_H
#define HYDRO_FIT_INTERPOLATE_H

#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* =========================================================================
 * Search and Interpolation on Triangular Meshes
 *
 * Provides:
 *   1. Point-in-triangle search using barycentric coordinates
 *   2. Interpolation of vertex-defined values to arbitrary points
 *   3. Rectangular grid interpolation (bilinear and nearest-neighbour)
 * ========================================================================= */

/* Search result: containing triangle and barycentric weights */
typedef struct {
    hydro_int triangle_index;  /* index of containing triangle, -1 if not found */
    double sigma[3];           /* barycentric coordinates (sum to 1) */
    hydro_int vertices[3];     /* vertex indices of the containing triangle */
} hydro_interp_result_t;

/* =========================================================================
 * Mesh-based Interpolation
 * ========================================================================= */

/* Find the triangle containing a point using linear search + barycentric test.
 *
 * point: [x, y] query point
 * vertex_coords: flat array [x0,y0, x1,y1, ...] of vertex coordinates
 * triangles: flat array [v0,v1,v2, ...] of triangle vertex indices
 * N_triangles: number of triangles
 * start_triangle: hint for which triangle to start search (-1 for no hint)
 *
 * Returns a hydro_interp_result_t with the containing triangle info.
 * If no triangle contains the point, triangle_index is -1.
 */
hydro_interp_result_t hydro_find_containing_triangle(
    const double* point,
    const double* vertex_coords,
    const hydro_int* triangles,
    hydro_int N_triangles,
    hydro_int start_triangle);

/* Interpolate scalar values defined at mesh vertices to an arbitrary point.
 *
 * vertex_values: scalar values at each vertex, length N_vertices
 * vertex_coords: vertex coordinates, length 2*N_vertices
 * triangles: triangle vertex indices, length 3*N_triangles
 * N_triangles: number of triangles
 * point: [x, y] query point
 * fill_value: returned value if point is outside the mesh
 *
 * Returns interpolated value at the point.
 */
double hydro_interpolate_at_point(
    const double* vertex_values,
    const double* vertex_coords,
    const hydro_int* triangles,
    hydro_int N_triangles,
    const double* point,
    double fill_value);

/* Interpolate at multiple points.
 *
 * vertex_values, vertex_coords, triangles, N_triangles: mesh data
 * points: flat array [x0,y0, x1,y1, ...], length 2*N_points
 * N_points: number of query points
 * output: array of length N_points, filled with interpolated values
 * fill_value: used for points outside the mesh
 */
void hydro_interpolate_batch(
    const double* vertex_values,
    const double* vertex_coords,
    const hydro_int* triangles,
    hydro_int N_triangles,
    const double* points,
    hydro_int N_points,
    double* output,
    double fill_value);

/* =========================================================================
 * Rectangular Grid (Bilinear / Nearest-Neighbour) Interpolation
 * ========================================================================= */

/* Interpolate from a regular 2D grid to arbitrary points using bilinear
 * or nearest-neighbour method.
 *
 * x: array of grid x-coordinates, length nx (must be sorted ascending)
 * y: array of grid y-coordinates, length ny (must be sorted ascending)
 * Z: 2D array of values, row-major: Z[iy*nx + ix] for (x[ix], y[iy])
 * nx, ny: grid dimensions
 * points: flat array [px0,py0, px1,py1, ...], length 2*N
 * N: number of query points
 * output: array of length N, filled with interpolated values
 * mode: 0 = nearest-neighbour (piecewise constant), 1 = bilinear
 * fill_value: used for points outside the grid extent
 */
void hydro_interpolate_regular_grid(
    const double* x, hydro_int nx,
    const double* y, hydro_int ny,
    const double* Z,
    const double* points, hydro_int N,
    double* output,
    int mode,
    double fill_value);

#ifdef __cplusplus
}
#endif

#endif /* HYDRO_FIT_INTERPOLATE_H */
