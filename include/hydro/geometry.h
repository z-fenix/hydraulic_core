#ifndef HYDRO_GEOMETRY_H
#define HYDRO_GEOMETRY_H

#include "types.h"

#ifdef __cplusplus
extern "C" {

#endif

/* =========================================================================
 * Polygon Operations
 * ========================================================================= */

/* Compute area of a polygon using the shoelace formula.
 * polygon: flat array [x0,y0, x1,y1, ... x_{n-1},y_{n-1}]
 * n: number of vertices
 * Returns absolute area (always positive).
 */
double hydro_polygon_area(const double* polygon, hydro_int n);

/* Determine whether a point is inside a triangle using barycentric
 * coordinates.
 * point: [x, y]
 * triangle: [x0,y0, x1,y1, x2,y2]
 * closed: if non-zero, points exactly on the boundary are "inside"
 * Returns 1 if inside, 0 otherwise.
 */
int hydro_is_inside_triangle(const double* point, const double* triangle,
                             int closed);

/* Determine whether one point is inside a polygon using the ray-casting
 * (crossing number) algorithm.
 * point: [x, y]
 * polygon: flat array of vertices [x0,y0, ...]
 * n: number of vertices
 * closed: if non-zero, points on the boundary are "inside"
 * Returns 1 if inside, 0 otherwise.
 */
int hydro_is_inside_polygon(const double* point, const double* polygon,
                            hydro_int n, int closed);

/* Separate an array of points into those inside and outside a polygon.
 * points: flat array [x0,y0, x1,y1, ..., x_{M-1},y_{M-1}]
 * M: number of points
 * polygon: flat array of polygon vertices [x0,y0, ...]
 * N: number of polygon vertices
 * closed: boundary treatment
 * indices: output array of length M, populated with point indices.
 *   First `count` entries are indices of points INSIDE the polygon.
 *   Remaining entries are indices of points OUTSIDE.
 * Returns count: number of points inside the polygon.
 */
hydro_int hydro_separate_points_by_polygon(
    const double* points, hydro_int M,
    const double* polygon, hydro_int N,
    int closed, hydro_int* indices);

/* Compute axis-aligned bounding box of a polygon.
 * polygon: flat array [x0,y0, ..., x_{n-1},y_{n-1}]
 * n: number of vertices
 * xmin, xmax, ymin, ymax: output bounding box
 */
void hydro_polygon_aabb(const double* polygon, hydro_int n,
                        double* xmin, double* xmax,
                        double* ymin, double* ymax);

/* =========================================================================
 * Line segment operations
 * ========================================================================= */

/* Determine if a point lies on a line segment.
 * point: [px, py]
 * line: [x0,y0, x1,y1]
 * rtol, atol: relative and absolute tolerance
 * Returns 1 if on line, 0 otherwise.
 */
int hydro_point_on_line(const double* point, const double* line,
                        double rtol, double atol);

/* Intersect two line segments.
 * line0: [x0,y0, x1,y1]
 * line1: [x2,y2, x3,y3]
 * result: output [x,y] of intersection (only valid if status==1)
 * Returns status:
 *   0: no intersection
 *   1: single intersection point
 *   2: collinear overlapping
 *   3: collinear non-overlapping
 *   4: parallel non-collinear
 */
int hydro_line_intersection(const double* line0, const double* line1,
                            double* result);

/* =========================================================================
 * Triangle operations
 * ========================================================================= */

/* Check if a polygon and a set of triangles overlap.
 * polygon: flat array of polygon vertices
 * poly_n: number of polygon vertices
 * triangles: flat array [x0,y0, x1,y1, x2,y2, ...] for M triangles
 * M: number of triangles
 * indices: output array of length M, filled with overlapping triangle indices
 * Returns count of overlapping triangles.
 */
hydro_int hydro_polygon_triangle_overlap(
    const double* polygon, hydro_int poly_n,
    const double* triangles, hydro_int M,
    hydro_int* indices);

/* =========================================================================
 * AABB (Axis-Aligned Bounding Box)
 * ========================================================================= */

typedef struct
{
    double xmin, xmax, ymin, ymax;
} hydro_aabb_t;

void hydro_aabb_init(hydro_aabb_t* box);
void hydro_aabb_extend(hydro_aabb_t* box, double x, double y);
int hydro_aabb_contains(const hydro_aabb_t* box, double x, double y);

#ifdef __cplusplus
}
#endif

#endif /* HYDRO_GEOMETRY_H */
