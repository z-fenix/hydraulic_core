/**
 * geometry.c — Polygon and line segment operations
 *
 * Core computational geometry ported from ANUGA:
 *   anuga/geometry/polygon.py
 *   anuga/geometry/polygon.c (C extension)
 */

#include "hydro/geometry.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* =========================================================================
 * Polygon Area (Shoelace Formula)
 * ========================================================================= */

double hydro_polygon_area(const double* polygon, hydro_int n) {
    if (n < 3) return 0.0;
    double area = 0.0;
    for (hydro_int i = 0; i < n; i++) {
        hydro_int j = (i + 1) % n;
        double xi  = polygon[2*i];
        double yi  = polygon[2*i + 1];
        double xj  = polygon[2*j];
        double yj  = polygon[2*j + 1];
        area += xi * yj - xj * yi;
    }
    return fabs(area) * 0.5;
}

/* =========================================================================
 * Point on Line Segment
 * ========================================================================= */

int hydro_point_on_line(const double* point, const double* line,
                         double rtol, double atol) {
    double px = point[0], py = point[1];
    double x0 = line[0], y0 = line[1];
    double x1 = line[2], y1 = line[3];

    double dx = x1 - x0, dy = y1 - y0;
    double len_sq = dx*dx + dy*dy;

    /* Handle degenerate line */
    if (len_sq < atol * atol) {
        double d2 = (px - x0)*(px - x0) + (py - y0)*(py - y0);
        return d2 <= atol*atol;
    }

    /* Project point onto line, compute parameter u */
    double u = ((px - x0)*dx + (py - y0)*dy) / len_sq;

    /* Check if projection lies within segment bounds */
    if (u < -rtol || u > 1.0 + rtol) return 0;

    /* Compute perpendicular distance from line */
    double proj_x = x0 + u*dx;
    double proj_y = y0 + u*dy;
    double perp_d2 = (px - proj_x)*(px - proj_x) + (py - proj_y)*(py - proj_y);

    /* Check if distance is within tolerance */
    double tol = atol + rtol * sqrt((x0*x0 + y0*y0 + x1*x1 + y1*y1) / 4.0);
    return perp_d2 <= tol*tol;
}

/* =========================================================================
 * Line Segment Intersection
 * ========================================================================= */

int hydro_line_intersection(const double* line0, const double* line1,
                             double* result) {
    double x0 = line0[0], y0 = line0[1];
    double x1 = line0[2], y1 = line0[3];
    double x2 = line1[0], y2 = line1[1];
    double x3 = line1[2], y3 = line1[3];

    double denom = (y3 - y2)*(x1 - x0) - (x3 - x2)*(y1 - y0);
    double u0 = (x3 - x2)*(y0 - y2) - (y3 - y2)*(x0 - x2);
    double u1 = (x2 - x0)*(y1 - y0) - (y2 - y0)*(x1 - x0);

    /* Tolerance for zero */
    double eps = 1e-12;

    if (fabs(denom) < eps) {
        /* Lines are parallel — check collinearity */
        if (fabs(u0) < eps && fabs(u1) < eps) {
            /* Collinear — check overlap */
            double pt[2], ln1[4];
            pt[0] = x0; pt[1] = y0;
            ln1[0] = x2; ln1[1] = y2; ln1[2] = x3; ln1[3] = y3;
            int s0 = hydro_point_on_line(pt, ln1, 1e-12, 1e-12);

            pt[0] = x1; pt[1] = y1;
            int s1 = hydro_point_on_line(pt, ln1, 1e-12, 1e-12);

            ln1[0] = x0; ln1[1] = y0; ln1[2] = x1; ln1[3] = y1;
            pt[0] = x2; pt[1] = y2;
            int s2 = hydro_point_on_line(pt, ln1, 1e-12, 1e-12);

            pt[0] = x3; pt[1] = y3;
            int s3 = hydro_point_on_line(pt, ln1, 1e-12, 1e-12);

            if (!s0 && !s1 && !s2 && !s3) return 3; /* collinear, no overlap */
            return 2; /* collinear overlapping */
        }
        return 4; /* parallel, not collinear */
    }

    /* Lines not parallel — compute intersection parameters */
    u0 /= denom;
    u1 /= denom;

    /* Check if intersection lies within both segments */
    if (u0 >= 0.0 && u0 <= 1.0 && u1 >= 0.0 && u1 <= 1.0) {
        if (result) {
            result[0] = x0 + u0*(x1 - x0);
            result[1] = y0 + u0*(y1 - y0);
        }
        return 1;
    }
    return 0; /* no intersection within segments */
}

/* =========================================================================
 * Inside Triangle Test (Barycentric Coordinates)
 * ========================================================================= */

int hydro_is_inside_triangle(const double* point, const double* triangle,
                              int closed) {
    double px = point[0], py = point[1];
    double x0 = triangle[0], y0 = triangle[1];
    double x1 = triangle[2], y1 = triangle[3];
    double x2 = triangle[4], y2 = triangle[5];

    /* Vectors */
    double v0x = x2 - x0, v0y = y2 - y0;
    double v1x = x1 - x0, v1y = y1 - y0;
    double vx  = px - x0, vy  = py - y0;

    /* Dot products */
    double a00 = v0x*v0x + v0y*v0y;
    double a01 = v0x*v1x + v0y*v1y;
    double a11 = v1x*v1x + v1y*v1y;
    double b0  = v0x*vx  + v0y*vy;
    double b1  = v1x*vx  + v1y*vy;

    double denom = a00*a11 - a01*a01;
    double eps = 1e-12;

    if (fabs(denom) < eps) return 0; /* degenerate triangle */

    double alpha = (b0*a11 - b1*a01) / denom;
    double beta  = (b1*a00 - b0*a01) / denom;

    if (closed) {
        return (alpha >= -eps && beta >= -eps && alpha + beta <= 1.0 + eps);
    } else {
        return (alpha > eps && beta > eps && alpha + beta < 1.0 - eps);
    }
}

/* =========================================================================
 * Inside Polygon Test (Ray Casting / Crossing Number)
 * ========================================================================= */

int hydro_is_inside_polygon(const double* point, const double* polygon,
                             hydro_int n, int closed) {
    double px = point[0], py = point[1];
    hydro_int inside = 0;
    int on_boundary = 0;

    for (hydro_int i = 0, j = n - 1; i < n; j = i++) {
        double xi = polygon[2*i], yi = polygon[2*i + 1];
        double xj = polygon[2*j], yj = polygon[2*j + 1];

        /* Check if point is on this edge */
        double line[4] = {xi, yi, xj, yj};
        if (hydro_point_on_line(point, line, 1e-12, 1e-12)) {
            on_boundary = 1;
            if (closed) return 1;
        }

        /* Ray casting: check if horizontal ray from point crosses edge.
         * Standard algorithm: count how many times a ray going right (+x)
         * from the point crosses an edge. Use strict inequality for y
         * to handle vertices correctly. */
        if (((yi > py) != (yj > py)) &&
            (px < (xj - xi)*(py - yi)/(yj - yi) + xi)) {
            inside = !inside;
        }
    }

    /* If point is on boundary and closed=0, return 0 (outside) */
    if (on_boundary && !closed) return 0;

    return inside;
}

/* =========================================================================
 * Separate Points by Polygon
 * ========================================================================= */

hydro_int hydro_separate_points_by_polygon(
    const double* points, hydro_int M,
    const double* polygon, hydro_int N,
    int closed, hydro_int* indices) {

    hydro_int count = 0;
    hydro_int out_idx = M - 1;
    hydro_int* temp = (hydro_int*)malloc(M * sizeof(hydro_int));

    for (hydro_int i = 0; i < M; i++) {
        int inside = hydro_is_inside_polygon(&points[2*i], polygon, N, closed);
        if (inside) {
            temp[count++] = i;
        } else {
            temp[out_idx--] = i;
        }
    }

    /* Copy to output */
    memcpy(indices, temp, M * sizeof(hydro_int));
    free(temp);
    return count;
}

/* =========================================================================
 * Polygon AABB
 * ========================================================================= */

void hydro_polygon_aabb(const double* polygon, hydro_int n,
                         double* xmin, double* xmax,
                         double* ymin, double* ymax) {
    if (n < 1) {
        *xmin = *xmax = *ymin = *ymax = 0.0;
        return;
    }
    *xmin = *xmax = polygon[0];
    *ymin = *ymax = polygon[1];
    for (hydro_int i = 1; i < n; i++) {
        double x = polygon[2*i], y = polygon[2*i + 1];
        if (x < *xmin) *xmin = x;
        if (x > *xmax) *xmax = x;
        if (y < *ymin) *ymin = y;
        if (y > *ymax) *ymax = y;
    }
}

/* =========================================================================
 * Polygon-Triangle Overlap
 * ========================================================================= */

hydro_int hydro_polygon_triangle_overlap(
    const double* polygon, hydro_int poly_n,
    const double* triangles, hydro_int M,
    hydro_int* indices) {

    hydro_int count = 0;
    for (hydro_int t = 0; t < M; t++) {
        const double* tri = &triangles[6*t];
        int overlaps = 0;

        /* Check if any triangle vertex is inside polygon */
        for (int v = 0; v < 3 && !overlaps; v++) {
            if (hydro_is_inside_polygon(&tri[2*v], polygon, poly_n, 1)) {
                overlaps = 1;
            }
        }

        /* Check if any polygon vertex is inside triangle */
        for (hydro_int p = 0; p < poly_n && !overlaps; p++) {
            if (hydro_is_inside_triangle(&polygon[2*p], tri, 1)) {
                overlaps = 1;
            }
        }

        /* Check for edge intersections */
        for (hydro_int p = 0; p < poly_n && !overlaps; p++) {
            hydro_int p_next = (p + 1) % poly_n;
            double poly_edge[4] = {
                polygon[2*p], polygon[2*p + 1],
                polygon[2*p_next], polygon[2*p_next + 1]
            };
            for (int e = 0; e < 3 && !overlaps; e++) {
                int e_next = (e + 1) % 3;
                double tri_edge[4] = {
                    tri[2*e], tri[2*e + 1],
                    tri[2*e_next], tri[2*e_next + 1]
                };
                if (hydro_line_intersection(poly_edge, tri_edge, NULL) == 1) {
                    overlaps = 1;
                }
            }
        }

        if (overlaps) {
            indices[count++] = t;
        }
    }
    return count;
}

/* =========================================================================
 * AABB Helpers
 * ========================================================================= */

void hydro_aabb_init(hydro_aabb_t* box) {
    box->xmin = box->ymin = 1e100;
    box->xmax = box->ymax = -1e100;
}

void hydro_aabb_extend(hydro_aabb_t* box, double x, double y) {
    if (x < box->xmin) box->xmin = x;
    if (x > box->xmax) box->xmax = x;
    if (y < box->ymin) box->ymin = y;
    if (y > box->ymax) box->ymax = y;
}

int hydro_aabb_contains(const hydro_aabb_t* box, double x, double y) {
    return (x >= box->xmin && x <= box->xmax &&
            y >= box->ymin && y <= box->ymax);
}
