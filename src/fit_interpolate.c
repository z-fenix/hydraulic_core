/**
 * fit_interpolate.c — Search and interpolation on triangular meshes
 *
 * Ported from ANUGA:
 *   anuga/fit_interpolate/interpolate.py
 *   anuga/fit_interpolate/interpolate2d.py
 *   anuga/fit_interpolate/fit.py
 */

#include "hydro/fit_interpolate.h"
#include "hydro/geometry.h"
#include <math.h>
#include <stdlib.h>

/* =========================================================================
 * Find Containing Triangle (Linear Search)
 * ========================================================================= */

hydro_interp_result_t hydro_find_containing_triangle(
    const double* point,
    const double* vertex_coords,
    const hydro_int* triangles,
    hydro_int N_triangles,
    hydro_int start_triangle) {

    hydro_interp_result_t result;
    result.triangle_index = -1;
    result.sigma[0] = result.sigma[1] = result.sigma[2] = 0.0;
    result.vertices[0] = result.vertices[1] = result.vertices[2] = -1;

    double px = point[0], py = point[1];

    /* Try the hint triangle first */
    if (start_triangle >= 0 && start_triangle < N_triangles) {
        hydro_int k3 = 3 * start_triangle;
        hydro_int v0 = triangles[k3], v1 = triangles[k3+1], v2 = triangles[k3+2];
        double tri[6] = {
            vertex_coords[2*v0], vertex_coords[2*v0+1],
            vertex_coords[2*v1], vertex_coords[2*v1+1],
            vertex_coords[2*v2], vertex_coords[2*v2+1]
        };
        if (hydro_is_inside_triangle(point, tri, 1)) {
            result.triangle_index = start_triangle;
            result.vertices[0] = v0;
            result.vertices[1] = v1;
            result.vertices[2] = v2;

            /* Compute barycentric coordinates */
            double x0 = tri[0], y0 = tri[1];
            double x1 = tri[2], y1 = tri[3];
            double x2 = tri[4], y2 = tri[5];

            double det = (x1 - x0)*(y2 - y0) - (x2 - x0)*(y1 - y0);
            if (fabs(det) > 1e-30) {
                result.sigma[0] = ((y1 - y2)*(px - x2) + (x2 - x1)*(py - y2)) / det;
                result.sigma[1] = ((y2 - y0)*(px - x0) + (x0 - x2)*(py - y0)) / det;
                result.sigma[2] = 1.0 - result.sigma[0] - result.sigma[1];
            }
            return result;
        }
    }

    /* Linear search through all triangles */
    for (hydro_int t = 0; t < N_triangles; t++) {
        hydro_int k3 = 3 * t;
        hydro_int v0 = triangles[k3], v1 = triangles[k3+1], v2 = triangles[k3+2];
        double tri[6] = {
            vertex_coords[2*v0], vertex_coords[2*v0+1],
            vertex_coords[2*v1], vertex_coords[2*v1+1],
            vertex_coords[2*v2], vertex_coords[2*v2+1]
        };
        if (hydro_is_inside_triangle(point, tri, 1)) {
            result.triangle_index = t;
            result.vertices[0] = v0;
            result.vertices[1] = v1;
            result.vertices[2] = v2;

            double x0 = tri[0], y0 = tri[1];
            double x1 = tri[2], y1 = tri[3];
            double x2 = tri[4], y2 = tri[5];

            double det = (x1 - x0)*(y2 - y0) - (x2 - x0)*(y1 - y0);
            if (fabs(det) > 1e-30) {
                result.sigma[0] = ((y1 - y2)*(px - x2) + (x2 - x1)*(py - y2)) / det;
                result.sigma[1] = ((y2 - y0)*(px - x0) + (x0 - x2)*(py - y0)) / det;
                result.sigma[2] = 1.0 - result.sigma[0] - result.sigma[1];
            }
            return result;
        }
    }

    return result;  /* not found */
}

/* =========================================================================
 * Single Point Interpolation
 * ========================================================================= */

double hydro_interpolate_at_point(
    const double* vertex_values,
    const double* vertex_coords,
    const hydro_int* triangles,
    hydro_int N_triangles,
    const double* point,
    double fill_value) {

    hydro_interp_result_t res = hydro_find_containing_triangle(
        point, vertex_coords, triangles, N_triangles, -1);

    if (res.triangle_index < 0) return fill_value;

    /* Interpolate: sum(sigma[i] * value[vertex[i]]) */
    double val = res.sigma[0] * vertex_values[res.vertices[0]] +
                 res.sigma[1] * vertex_values[res.vertices[1]] +
                 res.sigma[2] * vertex_values[res.vertices[2]];

    return val;
}

/* =========================================================================
 * Batch Interpolation
 * ========================================================================= */

void hydro_interpolate_batch(
    const double* vertex_values,
    const double* vertex_coords,
    const hydro_int* triangles,
    hydro_int N_triangles,
    const double* points,
    hydro_int N_points,
    double* output,
    double fill_value) {

    hydro_int last_tri = -1;

    for (hydro_int p = 0; p < N_points; p++) {
        const double* point = &points[2*p];
        hydro_interp_result_t res = hydro_find_containing_triangle(
            point, vertex_coords, triangles, N_triangles, last_tri);

        if (res.triangle_index >= 0) {
            last_tri = res.triangle_index;
            output[p] = res.sigma[0] * vertex_values[res.vertices[0]] +
                        res.sigma[1] * vertex_values[res.vertices[1]] +
                        res.sigma[2] * vertex_values[res.vertices[2]];
        } else {
            output[p] = fill_value;
        }
    }
}

/* =========================================================================
 * Regular Grid Interpolation (Bilinear / Nearest)
 * ========================================================================= */

/* Binary search for the rightmost x <= val in a sorted array */
static hydro_int find_upper_index(const double* arr, hydro_int n, double val) {
    if (val < arr[0]) return -1;
    if (val > arr[n-1]) return n;
    hydro_int lo = 0, hi = n - 1;
    while (lo <= hi) {
        hydro_int mid = (lo + hi) / 2;
        if (arr[mid] <= val && (mid == n-1 || arr[mid+1] > val)) {
            return mid;
        }
        if (arr[mid] < val) lo = mid + 1;
        else hi = mid - 1;
    }
    return lo;
}

void hydro_interpolate_regular_grid(
    const double* x, hydro_int nx,
    const double* y, hydro_int ny,
    const double* Z,
    const double* points, hydro_int N,
    double* output,
    int mode,
    double fill_value) {

    if (nx < 1 || ny < 1 || !Z) {
        for (hydro_int i = 0; i < N; i++) output[i] = fill_value;
        return;
    }

    for (hydro_int p = 0; p < N; p++) {
        double px = points[2*p], py = points[2*p+1];

        /* Find grid cell indices */
        hydro_int ix_lo = find_upper_index(x, nx, px);
        hydro_int iy_lo = find_upper_index(y, ny, py);

        /* Outside grid extent? */
        if (ix_lo < 0 || ix_lo >= nx-1 || iy_lo < 0 || iy_lo >= ny-1) {
            /* Handle boundary exactly */
            if (ix_lo == -1 && px >= x[0]) ix_lo = 0;
            if (iy_lo == -1 && py >= y[0]) iy_lo = 0;

            if (ix_lo < 0 || iy_lo < 0 || ix_lo >= nx || iy_lo >= ny) {
                output[p] = fill_value;
                continue;
            }

            /* On boundary — use nearest cell */
            if (ix_lo >= nx-1) ix_lo = nx-2;
            if (iy_lo >= ny-1) iy_lo = ny-2;
            if (ix_lo < 0) ix_lo = 0;
            if (iy_lo < 0) iy_lo = 0;

            double z00 = Z[iy_lo*nx + ix_lo];
            double z10 = Z[iy_lo*nx + ix_lo+1];
            double z01 = Z[(iy_lo+1)*nx + ix_lo];
            double z11 = Z[(iy_lo+1)*nx + ix_lo+1];

            if (mode == 0) {
                /* Nearest-neighbour: find closest corner */
                double d00 = (px-x[ix_lo])*(px-x[ix_lo]) + (py-y[iy_lo])*(py-y[iy_lo]);
                double d10 = (px-x[ix_lo+1])*(px-x[ix_lo+1]) + (py-y[iy_lo])*(py-y[iy_lo]);
                double d01 = (px-x[ix_lo])*(px-x[ix_lo]) + (py-y[iy_lo+1])*(py-y[iy_lo+1]);
                double d11 = (px-x[ix_lo+1])*(px-x[ix_lo+1]) + (py-y[iy_lo+1])*(py-y[iy_lo+1]);
                double dmin = d00;
                output[p] = z00;
                if (d10 < dmin) { dmin = d10; output[p] = z10; }
                if (d01 < dmin) { dmin = d01; output[p] = z01; }
                if (d11 < dmin) { output[p] = z11; }
            } else {
                /* Bilinear */
                double alpha = (px - x[ix_lo]) / (x[ix_lo+1] - x[ix_lo]);
                double beta  = (py - y[iy_lo]) / (y[iy_lo+1] - y[iy_lo]);
                output[p] = (1-alpha)*(1-beta)*z00 + alpha*(1-beta)*z10 +
                            (1-alpha)*beta*z01 + alpha*beta*z11;
            }
            continue;
        }

        /* Cell corners */
        double z00 = Z[iy_lo*nx + ix_lo];
        double z10 = Z[iy_lo*nx + ix_lo+1];
        double z01 = Z[(iy_lo+1)*nx + ix_lo];
        double z11 = Z[(iy_lo+1)*nx + ix_lo+1];

        if (mode == 0) {
            /* Nearest-neighbour */
            double d00 = (px-x[ix_lo])*(px-x[ix_lo]) + (py-y[iy_lo])*(py-y[iy_lo]);
            double d10 = (px-x[ix_lo+1])*(px-x[ix_lo+1]) + (py-y[iy_lo])*(py-y[iy_lo]);
            double d01 = (px-x[ix_lo])*(px-x[ix_lo]) + (py-y[iy_lo+1])*(py-y[iy_lo+1]);
            double d11 = (px-x[ix_lo+1])*(px-x[ix_lo+1]) + (py-y[iy_lo+1])*(py-y[iy_lo+1]);
            double dmin = d00;
            output[p] = z00;
            if (d10 < dmin) { dmin = d10; output[p] = z10; }
            if (d01 < dmin) { dmin = d01; output[p] = z01; }
            if (d11 < dmin) { output[p] = z11; }
        } else {
            /* Bilinear interpolation */
            double alpha = (px - x[ix_lo]) / (x[ix_lo+1] - x[ix_lo]);
            double beta  = (py - y[iy_lo]) / (y[iy_lo+1] - y[iy_lo]);
            output[p] = (1-alpha)*(1-beta)*z00 + alpha*(1-beta)*z10 +
                        (1-alpha)*beta*z01 + alpha*beta*z11;
        }
    }
}
