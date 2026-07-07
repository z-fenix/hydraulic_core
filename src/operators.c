/**
 * operators.c — Physical operators
 *
 * Ported from ANUGA:
 *   anuga/operators/kinematic_viscosity_operator.c  (C extension)
 *   anuga/operators/kinematic_viscosity_operator.py (Python wrapper)
 *   anuga/operators/mannings_operator.py
 *   anuga/operators/erosion_operators.py
 *   anuga/operators/set_stage_operator.py
 *   anuga/operators/set_elevation_operator.py
 */

#include "hydro/operators.h"
#include "hydro/quantity.h"
#include "hydro/solver.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* =========================================================================
 * Quicksort (needed by matrix builder, sorts column indices per row)
 * ========================================================================= */

static void swap_int(hydro_int* a, hydro_int* b)
{
    hydro_int t = *a;
    *a = *b;
    *b = t;
}

static hydro_int choose_pivot(hydro_int i, hydro_int j) { return (i + j) / 2; }

static void quicksort(hydro_int* list, hydro_int m, hydro_int n)
{
    if (m >= n) return;
    hydro_int key, i = m, j = n;
    hydro_int k = choose_pivot(m, n);
    swap_int(&list[m], &list[k]);
    key = list[m];
    i = m + 1;
    j = n;
    while (i <= j)
    {
        while (i <= n && list[i] <= key) i++;
        while (j >= m && list[j] > key) j--;
        if (i < j) swap_int(&list[i], &list[j]);
    }
    swap_int(&list[m], &list[j]);
    quicksort(list, m, j - 1);
    quicksort(list, j + 1, n);
}

/* =========================================================================
 * Manning Friction (Explicit)
 * ========================================================================= */

void hydro_manning_friction_explicit(hydro_domain_t* domain)
{
    hydro_int N = domain->number_of_elements;
    double g = domain->g, eps = domain->minimum_allowed_height;
    double dt = domain->timestep;
    double* stage_c = domain->stage_centroid_values;
    double* bed_c = domain->bed_centroid_values;
    double* xmom_c = domain->xmom_centroid_values;
    double* ymom_c = domain->ymom_centroid_values;
    double* fric_c = domain->friction_centroid_values;
    if (!xmom_c || !fric_c) return;

    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (hydro_int k = 0; k < N; k++) {
        double h = stage_c[k] - bed_c[k];
        if (h < eps) continue;
        double eta = fric_c[k];
        if (eta < 1e-15) continue;
        double uh = xmom_c[k], vh = ymom_c[k];
        double speed = sqrt(uh * uh + vh * vh) / h;
        double gamma = -g * eta * eta * speed / pow(h, 4.0 / 3.0);
        double eg = exp(gamma * dt);
        xmom_c[k] *= eg;
        ymom_c[k] *= eg;
    }
}

/* =========================================================================
 * Bed Shear Erosion
 * ========================================================================= */

void hydro_bed_shear_erosion_apply(hydro_domain_t* domain,
                                   double threshold, double base, const hydro_int* indices, hydro_int num)
{
    double dt = domain->timestep;
    double *sc = domain->stage_centroid_values, *bc = domain->bed_centroid_values;
    double *xc = domain->xmom_centroid_values, *yc = domain->ymom_centroid_values;
    if (!indices)
    {
        hydro_int N = domain->number_of_elements;
        #ifdef _OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (hydro_int k = 0; k < N; k++) {
            double m = sqrt(xc[k]*xc[k] + yc[k]*yc[k]);
            if (m <= threshold) continue;
            double de = m * dt, ne = bc[k] - de;
            if (ne < base) ne = base;
            double h = sc[k] - bc[k];
            bc[k] = ne;
            sc[k] = ne + h;
        }
    } else {
        #ifdef _OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (hydro_int j = 0; j < num; j++) {
            hydro_int k = indices[j];
            double m = sqrt(xc[k] * xc[k] + yc[k] * yc[k]);
            if (m <= threshold) continue;
            double de = m * dt, ne = bc[k] - de;
            if (ne < base) ne = base;
            double h = sc[k] - bc[k];
            bc[k] = ne;
            sc[k] = ne + h;
        }
    }
}

/* =========================================================================
 * Set Stage / Elevation
 * ========================================================================= */

void hydro_set_stage(hydro_domain_t* d, const double* vals,
                     const hydro_int* ind, hydro_int num)
{
    double *sc = d->stage_centroid_values, *bc = d->bed_centroid_values;
    if (!ind)
    {
        for (hydro_int k = 0; k < d->number_of_elements; k++)
        {
            sc[k] = vals[k];
            if (sc[k] < bc[k]) sc[k] = bc[k];
        }
    }
    else
    {
        for (hydro_int j = 0; j < num; j++)
        {
            hydro_int k = ind[j];
            sc[k] = vals[j];
            if (sc[k] < bc[k]) sc[k] = bc[k];
        }
    }
}

void hydro_set_elevation(hydro_domain_t* d, const double* vals,
                         const hydro_int* ind, hydro_int num)
{
    double *sc = d->stage_centroid_values, *bc = d->bed_centroid_values;
    if (!ind)
    {
        for (hydro_int k = 0; k < d->number_of_elements; k++)
        {
            bc[k] = vals[k];
            if (sc[k] < bc[k]) sc[k] = bc[k];
        }
    }
    else
    {
        for (hydro_int j = 0; j < num; j++)
        {
            hydro_int k = ind[j];
            bc[k] = vals[j];
            if (sc[k] < bc[k]) sc[k] = bc[k];
        }
    }
}

/* =========================================================================
 * Kinematic Viscosity — Geo Structure
 *
 * For each triangle edge, compute:
 *   geo_value = -edgelength / distance
 * where distance = |centroid_i - centroid_j| for interior edges,
 *                  |centroid_i - edge_midpoint| for boundary edges.
 * ========================================================================= */

int hydro_kinematic_viscosity_build_geo_structure(hydro_domain_t* d)
{
    hydro_int N = d->number_of_elements;
    hydro_int n_edges = d->number_of_edges;

    if (!d->geo_structure_indices)
    {
        d->geo_structure_indices = (hydro_int*)calloc((size_t)n_edges, sizeof(hydro_int));
    }
    if (!d->geo_structure_values)
    {
        d->geo_structure_values = (double*)calloc((size_t)n_edges, sizeof(double));
    }
    if (!d->geo_structure_indices || !d->geo_structure_values) return -1;

    double* cx = d->centroid_coordinates;
    double* edgelen = d->edgelengths;
    double* edge_coords = d->edge_coordinates;
    hydro_int* neighbours = d->neighbours;

    for (hydro_int i = 0; i < N; i++)
    {
        double this_x = cx[2 * i], this_y = cx[2 * i + 1];

        for (hydro_int edge = 0; edge < 3; edge++)
        {
            hydro_int ei = 3 * i + edge;
            hydro_int j = neighbours[ei];

            double other_x, other_y;
            if (j < 0)
            {
                /* Boundary: use edge midpoint */
                hydro_int bnd_idx = -j - 1;
                d->geo_structure_indices[ei] = N + bnd_idx;
                other_x = edge_coords[2 * ei];
                other_y = edge_coords[2 * ei + 1];
            }
            else
            {
                d->geo_structure_indices[ei] = j;
                other_x = cx[2 * j];
                other_y = cx[2 * j + 1];
            }

            double elen = edgelen[ei];
            double dx = this_x - other_x, dy = this_y - other_y;
            double dist = sqrt(dx * dx + dy * dy);
            d->geo_structure_values[ei] = (dist > 1e-30) ? -elen / dist : 0.0;
        }
    }
    return 0;
}

/* =========================================================================
 * Kinematic Viscosity — Build Elliptic Matrix
 *
 * Builds the matrix L representing div(h grad).
 * Each row has exactly 4 non-zeros (diagonal + up to 3 neighbours).
 * ========================================================================= */

hydro_sparse_csr_t* hydro_kinematic_viscosity_build_matrix(
    hydro_domain_t* d, const double* h, const double* h_b)
{
    hydro_int n = d->number_of_elements;
    hydro_int blen = d->boundary_length;
    hydro_int tot_len = n + blen;

    /* Build geo structure if not already built */
    if (!d->geo_structure_indices || !d->geo_structure_values)
    {
        if (hydro_kinematic_viscosity_build_geo_structure(d) != 0) return NULL;
    }

    hydro_sparse_csr_t* L = hydro_sparse_csr_create(n, tot_len, 4);
    if (!L) return NULL;

    double* data = L->data;
    hydro_int* colind = L->colind;
    hydro_int* geo_idx = d->geo_structure_indices;
    double* geo_val = d->geo_structure_values;

    for (hydro_int i = 0; i < n; i++)
    {
        hydro_int j[4];
        double v[3], v_i = 0.0;
        j[3] = i;

        for (int edge = 0; edge < 3; edge++)
        {
            hydro_int ei = 3 * i + edge;
            j[edge] = geo_idx[ei];

            double h_j;
            if (j[edge] < n) h_j = h[j[edge]];
            else h_j = h_b[j[edge] - n];

            /* Symmetric: v_edge = -0.5*(h_i + h_j) * geo_val */
            v[edge] = -0.5 * (h[i] + h_j) * geo_val[ei];
            v_i += 0.5 * (h[i] + h_j) * geo_val[ei];
        }

        /* Dry cell: zero out */
        if (h[i] <= 0.0)
        {
            v_i = 0.0;
            v[0] = 0.0;
            v[1] = 0.0;
            v[2] = 0.0;
        }

        /* Sort column indices to keep CSR in order */
        hydro_int sorted[4];
        for (int k = 0; k < 4; k++) sorted[k] = j[k];
        quicksort(sorted, 0, 3);

        for (int k = 0; k < 4; k++)
        {
            hydro_int idx = sorted[k];
            if (idx == i) data[4 * i + k] = v_i;
            else if (idx == j[0]) data[4 * i + k] = v[0];
            else if (idx == j[1]) data[4 * i + k] = v[1];
            else data[4 * i + k] = v[2];
            colind[4 * i + k] = idx;
        }
    }

    return L;
}

int hydro_kinematic_viscosity_update_matrix(
    hydro_domain_t* d, const double* h, const double* h_b,
    hydro_sparse_csr_t* L)
{
    hydro_int n = d->number_of_elements;

    if (!d->geo_structure_indices || !d->geo_structure_values) return -1;
    if (!L || L->N != n) return -1;

    double* data = L->data;
    hydro_int* colind = L->colind;
    hydro_int* geo_idx = d->geo_structure_indices;
    double* geo_val = d->geo_structure_values;

    for (hydro_int i = 0; i < n; i++)
    {
        hydro_int j[4];
        double v[3], v_i = 0.0;
        j[3] = i;

        for (int edge = 0; edge < 3; edge++)
        {
            hydro_int ei = 3 * i + edge;
            j[edge] = geo_idx[ei];

            double h_j;
            if (j[edge] < n) h_j = h[j[edge]];
            else h_j = h_b[j[edge] - n];

            v[edge] = -0.5 * (h[i] + h_j) * geo_val[ei];
            v_i += 0.5 * (h[i] + h_j) * geo_val[ei];
        }

        if (h[i] <= 0.0)
        {
            v_i = 0.0;
            v[0] = 0.0;
            v[1] = 0.0;
            v[2] = 0.0;
        }

        hydro_int sorted[4];
        for (int k = 0; k < 4; k++) sorted[k] = j[k];
        quicksort(sorted, 0, 3);

        for (int k = 0; k < 4; k++)
        {
            hydro_int idx = sorted[k];
            if (idx == i) data[4 * i + k] = v_i;
            else if (idx == j[0]) data[4 * i + k] = v[0];
            else if (idx == j[1]) data[4 * i + k] = v[1];
            else data[4 * i + k] = v[2];
        }
    }
    return 0;
}

/* =========================================================================
 * Kinematic Viscosity — Full Apply
 *
 * Solves (I - dt * L) * u_new = u_old for x and y velocity.
 * ========================================================================= */

int hydro_kinematic_viscosity_apply(
    hydro_domain_t* domain, const double* diffusivity, double dt)
{
    hydro_int n = domain->number_of_elements;
    if (n < 1 || !diffusivity) return -1;
    if (dt <= 0) return 0;

    /* Ensure x/y velocity are computed */
    hydro_quantity_update_derived(domain);

    double* u = domain->xvelocity_centroid_values;
    double* v = domain->yvelocity_centroid_values;
    double* h = domain->height_centroid_values;

    /* Boundary diffusivity: use the domain height boundary values */
    hydro_int bl = domain->boundary_length;
    double* h_bnd = NULL;
    if (bl > 0)
    {
        h_bnd = (double*)calloc((size_t)bl, sizeof(double));
        if (!h_bnd) return -1;
        /* Extrapolate first order to populate boundary values */
        hydro_quantity_extrapolate_first_order(domain);
        for (hydro_int bi = 0; bi < bl; bi++)
        {
            h_bnd[bi] = domain->height_boundary_values[bi];
        }
    }

    /* Build or update the elliptic matrix */
    hydro_sparse_csr_t* L = NULL;

    /* Use the actual diffusivity, not just height */
    double* diff_bnd = h_bnd;
    const double* diff = diffusivity;

    /* Build matrix fresh (for simplicity; caching could be added) */
    L = hydro_kinematic_viscosity_build_matrix(domain, diff, diff_bnd ? diff_bnd : diff);

    if (!L)
    {
        free(h_bnd);
        return -1;
    }

    /* Allocate work arrays for CG solve */
    double* u_copy = (double*)calloc((size_t)n, sizeof(double));
    double* v_copy = (double*)calloc((size_t)n, sizeof(double));
    if (!u_copy || !v_copy)
    {
        hydro_sparse_csr_destroy(L);
        free(h_bnd);
        free(u_copy);
        free(v_copy);
        return -1;
    }

    /* Copy current velocity */
    for (hydro_int i = 0; i < n; i++)
    {
        u_copy[i] = u[i];
        v_copy[i] = v[i];
    }

    /* CG tolerances */
    double tol = fmin(dt, 1e-5);
    hydro_int max_iter = fmin(n * 2 + 1000, 10000);

    /* Solve for u-velocity */
    hydro_cg_stats_t stats_u;
    int ret_u = hydro_parabolic_cg_solve(L, dt, u_copy, u, n, max_iter, tol, &stats_u);

    /* Solve for v-velocity */
    hydro_cg_stats_t stats_v;
    int ret_v = hydro_parabolic_cg_solve(L, dt, v_copy, v, n, max_iter, tol, &stats_v);

    /* Update x/y momentum from velocity */
    for (hydro_int i = 0; i < n; i++)
    {
        double hi = h[i];
        if (hi < domain->minimum_allowed_height) hi = domain->minimum_allowed_height;
        domain->xmom_centroid_values[i] = u[i] * hi;
        domain->ymom_centroid_values[i] = v[i] * hi;
    }

    hydro_sparse_csr_destroy(L);
    free(h_bnd);
    free(u_copy);
    free(v_copy);

    return (ret_u == 0 && ret_v == 0) ? 0 : -1;
}
