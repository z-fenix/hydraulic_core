/**
 * quantity.c — Quantity operations
 *
 * Ported from:
 *   anuga/abstract_2d_finite_volumes/quantity_openmp.c
 *   anuga/shallow_water/sw_domain_openmp.c
 *   anuga/utilities/util_ext.h
 */

#include "hydro/quantity.h"
#include "hydro/config.h"
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* ==========================================================================
 * Utility: _gradient — compute gradient from 3 points
 * Ported from anuga/utilities/util_ext.h
 * ========================================================================== */

static inline void hydro_gradient(
    double x0, double y0, double x1, double y1, double x2, double y2,
    double q0, double q1, double q2,
    double* a, double* b)
{
    double det = (y2 - y0) * (x1 - x0) - (y1 - y0) * (x2 - x0);
    *a = ((y2 - y0) * (q1 - q0) - (y1 - y0) * (q2 - q0)) / det;
    *b = ((x1 - x0) * (q2 - q0) - (x2 - x0) * (q1 - q0)) / det;
}

/* ==========================================================================
 * Utility: _gradient2 — gradient from 2 points
 * ========================================================================== */

static inline void hydro_gradient2(
    double x0, double y0, double x1, double y1,
    double q0, double q1, double* a, double* b)
{
    double xx = x1 - x0, yy = y1 - y0, qq = q1 - q0;
    double det = xx * xx + yy * yy;
    *a = xx * qq / det;
    *b = yy * qq / det;
}

/* ==========================================================================
 * Compute edge coordinate differences (centroid→edge vector components)
 * ========================================================================== */

static inline void hydro_compute_edge_diffs(
    double cx, double cy,
    double x0, double y0,
    double x1, double y1,
    double x2, double y2,
    double* dx0, double* dx1, double* dx2,
    double* dy0, double* dy1, double* dy2)
{
    *dx0 = x0 - cx;
    *dy0 = y0 - cy;
    *dx1 = x1 - cx;
    *dy1 = y1 - cy;
    *dx2 = x2 - cx;
    *dy2 = y2 - cy;
}

/* ==========================================================================
 * Set all edge values from centroid
 * ========================================================================== */

static inline void hydro_set_all_edge_values_from_centroid(
    hydro_domain_t* d, hydro_int k, hydro_int k3,
    double* centroid_vals, double* edge_vals)
{
    double cv = centroid_vals[k];
    edge_vals[k3] = cv;
    edge_vals[k3 + 1] = cv;
    edge_vals[k3 + 2] = cv;
}

/* ==========================================================================
 * Quantity update: Q_new = (Q_old + dt*E) / (1 - dt*SI)
 * ========================================================================== */

void hydro_quantity_update(hydro_domain_t* domain, double timestep)
{
    hydro_int N = domain->number_of_elements;

    if (!domain->stage_centroid_values || !domain->stage_explicit_update) return;

    #ifdef _OPENMP
    #pragma omp parallel for simd schedule(static)
    #endif
    for (hydro_int k = 0; k < N; k++) {
        /* Stage */
        double s = domain->stage_centroid_values[k];
        if (s != 0.0) domain->stage_semi_implicit_update[k] /= s;
        else domain->stage_semi_implicit_update[k] = 0.0;

        double s_new = s + timestep * domain->stage_explicit_update[k];
        double denom = 1.0 - timestep * domain->stage_semi_implicit_update[k];
        if (fabs(denom) > 1e-12)
            s_new /= denom;

        /* Clamp: stage must stay at or above bed elevation */
        if (domain->bed_centroid_values)
        {
            double bed = domain->bed_centroid_values[k];
            if (s_new < bed) s_new = bed;
        }
        domain->stage_centroid_values[k] = s_new;
        domain->stage_explicit_update[k] = 0.0;
        domain->stage_semi_implicit_update[k] = 0.0;
    }

    if (!domain->xmom_centroid_values || !domain->xmom_explicit_update) return;
    #ifdef _OPENMP
    #pragma omp parallel for simd schedule(static)
    #endif
    for (hydro_int k = 0; k < N; k++) {
        double xm = domain->xmom_centroid_values[k];
        if (xm != 0.0) domain->xmom_semi_implicit_update[k] /= xm;
        else domain->xmom_semi_implicit_update[k] = 0.0;

        domain->xmom_centroid_values[k] += timestep * domain->xmom_explicit_update[k];
        double denom = 1.0 - timestep * domain->xmom_semi_implicit_update[k];
        if (fabs(denom) > 1e-12)
            domain->xmom_centroid_values[k] /= denom;
        domain->xmom_explicit_update[k] = 0.0;
        domain->xmom_semi_implicit_update[k] = 0.0;
    }

    if (!domain->ymom_centroid_values || !domain->ymom_explicit_update) return;
    #ifdef _OPENMP
    #pragma omp parallel for simd schedule(static)
    #endif
    for (hydro_int k = 0; k < N; k++) {
        double ym = domain->ymom_centroid_values[k];
        if (ym != 0.0) domain->ymom_semi_implicit_update[k] /= ym;
        else domain->ymom_semi_implicit_update[k] = 0.0;

        domain->ymom_centroid_values[k] += timestep * domain->ymom_explicit_update[k];
        double denom = 1.0 - timestep * domain->ymom_semi_implicit_update[k];
        if (fabs(denom) > 1e-12)
            domain->ymom_centroid_values[k] /= denom;
        domain->ymom_explicit_update[k] = 0.0;
        domain->ymom_semi_implicit_update[k] = 0.0;
    }
}

/* ==========================================================================
 * Compute gradients from neighbour centroid values
 * Ported from quantity_openmp.c:_compute_gradients()
 * ========================================================================== */

static void hydro_compute_gradients(
    hydro_domain_t* domain,
    double* centroid_values,
    double* x_gradient,
    double* y_gradient)
{
    hydro_int N = domain->number_of_elements;

    #ifdef _OPENMP
    #pragma omp parallel for simd schedule(static)
    #endif
    for (hydro_int k = 0; k < N; k++) {
        hydro_int k3 = 3 * k;

        /* Dry cell: no gradient (use first-order) */
        if (domain->height_centroid_values &&
            domain->height_centroid_values[k] < domain->minimum_allowed_height)
        {
            x_gradient[k] = 0.0;
            y_gradient[k] = 0.0;
            continue;
        }

        if (domain->number_of_boundaries[k] < 2)
        {
            /* Two or three true neighbours — fit a plane */
            hydro_int k0 = domain->surrogate_neighbours[k3];
            hydro_int k1 = domain->surrogate_neighbours[k3 + 1];
            hydro_int k2 = domain->surrogate_neighbours[k3 + 2];

            if (k0 == k1 || k1 == k2)
            {
                x_gradient[k] = 0.0;
                y_gradient[k] = 0.0;
                continue;
            }

            double q0 = centroid_values[k0];
            double q1 = centroid_values[k1];
            double q2 = centroid_values[k2];

            double x0 = domain->centroid_coordinates[2 * k0];
            double y0 = domain->centroid_coordinates[2 * k0 + 1];
            double x1 = domain->centroid_coordinates[2 * k1];
            double y1 = domain->centroid_coordinates[2 * k1 + 1];
            double x2 = domain->centroid_coordinates[2 * k2];
            double y2 = domain->centroid_coordinates[2 * k2 + 1];

            hydro_gradient(x0, y0, x1, y1, x2, y2, q0, q1, q2,
                           &x_gradient[k], &y_gradient[k]);
        }
        else if (domain->number_of_boundaries[k] == 2)
        {
            /* One true neighbour — gradient along that direction */
            hydro_int k0 = k; /* self */
            for (int i = 0; i < 3; i++)
            {
                hydro_int nb = domain->surrogate_neighbours[k3 + i];
                if (nb != k)
                {
                    k0 = nb;
                    break;
                }
            }
            if (k0 == k)
            {
                x_gradient[k] = 0.0;
                y_gradient[k] = 0.0;
                continue;
            }

            double q0 = centroid_values[k0], q1 = centroid_values[k];
            double x0 = domain->centroid_coordinates[2 * k0];
            double y0 = domain->centroid_coordinates[2 * k0 + 1];
            double x1 = domain->centroid_coordinates[2 * k];
            double y1 = domain->centroid_coordinates[2 * k + 1];

            hydro_gradient2(x0, y0, x1, y1, q0, q1,
                            &x_gradient[k], &y_gradient[k]);
        }
        else
        {
            /* No true neighbours — first-order fallback */
            x_gradient[k] = 0.0;
            y_gradient[k] = 0.0;
        }
    }
}

/* ==========================================================================
 * Extrapolate from gradients to vertices, then average to edges
 * ========================================================================== */

static void hydro_extrapolate_from_gradient(
    hydro_domain_t* domain,
    double* centroid_values,
    double* vertex_values,
    double* edge_values,
    double* x_gradient,
    double* y_gradient)
{
    hydro_int N = domain->number_of_elements;

    #ifdef _OPENMP
    #pragma omp parallel for simd schedule(static)
    #endif
    for (hydro_int k = 0; k < N; k++) {
        hydro_int k3 = 3 * k;
        hydro_int k6 = 6 * k;
        hydro_int k2 = 2 * k;

        double cx = domain->centroid_coordinates[k2];
        double cy = domain->centroid_coordinates[k2 + 1];
        double a = x_gradient[k];
        double b = y_gradient[k];
        double cv = centroid_values[k];

        /* Vertex coordinates */
        double x0 = domain->vertex_coordinates[k6];
        double y0 = domain->vertex_coordinates[k6 + 1];
        double x1 = domain->vertex_coordinates[k6 + 2];
        double y1 = domain->vertex_coordinates[k6 + 3];
        double x2 = domain->vertex_coordinates[k6 + 4];
        double y2 = domain->vertex_coordinates[k6 + 5];

        /* Extrapolate to vertices */
        vertex_values[k3] = cv + a * (x0 - cx) + b * (y0 - cy);
        vertex_values[k3 + 1] = cv + a * (x1 - cx) + b * (y1 - cy);
        vertex_values[k3 + 2] = cv + a * (x2 - cx) + b * (y2 - cy);

        /* Average vertices → edges */
        edge_values[k3] = 0.5 * (vertex_values[k3 + 1] + vertex_values[k3 + 2]);
        edge_values[k3 + 1] = 0.5 * (vertex_values[k3 + 2] + vertex_values[k3]);
        edge_values[k3 + 2] = 0.5 * (vertex_values[k3] + vertex_values[k3 + 1]);
    }
}

/* ==========================================================================
 * Slope limiter — enforce min/max bounds from neighbour values
 * Ported from quantity_openmp.c:_limit_edges_by_all_neighbours()
 * ========================================================================== */

static void hydro_limit_edges(
    hydro_domain_t* domain,
    double* centroid_values,
    double* vertex_values,
    double* edge_values,
    double* x_gradient,
    double* y_gradient,
    double beta)
{
    hydro_int N = domain->number_of_elements;

    #ifdef _OPENMP
    #pragma omp parallel for simd schedule(static)
    #endif
    for (hydro_int k = 0; k < N; k++) {
        hydro_int k3 = 3 * k;
        double qc = centroid_values[k];
        double qmin = qc, qmax = qc;

        /* Find min/max among neighbours */
        for (int i = 0; i < 3; i++)
        {
            hydro_int n = domain->neighbours[k3 + i];
            double qn = (n >= 0) ? centroid_values[n] : qc;
            if (qn < qmin) qmin = qn;
            if (qn > qmax) qmax = qn;
        }

        /* Compute limiter phi */
        double phi = 1.0;
        double dqa[3];
        for (int i = 0; i < 3; i++)
        {
            double dq = edge_values[k3 + i] - qc;
            dqa[i] = dq;

            double r = 1.0;
            if (dq > 0.0) r = (qmax - qc) / dq;
            if (dq < 0.0) r = (qmin - qc) / dq;

            double rbeta = fmin(r * beta, 1.0);
            if (rbeta < phi) phi = rbeta;
        }

        /* Apply limiter to gradients, edges, and vertices */
        x_gradient[k] *= phi;
        y_gradient[k] *= phi;

        edge_values[k3] = qc + phi * dqa[0];
        edge_values[k3 + 1] = qc + phi * dqa[1];
        edge_values[k3 + 2] = qc + phi * dqa[2];

        /* Reconstruct vertex values from limited edges */
        vertex_values[k3] = edge_values[k3 + 1] + edge_values[k3 + 2] - edge_values[k3];
        vertex_values[k3 + 1] = edge_values[k3 + 2] + edge_values[k3] - edge_values[k3 + 1];
        vertex_values[k3 + 2] = edge_values[k3] + edge_values[k3 + 1] - edge_values[k3 + 2];
    }
}

/* ==========================================================================
 * Second-order extrapolation: centroid→edges (DE-optimized edge-only path)
 *
 * This is a simplified version of _openmp_extrapolate_second_order_edge_sw.
 * It computes gradients, extrapolates to edges, and limits.
 * ========================================================================== */

void hydro_quantity_extrapolate_second_order_edge(hydro_domain_t* domain)
{
    /* Compute gradients for each conserved quantity */
    /* Use the x_centroid_work / y_centroid_work arrays for temporary gradient storage */

    /* Stage */
    hydro_compute_gradients(domain,
                            domain->stage_centroid_values,
                            domain->x_centroid_work,
                            domain->y_centroid_work);
    hydro_extrapolate_from_gradient(domain,
                                    domain->stage_centroid_values,
                                    domain->stage_vertex_values,
                                    domain->stage_edge_values,
                                    domain->x_centroid_work,
                                    domain->y_centroid_work);
    hydro_limit_edges(domain,
                      domain->stage_centroid_values,
                      domain->stage_vertex_values,
                      domain->stage_edge_values,
                      domain->x_centroid_work,
                      domain->y_centroid_work,
                      domain->beta_w);

    /* X-momentum */
    hydro_compute_gradients(domain,
                            domain->xmom_centroid_values,
                            domain->x_centroid_work,
                            domain->y_centroid_work);
    hydro_extrapolate_from_gradient(domain,
                                    domain->xmom_centroid_values,
                                    domain->xmom_vertex_values,
                                    domain->xmom_edge_values,
                                    domain->x_centroid_work,
                                    domain->y_centroid_work);
    hydro_limit_edges(domain,
                      domain->xmom_centroid_values,
                      domain->xmom_vertex_values,
                      domain->xmom_edge_values,
                      domain->x_centroid_work,
                      domain->y_centroid_work,
                      domain->beta_uh);

    /* Y-momentum */
    hydro_compute_gradients(domain,
                            domain->ymom_centroid_values,
                            domain->x_centroid_work,
                            domain->y_centroid_work);
    hydro_extrapolate_from_gradient(domain,
                                    domain->ymom_centroid_values,
                                    domain->ymom_vertex_values,
                                    domain->ymom_edge_values,
                                    domain->x_centroid_work,
                                    domain->y_centroid_work);
    hydro_limit_edges(domain,
                      domain->ymom_centroid_values,
                      domain->ymom_vertex_values,
                      domain->ymom_edge_values,
                      domain->x_centroid_work,
                      domain->y_centroid_work,
                      domain->beta_vh);

    /* Height and bed */
    hydro_compute_gradients(domain,
                            domain->height_centroid_values,
                            domain->x_centroid_work,
                            domain->y_centroid_work);
    hydro_extrapolate_from_gradient(domain,
                                    domain->height_centroid_values,
                                    domain->height_vertex_values,
                                    domain->height_edge_values,
                                    domain->x_centroid_work,
                                    domain->y_centroid_work);
    hydro_limit_edges(domain,
                      domain->height_centroid_values,
                      domain->height_vertex_values,
                      domain->height_edge_values,
                      domain->x_centroid_work,
                      domain->y_centroid_work,
                      domain->beta_w);

    /* Bed from height + stage */
    hydro_int N = domain->number_of_elements;
    #ifdef _OPENMP
    #pragma omp parallel for simd schedule(static)
    #endif
    for (hydro_int k = 0; k < N; k++) {
        hydro_int k3 = 3 * k;
        domain->bed_edge_values[k3] = domain->stage_edge_values[k3] - domain->height_edge_values[k3];
        domain->bed_edge_values[k3 + 1] = domain->stage_edge_values[k3 + 1] - domain->height_edge_values[k3 + 1];
        domain->bed_edge_values[k3 + 2] = domain->stage_edge_values[k3 + 2] - domain->height_edge_values[k3 + 2];
    }
}

/* ==========================================================================
 * Full second-order extrapolation (centroid→vertices→edges)
 * ========================================================================== */

void hydro_quantity_extrapolate_second_order(hydro_domain_t* domain)
{
    /* Same as edge version but also populates vertex values for storage */
    hydro_quantity_extrapolate_second_order_edge(domain);
}

/* ==========================================================================
 * First-order extrapolation
 * ========================================================================== */

void hydro_quantity_extrapolate_first_order(hydro_domain_t* domain)
{
    hydro_int N = domain->number_of_elements;

    struct
    {
        double** c;
        double** e;
        double** v;
    } q[] = {
        {&domain->stage_centroid_values, &domain->stage_edge_values, &domain->stage_vertex_values},
        {&domain->xmom_centroid_values, &domain->xmom_edge_values, &domain->xmom_vertex_values},
        {&domain->ymom_centroid_values, &domain->ymom_edge_values, &domain->ymom_vertex_values},
        {&domain->bed_centroid_values, &domain->bed_edge_values, &domain->bed_vertex_values},
        {&domain->height_centroid_values, &domain->height_edge_values, &domain->height_vertex_values},
    };

    for (int qi = 0; qi < 5; qi++)
    {
        if (!*q[qi].c) continue;
        #ifdef _OPENMP
        #pragma omp parallel for simd schedule(static)
        #endif
        for (hydro_int k = 0; k < N; k++) {
            hydro_int k3 = 3 * k;
            double val = (*q[qi].c)[k];
            if (*q[qi].e)
            {
                (*q[qi].e)[k3] = val;
                (*q[qi].e)[k3 + 1] = val;
                (*q[qi].e)[k3 + 2] = val;
            }
            if (*q[qi].v)
            {
                (*q[qi].v)[k3] = val;
                (*q[qi].v)[k3 + 1] = val;
                (*q[qi].v)[k3 + 2] = val;
            }
        }
    }
}

/* ==========================================================================
 * Distribute edges → vertices
 * Ported from sw_domain_openmp.c:_openmp_distribute_edges_to_vertices()
 * ========================================================================== */

void hydro_quantity_distribute_edges_to_vertices(hydro_domain_t* domain)
{
    hydro_int N = domain->number_of_elements;

    struct
    {
        double* e;
        double* v;
    } q[] = {
        {domain->stage_edge_values, domain->stage_vertex_values},
        {domain->xmom_edge_values, domain->xmom_vertex_values},
        {domain->ymom_edge_values, domain->ymom_vertex_values},
        {domain->bed_edge_values, domain->bed_vertex_values},
        {domain->height_edge_values, domain->height_vertex_values},
    };

    for (int qi = 0; qi < 5; qi++)
    {
        if (!q[qi].e || !q[qi].v) continue;
        #ifdef _OPENMP
        #pragma omp parallel for simd schedule(static)
        #endif
        for (hydro_int k = 0; k < N; k++) {
            hydro_int k3 = 3 * k;
            q[qi].v[k3] = q[qi].e[k3 + 1] + q[qi].e[k3 + 2] - q[qi].e[k3];
            q[qi].v[k3 + 1] = q[qi].e[k3 + 2] + q[qi].e[k3] - q[qi].e[k3 + 1];
            q[qi].v[k3 + 2] = q[qi].e[k3] + q[qi].e[k3 + 1] - q[qi].e[k3 + 2];
        }
    }
}

/* ==========================================================================
 * Backup / SAXPY for RK multi-stage
 * ========================================================================== */

void hydro_quantity_backup(hydro_domain_t* domain)
{
    hydro_int N = domain->number_of_elements;
    if (!domain->stage_backup_values || !domain->stage_centroid_values) return;

    #ifdef _OPENMP
    #pragma omp parallel for simd schedule(static)
    #endif
    for (hydro_int k = 0; k < N; k++) {
        domain->stage_backup_values[k] = domain->stage_centroid_values[k];
        domain->xmom_backup_values[k] = domain->xmom_centroid_values[k];
        domain->ymom_backup_values[k] = domain->ymom_centroid_values[k];
    }
}

void hydro_quantity_saxpy(hydro_domain_t* domain, double a, double b, double c)
{
    hydro_int N = domain->number_of_elements;
    if (!domain->stage_backup_values || !domain->stage_centroid_values) return;

    #ifdef _OPENMP
    #pragma omp parallel for simd schedule(static)
    #endif
    for (hydro_int k = 0; k < N; k++) {
        domain->stage_centroid_values[k] =
            a * domain->stage_backup_values[k] + b * domain->stage_centroid_values[k];
        domain->xmom_centroid_values[k] =
            a * domain->xmom_backup_values[k] + b * domain->xmom_centroid_values[k];
        domain->ymom_centroid_values[k] =
            a * domain->ymom_backup_values[k] + b * domain->ymom_centroid_values[k];

        if (c != 0.0)
        {
            domain->stage_centroid_values[k] /= c;
            domain->xmom_centroid_values[k] /= c;
            domain->ymom_centroid_values[k] /= c;
        }
    }
}

/* ==========================================================================
 * Update derived quantities: height, velocity from conserved
 * ========================================================================== */

void hydro_quantity_update_derived(hydro_domain_t* domain)
{
    hydro_int N = domain->number_of_elements;
    hydro_int n_edges = domain->number_of_edges;
    double h0 = domain->minimum_allowed_height;

    if (!domain->stage_centroid_values || !domain->bed_centroid_values) return;

    /* Allocate if needed */
    if (!domain->height_centroid_values)
        domain->height_centroid_values =
            (double*)calloc((size_t)N, sizeof(double));
    if (!domain->xvelocity_centroid_values)
        domain->xvelocity_centroid_values =
            (double*)calloc((size_t)N, sizeof(double));
    if (!domain->yvelocity_centroid_values)
        domain->yvelocity_centroid_values =
            (double*)calloc((size_t)N, sizeof(double));
    if (!domain->height_edge_values)
        domain->height_edge_values =
            (double*)calloc((size_t)n_edges, sizeof(double));
    if (!domain->height_vertex_values)
        domain->height_vertex_values =
            (double*)calloc((size_t)n_edges, sizeof(double));
    if (!domain->xvelocity_edge_values)
        domain->xvelocity_edge_values =
            (double*)calloc((size_t)n_edges, sizeof(double));
    if (!domain->yvelocity_edge_values)
        domain->yvelocity_edge_values =
            (double*)calloc((size_t)n_edges, sizeof(double));

    #ifdef _OPENMP
    #pragma omp parallel for simd schedule(static)
    #endif
    for (hydro_int k = 0; k < N; k++) {
        double bed = domain->bed_centroid_values[k];
        double stage = domain->stage_centroid_values[k];
        double h = stage - bed;

        /* Ensure non-negative water depth:
         * If stage is below bed, raise stage to bed so height ≥ 0.
         * For truly dry cells (h == 0), leave height at 0 — the flux
         * computation will handle dry-wet interfaces correctly.
         * Only clamp h to h0 when it's positive but tiny, to protect
         * velocity computations from division by zero. */
        if (h < 0.0)
        {
            stage = bed;
            domain->stage_centroid_values[k] = bed;
            h = 0.0; /* truly dry — flux handles dry-wet interfaces */
        }
        else if (h < h0)
        {
            /* Near-dry: keep the small depth for velocity protection
             * but don't artificially inflate it. */
            h = h0;
        }
        domain->height_centroid_values[k] = h;

        if (domain->xmom_centroid_values)
        {
            domain->xvelocity_centroid_values[k] =
                domain->xmom_centroid_values[k] / h;
        }
        if (domain->ymom_centroid_values)
        {
            domain->yvelocity_centroid_values[k] =
                domain->ymom_centroid_values[k] / h;
        }
    }
}
