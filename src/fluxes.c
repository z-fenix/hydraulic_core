/**
 * fluxes.c — Central-upwind flux computation for shallow water equations
 *
 * Ported from anuga/shallow_water/sw_domain_openmp.c:
 *   __flux_function_central(), __rotate(),
 *   _openmp_compute_fluxes_central(), _openmp_protect(),
 *   _openmp_fix_negative_cells()
 */

#include "hydro/fluxes.h"
#include "hydro/config.h"
#include <math.h>
#include <string.h>
#include <stdio.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* ==========================================================================
 * Coordinate rotation
 * ========================================================================== */

static inline void hydro_rotate(double* q, double n1, double n2) {
    double q1 = q[1], q2 = q[2];
    q[1] =  n1 * q1 + n2 * q2;
    q[2] = -n2 * q1 + n1 * q2;
}

/* ==========================================================================
 * Velocity computation
 * ========================================================================== */

static inline void hydro_compute_velocity_terms(
    double h, double h_edge,
    double uh_raw, double vh_raw,
    double* u, double* uh, double* v, double* vh)
{
    /* Velocity protection: u = uh / (h + h0/h) prevents blowup
     * as h -> 0.  Ported from ANUGA sw_domain_openmp.c */
    const double h0 = 1.0e-6;  /* velocity_protection */
    if (h_edge > 0.0) {
        double inv = 1.0 / (h_edge + h0 / h_edge);
        *u  = uh_raw * inv;
        *uh = h * (*u);
        *v  = vh_raw * inv;
        *vh = h * inv * vh_raw;
    } else {
        *u  = 0.0; *uh = 0.0;
        *v  = 0.0; *vh = 0.0;
    }
}

/* ==========================================================================
 * Local Froude number
 * ========================================================================== */

static inline double hydro_compute_local_froude(
    hydro_int low_froude,
    double u_left, double u_right,
    double v_left, double v_right,
    double soundspeed_left, double soundspeed_right)
{
    double numerator = u_right*u_right + u_left*u_left
                     + v_right*v_right + v_left*v_left;
    double denominator = soundspeed_left*soundspeed_left
                       + soundspeed_right*soundspeed_right + 1.0e-10;

    if (low_froude == 1) {
        return sqrt(fmax(0.001, fmin(1.0, numerator / denominator)));
    } else if (low_froude == 2) {
        double fr = sqrt(numerator / denominator);
        return sqrt(fmin(1.0, 0.01 + fmax(fr - 0.01, 0.0)));
    }
    return 1.0;
}

static inline double hydro_compute_s_max(
    double u_left, double u_right,
    double c_left, double c_right)
{
    double s = fmax(u_left + c_left, u_right + c_right);
    return (s < 0.0) ? 0.0 : s;
}

static inline double hydro_compute_s_min(
    double u_left, double u_right,
    double c_left, double c_right)
{
    double s = fmin(u_left - c_left, u_right - c_right);
    return (s > 0.0) ? 0.0 : s;
}

/* ==========================================================================
 * Central-upwind flux function (the core kernel)
 *
 * Based on Kurganov, Noelle, Petrova (2001), eq (3.15) on page 714.
 * Uses stage w = h+z formulation with discontinuous bed elevation.
 * ========================================================================== */

int hydro_flux_function_central(
    const double* ql, const double* qr,
    double h_left, double h_right,
    double hle, double hre,
    double n1, double n2,
    double epsilon, double ze, double g,
    double* edgeflux, double* max_speed,
    double* pressure_flux,
    hydro_int low_froude)
{
    double uh_left, vh_left, u_left, v_left;
    double uh_right, vh_right, u_right, v_right;
    double soundspeed_left, soundspeed_right;
    double denom;

    double q_left_rotated[3], q_right_rotated[3];
    double flux_left[3], flux_right[3];

    /* Copy and rotate to align with normal direction */
    for (int i = 0; i < 3; i++) {
        q_left_rotated[i]  = ql[i];
        q_right_rotated[i] = qr[i];
    }

    hydro_rotate(q_left_rotated, n1, n2);
    hydro_rotate(q_right_rotated, n1, n2);

    /* Compute velocities in rotated frame */
    uh_left  = q_left_rotated[1];
    vh_left  = q_left_rotated[2];
    hydro_compute_velocity_terms(h_left, hle,
        q_left_rotated[1], q_left_rotated[2],
        &u_left, &uh_left, &v_left, &vh_left);

    uh_right = q_right_rotated[1];
    vh_right = q_right_rotated[2];
    hydro_compute_velocity_terms(h_right, hre,
        q_right_rotated[1], q_right_rotated[2],
        &u_right, &uh_right, &v_right, &vh_right);

    /* Wave speeds */
    soundspeed_left  = sqrt(g * h_left);
    soundspeed_right = sqrt(g * h_right);

    double local_fr = hydro_compute_local_froude(
        low_froude, u_left, u_right, v_left, v_right,
        soundspeed_left, soundspeed_right);

    double s_max = hydro_compute_s_max(
        u_left, u_right, soundspeed_left, soundspeed_right);
    double s_min = hydro_compute_s_min(
        u_left, u_right, soundspeed_left, soundspeed_right);

    /* Physical fluxes in rotated frame (without pressure term) */
    flux_left[0]  = u_left * h_left;
    flux_left[1]  = u_left * uh_left;
    flux_left[2]  = u_left * vh_left;

    flux_right[0] = u_right * h_right;
    flux_right[1] = u_right * uh_right;
    flux_right[2] = u_right * vh_right;

    /* Central-upwind flux formula */
    denom = s_max - s_min;
    double inv_denom = 1.0 / fmax(denom, 1.0e-100);
    double s_max_s_min = s_max * s_min;

    if (denom < epsilon) {
        /* Both wave speeds are very small — zero flux */
        memset(edgeflux, 0, 3 * sizeof(double));
        *max_speed = 0.0;
        *pressure_flux = 0.5 * g * 0.5 * (h_left*h_left + h_right*h_right);
    } else {
        *max_speed = fmax(s_max, -s_min);

        /* Stage flux */
        double flux_0 = s_max * flux_left[0] - s_min * flux_right[0];
        flux_0 += s_max_s_min * (fmax(q_right_rotated[0], ze)
                               - fmax(q_left_rotated[0], ze));
        edgeflux[0] = flux_0 * inv_denom;

        /* X-momentum flux */
        double flux_1 = s_max * flux_left[1] - s_min * flux_right[1];
        flux_1 += local_fr * s_max_s_min * (uh_right - uh_left);
        edgeflux[1] = flux_1 * inv_denom;

        /* Y-momentum flux */
        double flux_2 = s_max * flux_left[2] - s_min * flux_right[2];
        flux_2 += local_fr * s_max_s_min * (vh_right - vh_left);
        edgeflux[2] = flux_2 * inv_denom;

        /* Pressure flux (separated for wet/dry treatment) */
        *pressure_flux = 0.5 * g
            * (s_max * h_left*h_left - s_min * h_right*h_right)
            * inv_denom;

        /* Rotate back to global coordinates */
        hydro_rotate(edgeflux, n1, -n2);
    }

    return 0;
}

/* ==========================================================================
 * Compute fluxes for all edges — the main OpenMP loop
 * ========================================================================== */

double hydro_compute_fluxes_central(
    hydro_domain_t* domain, double evolve_max_timestep)
{
    hydro_int N = domain->number_of_elements;
    double g      = domain->g;
    double epsilon = domain->epsilon;
    hydro_int low_froude = domain->low_froude;

    double local_timestep = 1.0e+100;
    double boundary_flux_sum = 0.0;

    /* Reset explicit updates and compute fluxes per triangle */
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static) \
        firstprivate(N, g, epsilon, low_froude) reduction(min:local_timestep) reduction(+:boundary_flux_sum)
    #endif
    for (hydro_int k = 0; k < N; k++) {
        hydro_int k3 = 3 * k;
        double area     = domain->areas[k];
        double speed_max_last = 0.0;

        /* Zero the explicit updates for this triangle */
        domain->stage_explicit_update[k] = 0.0;
        domain->xmom_explicit_update[k]  = 0.0;
        domain->ymom_explicit_update[k]  = 0.0;

        /* Loop over 3 edges */
        for (hydro_int i = 0; i < 3; i++) {
            hydro_int ki   = k3 + i;
            hydro_int ki2  = 2 * ki;

            double edgeflux[3];
            double pressure_flux;
            double max_speed_local;

            /* Extract edge data */
            hydro_int n = domain->neighbours[ki];
            int is_boundary = (n < 0);

            double ql[3] = {domain->stage_edge_values[ki],
                            domain->xmom_edge_values[ki],
                            domain->ymom_edge_values[ki]};
            double zl    = domain->bed_edge_values[ki];
            double hle   = domain->height_edge_values[ki];
            double len   = domain->edgelengths[ki];
            double nx    = domain->normals[ki2];
            double ny    = domain->normals[ki2 + 1];

            double qr[3], zr, hre;

            if (is_boundary) {
                hydro_int bnd_idx = -n - 1;
                qr[0] = domain->stage_boundary_values[bnd_idx];
                qr[1] = domain->xmom_boundary_values[bnd_idx];
                qr[2] = domain->ymom_boundary_values[bnd_idx];
                zr    = zl;
                hre   = fmax(qr[0] - zr, 0.0);
            } else {
                hydro_int m  = domain->neighbour_edges[ki];
                hydro_int nm = n * 3 + m;
                qr[0] = domain->stage_edge_values[nm];
                qr[1] = domain->xmom_edge_values[nm];
                qr[2] = domain->ymom_edge_values[nm];
                zr    = domain->bed_edge_values[nm];
                hre   = domain->height_edge_values[nm];
            }

            double z_half = fmax(zl, zr);
            double h_left  = fmax(hle + zl - z_half, 0.0);
            double h_right = fmax(hre + zr - z_half, 0.0);

            /* Compute flux */
            if (h_left == 0.0 && h_right == 0.0) {
                edgeflux[0] = edgeflux[1] = edgeflux[2] = 0.0;
                max_speed_local = 0.0;
                pressure_flux   = 0.0;
            } else {
                hydro_flux_function_central(
                    ql, qr, h_left, h_right, hle, hre,
                    nx, ny, epsilon, z_half, g,
                    edgeflux, &max_speed_local, &pressure_flux,
                    low_froude);
            }

            /* Scale flux by edge length (negative for outward normal convention) */
            edgeflux[0] *= -len;
            edgeflux[1] *= -len;
            edgeflux[2] *= -len;

            /* CFL condition */
            if (domain->tri_full_flag[k] == 1 && max_speed_local > epsilon) {
                double et = domain->radii[k] / fmax(max_speed_local, epsilon);
                local_timestep = fmin(local_timestep, et);
                speed_max_last = fmax(speed_max_last, max_speed_local);
            }

            /* Accumulate explicit update */
            domain->stage_explicit_update[k] += edgeflux[0];
            domain->xmom_explicit_update[k]  += edgeflux[1];
            domain->ymom_explicit_update[k]  += edgeflux[2];

            /* Track boundary flux */
            if ((is_boundary && domain->tri_full_flag[k] == 1)
                || (!is_boundary && domain->tri_full_flag[k] == 1
                    && domain->tri_full_flag[n] == 0)) {
                boundary_flux_sum += edgeflux[0];
            }

            /* Gravity / pressure gradient contribution */
            double bedslope_work = len * (
                -g * 0.5 * (h_left*h_left - hle*hle
                          - (hle + domain->height_centroid_values[k])
                          * (zl - domain->bed_centroid_values[k]))
                + pressure_flux);
            domain->xmom_explicit_update[k] -= domain->normals[ki2] * bedslope_work;
            domain->ymom_explicit_update[k] -= domain->normals[ki2 + 1] * bedslope_work;
        }

        /* Normalise by area */
        double inv_area = 1.0 / area;
        domain->stage_explicit_update[k] *= inv_area;
        domain->xmom_explicit_update[k]  *= inv_area;
        domain->ymom_explicit_update[k]  *= inv_area;

        domain->max_speed[k] = speed_max_last;
    }

    /* Store boundary flux sum */
    domain->boundary_flux_sum[0] = boundary_flux_sum;

    (void)evolve_max_timestep;
    return local_timestep;
}

/* ==========================================================================
 * Positivity protection
 * ========================================================================== */

double hydro_protect(hydro_domain_t* domain) {
    double mass_error = 0.0;
    double h0 = domain->minimum_allowed_height;
    hydro_int N = domain->number_of_elements;

    #ifdef _OPENMP
    #pragma omp parallel for schedule(static) reduction(+:mass_error)
    #endif
    for (hydro_int k = 0; k < N; k++) {
        hydro_int k3 = 3 * k;
        double hc = domain->stage_centroid_values[k]
                  - domain->bed_centroid_values[k];

        if (hc < h0) {
            /* Set momentum to zero and ensure h >= 0 */
            domain->xmom_centroid_values[k] = 0.0;
            domain->ymom_centroid_values[k] = 0.0;

            if (hc <= 0.0) {
                double bmin = domain->bed_centroid_values[k];

                if (domain->stage_centroid_values[k] < bmin) {
                    mass_error += (bmin - domain->stage_centroid_values[k])
                                * domain->areas[k];
                    domain->stage_centroid_values[k] = bmin;

                    /* Also set vertex values for consistency */
                    domain->stage_vertex_values[k3]     = bmin;
                    domain->stage_vertex_values[k3 + 1] = bmin;
                    domain->stage_vertex_values[k3 + 2] = bmin;
                }
            }
        }
    }

    return mass_error;
}

/* ==========================================================================
 * Fix negative cells after update
 * ========================================================================== */

hydro_int hydro_fix_negative_cells(hydro_domain_t* domain) {
    hydro_int num_negative = 0;
    hydro_int N = domain->number_of_elements;

    #ifdef _OPENMP
    #pragma omp parallel for schedule(static) reduction(+:num_negative)
    #endif
    for (hydro_int k = 0; k < N; k++) {
        double h = domain->stage_centroid_values[k]
                 - domain->bed_centroid_values[k];
        if (h < 0.0 && domain->tri_full_flag[k] > 0) {
            num_negative++;
            domain->stage_centroid_values[k] = domain->bed_centroid_values[k];
            domain->xmom_centroid_values[k]  = 0.0;
            domain->ymom_centroid_values[k]  = 0.0;
        }
    }

    return num_negative;
}

/* ==========================================================================
 * Get edge data helper (for external use / debugging)
 * ========================================================================== */

void hydro_get_edge_data(
    const hydro_domain_t* domain, hydro_int k, hydro_int i,
    hydro_edge_data_t* edge)
{
    hydro_int k3i = 3 * k + i;
    hydro_int n  = domain->neighbours[k3i];

    edge->ki  = k3i;
    edge->ki2 = 2 * k3i;

    edge->ql[0] = domain->stage_edge_values[k3i];
    edge->ql[1] = domain->xmom_edge_values[k3i];
    edge->ql[2] = domain->ymom_edge_values[k3i];
    edge->zl     = domain->bed_edge_values[k3i];
    edge->hle    = domain->height_edge_values[k3i];
    edge->length = domain->edgelengths[k3i];

    edge->n = n;
    edge->is_boundary = (n < 0);

    edge->normal_x = domain->normals[2*k3i];
    edge->normal_y = domain->normals[2*k3i + 1];

    edge->hc   = domain->height_centroid_values[k];
    edge->zc   = domain->bed_centroid_values[k];
    edge->hc_n = edge->hc;
    edge->zc_n = edge->zc;

    if (edge->is_boundary) {
        edge->qr[0] = 0; edge->qr[1] = 0; edge->qr[2] = 0;
        edge->zr  = edge->zl;
        edge->hre = 0;
    } else {
        hydro_int m  = domain->neighbour_edges[k3i];
        hydro_int nm = n * 3 + m;
        edge->hc_n = domain->height_centroid_values[n];
        edge->zc_n = domain->bed_centroid_values[n];
        edge->qr[0] = domain->stage_edge_values[nm];
        edge->qr[1] = domain->xmom_edge_values[nm];
        edge->qr[2] = domain->ymom_edge_values[nm];
        edge->zr  = domain->bed_edge_values[nm];
        edge->hre = domain->height_edge_values[nm];
    }

    edge->z_half  = fmax(edge->zl, edge->zr);
    edge->h_left  = fmax(edge->hle + edge->zl - edge->z_half, 0.0);
    edge->h_right = fmax(edge->hre + edge->zr - edge->z_half, 0.0);

    edge->is_riverwall = 0;
}
