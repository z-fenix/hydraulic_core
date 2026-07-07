/**
 * forcing.c — Wind stress, barometric pressure, and rainfall forcing terms
 *
 * Ported from ANUGA:
 *   anuga/shallow_water/forcing.py
 *   anuga/operators/rate_operators.py
 */

#include "hydro/forcing.h"
#include <math.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* Physical constants from anuga/config.py */
#define HYDRO_RHO_A  1.2e-3   /* atmospheric density */
#define HYDRO_RHO_W  1023.0   /* water density (salt water) */
#define HYDRO_ETA_W  3.0e-3   /* wind drag coefficient */

/* Computed constant: eta_w * rho_a / rho_w */
#define HYDRO_WIND_CONST (HYDRO_ETA_W * HYDRO_RHO_A / HYDRO_RHO_W)

/* =========================================================================
 * Gradient on a Triangle
 * ========================================================================= */

void hydro_gradient_triangle(double x0, double y0,
                             double x1, double y1,
                             double x2, double y2,
                             double q0, double q1, double q2,
                             double* dqdx, double* dqdy)
{
    double det = (y2 - y0) * (x1 - x0) - (y1 - y0) * (x2 - x0);
    if (fabs(det) < 1e-30)
    {
        *dqdx = *dqdy = 0.0;
        return;
    }
    *dqdx = ((y2 - y0) * (q1 - q0) - (y1 - y0) * (q2 - q0)) / det;
    *dqdy = ((x1 - x0) * (q2 - q0) - (x2 - x0) * (q1 - q0)) / det;
}

/* =========================================================================
 * Wind Stress
 * ========================================================================= */

void hydro_wind_stress_apply(hydro_domain_t* domain,
                             const double* speed, double speed_scalar,
                             const double* direction, double direction_scalar,
                             hydro_int N, double time)
{
    (void)time;
    if (N < 1) return;

    const double const_wind = HYDRO_WIND_CONST;
    double* xmom_up = domain->xmom_explicit_update;
    double* ymom_up = domain->ymom_explicit_update;

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (hydro_int k = 0; k < N; k++)
    {
        double s = speed ? speed[k] : speed_scalar;
        double phi_deg = direction ? direction[k] : direction_scalar;

        /* Convert to radians */
        double phi = phi_deg * M_PI / 180.0;

        /* Wind velocity components */
        double u = s * cos(phi);
        double v = s * sin(phi);

        /* Wind stress magnitude */
        double S = const_wind * sqrt(u * u + v * v);

        /* Add to momentum explicit_update */
        xmom_up[k] += S * u;
        ymom_up[k] += S * v;
    }
}

/* =========================================================================
 * Barometric Pressure Gradient
 * ========================================================================= */

void hydro_barometric_pressure_apply(hydro_domain_t* domain,
                                     const double* pressure,
                                     hydro_int N_nodes, double time)
{
    (void)time;
    hydro_int N = domain->number_of_elements;
    if (N < 1 || !pressure) return;

    double* xmom_up = domain->xmom_explicit_update;
    double* ymom_up = domain->ymom_explicit_update;
    double* stage_c = domain->stage_centroid_values;
    double* bed_c = domain->bed_centroid_values;
    hydro_int* tri = domain->triangles;
    double* vc = domain->vertex_coordinates;

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (hydro_int k = 0; k < N; k++)
    {
        hydro_int k3 = 3 * k, k6 = 6 * k;
        hydro_int v0 = tri[k3], v1 = tri[k3 + 1], v2 = tri[k3 + 2];

        /* Pressure at three vertices */
        double p0 = (v0 < N_nodes) ? pressure[v0] : 0.0;
        double p1 = (v1 < N_nodes) ? pressure[v1] : 0.0;
        double p2 = (v2 < N_nodes) ? pressure[v2] : 0.0;

        /* Triangle vertex coordinates */
        double x0 = vc[k6], y0 = vc[k6 + 1];
        double x1 = vc[k6 + 2], y1 = vc[k6 + 3];
        double x2 = vc[k6 + 4], y2 = vc[k6 + 5];

        /* Pressure gradient on triangle */
        double px, py;
        hydro_gradient_triangle(x0, y0, x1, y1, x2, y2, p0, p1, p2, &px, &py);

        /* Water depth at centroid */
        double h = stage_c[k] - bed_c[k];
        double eps = domain->minimum_allowed_height;
        if (h < eps) h = eps;

        /* Add pressure gradient force: -h * grad(P) / rho_w */
        /* Note: pressure is in hPa, convert to Pa: 1 hPa = 100 Pa */
        double factor = -h * 100.0 / HYDRO_RHO_W;
        xmom_up[k] += factor * px;
        ymom_up[k] += factor * py;
    }
}

/* =========================================================================
 * Rainfall / Rate Operator
 * ========================================================================= */

void hydro_rainfall_apply(hydro_domain_t* domain, double rate)
{
    hydro_rate_apply(domain, &rate, NULL, domain->number_of_elements);
}

void hydro_rate_apply(hydro_domain_t* domain,
                      const double* rate,
                      const hydro_int* indices,
                      hydro_int num_indices)
{
    if (!rate || num_indices < 1) return;

    double dt = domain->timestep;
    double eps = domain->minimum_allowed_height;
    double* stage_c = domain->stage_centroid_values;
    double* bed_c = domain->bed_centroid_values;
    double* xmom_c = domain->xmom_centroid_values;
    double* ymom_c = domain->ymom_centroid_values;

    if (indices == NULL)
    {
        /* Apply to all triangles */
        hydro_int N = domain->number_of_elements;
        int all_positive = 1;
        for (hydro_int i = 0; i < N; i++)
        {
            if (rate[i] < 0.0)
            {
                all_positive = 0;
                break;
            }
        }

        if (all_positive) {
            #ifdef _OPENMP
            #pragma omp parallel for schedule(static)
            #endif
            for (hydro_int i = 0; i < N; i++) {
                stage_c[i] += dt * rate[i];
            }
        } else {
            #ifdef _OPENMP
            #pragma omp parallel for schedule(static)
            #endif
            for (hydro_int i = 0; i < N; i++) {
                double local_rate = dt * rate[i];
                double h = stage_c[i] - bed_c[i];
                if (h < eps) h = eps;

                if (local_rate >= 0.0)
                {
                    stage_c[i] += local_rate;
                }
                else
                {
                    /* Don't drain below bed */
                    if (local_rate < -h) local_rate = -h;
                    double factor = (local_rate + h) / (h + 1e-10);
                    stage_c[i] += local_rate;
                    xmom_c[i] *= factor;
                    ymom_c[i] *= factor;
                }
            }
        }
    }
    else
    {
        /* Apply to specific indices only */
        #ifdef _OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (hydro_int j = 0; j < num_indices; j++) {
            hydro_int i = indices[j];
            double local_rate = dt * rate[j];
            double h = stage_c[i] - bed_c[i];
            if (h < eps) h = eps;

            if (local_rate >= 0.0)
            {
                stage_c[i] += local_rate;
            }
            else
            {
                if (local_rate < -h) local_rate = -h;
                double factor = (local_rate + h) / (h + 1e-10);
                stage_c[i] += local_rate;
                xmom_c[i] *= factor;
                ymom_c[i] *= factor;
            }
        }
    }
}
