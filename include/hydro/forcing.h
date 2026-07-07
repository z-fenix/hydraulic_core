#ifndef HYDRO_FORCING_H
#define HYDRO_FORCING_H

#include "types.h"
#include "domain.h"

#ifdef __cplusplus
extern "C" {

#endif

/* =========================================================================
 * Wind Stress
 *
 * Applies wind stress to x and y momentum based on wind speed and direction.
 * Formula: tau = eta_w * rho_a/rho_w * |U_wind| * U_wind
 * ========================================================================= */

/* Apply wind stress to domain momentum explicit_update.
 * speed: wind speed in m/s at each centroid (or scalar, or callable)
 * direction: wind direction in degrees (meteorological) at each centroid
 * N: number of values (must equal domain->number_of_elements)
 *
 * If speed is NULL, a uniform speed_scalar is used for all centroids.
 * If direction is NULL, a uniform direction_scalar is used.
 */
void hydro_wind_stress_apply(hydro_domain_t* domain,
                             const double* speed, double speed_scalar,
                             const double* direction, double direction_scalar,
                             hydro_int N, double time);

/* =========================================================================
 * Barometric Pressure Gradient
 *
 * Applies pressure gradient force to momentum.
 * Formula: d(mom)/dt = -h * grad(P_atm) / rho_w
 * Pressure values are defined at mesh vertices.
 * ========================================================================= */

/* Apply barometric pressure gradient to domain momentum explicit_update.
 * pressure: pressure values at each vertex (hPa or Pa)
 * N: number of vertices (must equal domain->number_of_nodes)
 *
 * For each triangle, computes dp/dx, dp/dy from vertex pressures
 * and adds -height * grad(p) / rho_w to momentum explicit_update.
 */
void hydro_barometric_pressure_apply(hydro_domain_t* domain,
                                     const double* pressure,
                                     hydro_int N, double time);

/* =========================================================================
 * Rainfall / Inflow (General Forcing)
 *
 * Adds or removes water from the domain at a specified rate.
 * Positive rate = inflow (rain), negative rate = outflow (drain).
 * ========================================================================= */

/* Apply a uniform rainfall rate (m/s) to the entire domain.
 * rate: rainfall rate in m/s (positive = rain, negative = evaporation)
 * The rate is added directly to stage centroid values.
 * For negative rates, momentum is scaled to prevent velocity blowup.
 */
void hydro_rainfall_apply(hydro_domain_t* domain, double rate);

/* Apply a spatially-varying rate (m/s) to specified triangles.
 * rate: array of rate values (length = len(indices) or N_total)
 * indices: array of triangle indices to apply rate to
 * num_indices: length of indices array
 * If indices is NULL, applies to all triangles.
 */
void hydro_rate_apply(hydro_domain_t* domain,
                      const double* rate,
                      const hydro_int* indices,
                      hydro_int num_indices);

/* =========================================================================
 * Gradient Computation (utility, used by pressure and slope limiters)
 * ========================================================================= */

/* Compute gradient of a scalar field on a triangle.
 * x0,y0, x1,y1, x2,y2: triangle vertex coordinates
 * q0, q1, q2: scalar values at vertices
 * dqdx, dqdy: output gradient components
 */
void hydro_gradient_triangle(double x0, double y0,
                             double x1, double y1,
                             double x2, double y2,
                             double q0, double q1, double q2,
                             double* dqdx, double* dqdy);

#ifdef __cplusplus
}
#endif

#endif /* HYDRO_FORCING_H */
