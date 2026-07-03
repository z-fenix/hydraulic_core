#ifndef HYDRO_FLUXES_H
#define HYDRO_FLUXES_H

#include "types.h"
#include "domain.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Compute fluxes across all edges using the central-upwind scheme.
 *
 * Postconditions:
 *   domain->explicit_update[*] = flux divergence (scaled by 1/area)
 *   domain->flux_timestep = CFL-constrained minimum timestep
 *
 * @param domain          The domain
 * @param evolve_max_timestep  Maximum allowed timestep
 * @return                The computed flux_timestep
 */
void hydro_edge_precompute(hydro_domain_t* domain);

double hydro_compute_fluxes_central(
    hydro_domain_t* domain,
    double          evolve_max_timestep);

/**
 * Central-upwind flux function for a single edge.
 *
 * @param ql          Left conserved quantities [w, uh, vh]
 * @param qr          Right conserved quantities [w, uh, vh]
 * @param h_left      Left water depth
 * @param h_right     Right water depth
 * @param hle         Left edge height
 * @param hre         Right edge height
 * @param n1, n2      Edge normal components
 * @param epsilon     Dry cell threshold
 * @param ze          Edge bed elevation (max of left/right)
 * @param g           Gravity
 * @param edgeflux    Output: flux across edge [w_flux, uh_flux, vh_flux]
 * @param max_speed   Output: maximum wave speed
 * @param pressure_flux  Output: pressure flux component
 * @param low_froude  Low Froude correction mode
 * @return 0 on success
 */
int hydro_flux_function_central(
    const double* ql,
    const double* qr,
    double        h_left,
    double        h_right,
    double        hle,
    double        hre,
    double        n1,
    double        n2,
    double        epsilon,
    double        ze,
    double        g,
    double*       edgeflux,
    double*       max_speed,
    double*       pressure_flux,
    hydro_int     low_froude);

/**
 * Positivity protection — ensure no negative water heights.
 *
 * Redistributes mass from neighbouring cells to fix negative depths.
 *
 * @param domain  The domain
 * @return        Mass error from the correction
 */
double hydro_protect(hydro_domain_t* domain);

/**
 * Fix negative cells after update_conserved_quantities.
 *
 * @param domain  The domain
 * @return        Number of cells fixed
 */
hydro_int hydro_fix_negative_cells(hydro_domain_t* domain);

/**
 * Populate edge data structure for flux computation on element k, edge i.
 */
void hydro_get_edge_data(
    const hydro_domain_t* domain,
    hydro_int             k,
    hydro_int             i,
    hydro_edge_data_t*    edge);

#ifdef __cplusplus
}
#endif

#endif /* HYDRO_FLUXES_H */
