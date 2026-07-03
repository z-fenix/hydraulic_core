#ifndef HYDRO_QUANTITY_H
#define HYDRO_QUANTITY_H

#include "types.h"
#include "domain.h"

/* ==========================================================================
 * Quantity operations — ported from quantity_openmp.c
 *
 * These functions operate on the domain's quantity arrays directly.
 * ========================================================================== */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Update conserved quantities: Q_new = (Q_old + dt*E) / (1 - dt*SI)
 * where E = explicit_update, SI = semi_implicit_update / Q_old.
 * Resets explicit_update and semi_implicit_update to zero after update.
 *
 * @param domain   The domain
 * @param timestep The timestep to advance
 */
void hydro_quantity_update(hydro_domain_t* domain, double timestep);

/**
 * First-order extrapolation: set all edge_values = centroid_values.
 * Used when spatial_order == 1.
 */
void hydro_quantity_extrapolate_first_order(hydro_domain_t* domain);

/**
 * Second-order spatial reconstruction.
 * 1. Compute gradients from neighbour centroid values
 * 2. Extrapolate to vertices using gradient
 * 3. Average vertex values to get edge values
 * 4. Apply limiter
 */
void hydro_quantity_extrapolate_second_order(hydro_domain_t* domain);

/**
 * Extrapolate centroid values to edges only (skip vertices).
 * Used by DE algorithms for efficiency.
 */
void hydro_quantity_extrapolate_second_order_edge(hydro_domain_t* domain);

/**
 * Distribute edge values to vertices (averaging adjacent edges).
 */
void hydro_quantity_distribute_edges_to_vertices(hydro_domain_t* domain);

/**
 * Backup centroid values (for RK multi-stage).
 * centroid_backup = centroid
 */
void hydro_quantity_backup(hydro_domain_t* domain);

/**
 * SAXPY on conserved quantities (for RK combination).
 * centroid = a * centroid_backup + b * centroid
 * If c != 0: centroid /= c
 */
void hydro_quantity_saxpy(hydro_domain_t* domain, double a, double b, double c);

/**
 * Update derived quantities from conserved quantities:
 * height = stage - elevation
 * xvelocity = xmomentum / max(height, minimum_allowed_height)
 * yvelocity = ymomentum / max(height, minimum_allowed_height)
 */
void hydro_quantity_update_derived(hydro_domain_t* domain);

#ifdef __cplusplus
}
#endif

#endif /* HYDRO_QUANTITY_H */
