#ifndef HYDRO_OPERATORS_H
#define HYDRO_OPERATORS_H

#include "types.h"
#include "domain.h"
#include "sparse.h"

#ifdef __cplusplus
extern "C" {

#endif

/* =========================================================================
 * Rate Operator — add/remove water at a specified rate (m/s)
 * (Implemented in forcing.h/c as hydro_rate_apply / hydro_rainfall_apply)
 * ========================================================================= */

/* =========================================================================
 * Friction Operator — explicit Manning friction as exponential decay
 * ========================================================================= */

void hydro_manning_friction_explicit(hydro_domain_t* domain);

/* =========================================================================
 * Bed Shear Erosion Operator
 * ========================================================================= */

void hydro_bed_shear_erosion_apply(hydro_domain_t* domain,
                                   double threshold, double base,
                                   const hydro_int* indices,
                                   hydro_int num_indices);

/* =========================================================================
 * Set Stage / Elevation Operators
 * ========================================================================= */

void hydro_set_stage(hydro_domain_t* domain,
                     const double* stage_values,
                     const hydro_int* indices,
                     hydro_int num_indices);

void hydro_set_elevation(hydro_domain_t* domain,
                         const double* elev_values,
                         const hydro_int* indices,
                         hydro_int num_indices);

/* =========================================================================
 * Kinematic Viscosity Operator — parabolic smoothing of velocity
 *
 * Solves:  (I - dt * div(h grad)) u^{n+1} = u^n
 *          (I - dt * div(h grad)) v^{n+1} = v^n
 *
 * This is a parabolic PDE solved via the finite element method on
 * triangular meshes, using conjugate gradient.
 * ========================================================================= */

/**
 * Build the geometric structure for the elliptic operator.
 *
 * For each triangle edge, computes the coefficient:
 *   geo_values[i, edge] = -edgelength / distance(centroid_i, centroid_or_midpoint)
 *
 * Stores in domain work arrays (geo_structure_indices, geo_structure_values).
 * Called once after mesh setup.
 *
 * @param domain  Domain with mesh geometry set
 * @return 0 on success
 */
int hydro_kinematic_viscosity_build_geo_structure(hydro_domain_t* domain);

/**
 * Build the elliptic matrix L representing div(h grad).
 *
 * The matrix has exactly 4 entries per row (diagonal + up to 3 neighbours).
 * Matrix entries: L_{i,j} = -0.5*(h_i + h_j) * geo_values[i, edge]
 *
 * @param domain         Domain
 * @param diffusivity    Diffusivity values at centroids (h) [n_elements]
 * @param diffusivity_b  Diffusivity values at boundary edges [boundary_length]
 * @return CSR matrix representing div(h grad), or NULL on failure
 */
hydro_sparse_csr_t* hydro_kinematic_viscosity_build_matrix(
    hydro_domain_t* domain,
    const double* diffusivity,
    const double* diffusivity_b);

/**
 * Update only the data values of an existing elliptic matrix.
 * The sparsity pattern (colind, rowptr) remains unchanged.
 *
 * @param domain         Domain
 * @param diffusivity    New diffusivity values [n_elements]
 * @param diffusivity_b  New boundary diffusivity [boundary_length]
 * @param L              Matrix to update (data array modified in-place)
 * @return 0 on success
 */
int hydro_kinematic_viscosity_update_matrix(
    hydro_domain_t* domain,
    const double* diffusivity,
    const double* diffusivity_b,
    hydro_sparse_csr_t* L);

/**
 * Apply one step of the kinematic viscosity operator.
 *
 * Solves (I - dt * L) * u_new = u_old for x and y velocity.
 * Updates domain->xvelocity_centroid_values and yvelocity_centroid_values.
 *
 * @param domain       Domain (must have xvelocity/yvelocity centroid values set)
 * @param diffusivity  Diffusivity (= height or user-specified) [n_elements]
 * @param dt           Accumulated timestep
 * @return 0 on success
 */
int hydro_kinematic_viscosity_apply(
    hydro_domain_t* domain,
    const double* diffusivity,
    double dt);

#ifdef __cplusplus
}
#endif

#endif /* HYDRO_OPERATORS_H */
