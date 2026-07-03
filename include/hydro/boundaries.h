#ifndef HYDRO_BOUNDARIES_H
#define HYDRO_BOUNDARIES_H

#include "types.h"
#include "domain.h"

/* =========================================================================
 * Boundary condition types
 *
 * Ported from anuga/shallow_water/boundaries.py
 * ========================================================================= */

typedef enum {
    HYDRO_BC_NONE          = 0,   /* unset / interior  */
    HYDRO_BC_REFLECTIVE    = 1,   /* reflective wall   */
    HYDRO_BC_DIRICHLET     = 2,   /* fixed stage, zero momentum          */
    HYDRO_BC_TRANSMISSIVE  = 3,   /* copy interior momentum, set stage   */
    HYDRO_BC_TIME          = 4,   /* time-varying stage, zero momentum   */
    HYDRO_BC_DIRICHLET_DISCHARGE = 5,  /* fixed stage + inward normal discharge */
    HYDRO_BC_TRANSMISSIVE_STAGE   = 6,  /* transmissive stage, zero momentum   */
} hydro_bc_type_t;

/* Parameters for boundary conditions */
typedef struct {
    double stage;    /* external stage (m), used by Dirichlet/Time BCs */
    double wh0;      /* discharge in m^2/s, used by Dirichlet_discharge */
} hydro_bc_params_t;

#ifdef __cplusplus
extern "C" {
#endif

/* =========================================================================
 * Boundary configuration
 * ========================================================================= */

/**
 * Set boundary condition type and parameters for a specific tag.
 *
 * All boundary edges with the given tag will use this BC.
 * If tag is 0, sets the default BC for un-tagged edges.
 *
 * @param domain      The domain
 * @param boundary_tag Tag value (1..127), or 0 for default
 * @param bc_type     Boundary condition type
 * @param bc_params   BC parameters (stage, momentum); NULL = use zeros
 */
void hydro_domain_set_boundary(
    hydro_domain_t* domain,
    hydro_int       boundary_tag,
    hydro_bc_type_t bc_type,
    const hydro_bc_params_t* bc_params);

/**
 * Update stage/height/momentum boundary values for Dirichlet/Time BCs
 * at runtime.  Called each timestep when the external stage changes.
 *
 * @param domain      The domain
 * @param boundary_tag Tag whose BC to update
 * @param stage       External stage value
 * @param xmom        External x-momentum (for discharge BC, 0 otherwise)
 * @param ymom        External y-momentum (for discharge BC, 0 otherwise)
 */
void hydro_boundary_update_stage_time(
    hydro_domain_t* domain,
    hydro_int       boundary_tag,
    double          stage,
    double          xmom,
    double          ymom);

/* =========================================================================
 * Boundary evaluation (called once per timestep)
 * ========================================================================= */

/**
 * Apply all boundary conditions — update edge values on boundary edges.
 *
 * Dispatches to the correct BC evaluation based on per-tag BC type.
 */
void hydro_boundary_update(hydro_domain_t* domain);

/* =========================================================================
 * Per-segment boundary evaluators (can be called standalone)
 * ========================================================================= */

/**
 * Evaluate reflective BC on a segment of boundary edges.
 * stage/bed/height = interior values, momentum reflected across normal.
 */
void hydro_boundary_evaluate_reflective_segment(
    hydro_domain_t* domain,
    hydro_int       num_edges,
    const hydro_int* edge_segment,
    const hydro_int* vol_ids,
    const hydro_int* edge_ids);

/**
 * Evaluate Dirichlet BC: fixed external stage, zero momentum.
 */
void hydro_boundary_evaluate_dirichlet_segment(
    hydro_domain_t* domain,
    hydro_int       num_edges,
    const hydro_int* edge_segment,
    const hydro_int* vol_ids,
    const hydro_int* edge_ids,
    double          stage);

/**
 * Evaluate Transmissive BC: copy interior momentum, optionally set stage.
 * If set_stage != 0, stage = external_stage; otherwise stage = interior.
 */
void hydro_boundary_evaluate_transmissive_segment(
    hydro_domain_t* domain,
    hydro_int       num_edges,
    const hydro_int* edge_segment,
    const hydro_int* vol_ids,
    const hydro_int* edge_ids,
    double          external_stage,
    int             set_stage);

/**
 * Evaluate Dirichlet discharge BC: fixed stage + momentum in inward normal.
 */
void hydro_boundary_evaluate_discharge_segment(
    hydro_domain_t* domain,
    hydro_int       num_edges,
    const hydro_int* edge_segment,
    const hydro_int* vol_ids,
    const hydro_int* edge_ids,
    double          stage,
    double          discharge);

#ifdef __cplusplus
}
#endif

#endif /* HYDRO_BOUNDARIES_H */
