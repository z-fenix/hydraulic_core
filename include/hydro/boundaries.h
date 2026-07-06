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
    HYDRO_BC_TIME_SERIES   = 7,   /* flow rate Q(t) → derive stage+mom   */
} hydro_bc_type_t;

/* Parameters for boundary conditions */
typedef struct {
    double stage;    /* external stage (m), used by Dirichlet/Time BCs */
    double wh0;      /* discharge in m^2/s, used by Dirichlet_discharge */
} hydro_bc_params_t;

/* Time-series boundary data — stores Q(t) for HYDRO_BC_TIME_SERIES */
typedef struct {
    double* times;      /* [n_points] time values (seconds) */
    double* q_values;   /* [n_points] discharge values (m³/s) */
    int     n_points;
    double  default_stage; /* fallback stage when Q out of range */
} hydro_time_series_t;

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
 * Set time-series boundary data for a specific tag.
 *
 * Stores Q(t) pairs; the boundary evaluator will interpolate Q at the
 * current simulation time and derive stage + momentum from it.
 *
 * @param domain      The domain
 * @param boundary_tag Tag whose BC to update
 * @param times        Time array [n_points]
 * @param q_values     Discharge array [n_points]
 * @param n_points     Number of data points
 * @param default_stage  Fallback stage when Q is out of range
 */
void hydro_boundary_set_time_series(
    hydro_domain_t* domain,
    hydro_int       boundary_tag,
    const double*   times,
    const double*   q_values,
    int             n_points,
    double          default_stage);

/**
 * Update stage/momentum boundary values for HYDRO_BC_TIME_SERIES
 * at runtime.  Called each timestep — interpolates Q(current_time),
 * derives stage from Q using Manning's equation, and sets
 * boundary_stage_tag / boundary_xmom_tag / boundary_ymom_tag.
 *
 * The effective channel width is auto-derived from the boundary edge
 * geometry (sum of edgelengths for edges with this tag).
 *
 * @param domain      The domain
 * @param boundary_tag Tag whose BC to update
 * @param current_time  Current simulation time
 */
void hydro_boundary_update_time_series(
    hydro_domain_t* domain,
    hydro_int       boundary_tag,
    double          current_time);

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
