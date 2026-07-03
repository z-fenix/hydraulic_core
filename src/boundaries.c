/**
 * boundaries.c — Boundary condition evaluation
 *
 * Ported from ANUGA:
 *   anuga/shallow_water/boundaries.py (all boundary classes)
 *   anuga/shallow_water/sw_domain_openmp.c (evaluate_reflective_segment)
 *
 * Boundary condition types supported:
 *   HYDRO_BC_REFLECTIVE           — closed wall (mass-conserving)
 *   HYDRO_BC_DIRICHLET            — fixed stage, zero momentum
 *   HYDRO_BC_TRANSMISSIVE         — set stage, copy interior momentum
 *   HYDRO_BC_TIME                 — time-varying stage, zero momentum
 *   HYDRO_BC_DIRICHLET_DISCHARGE  — fixed stage + inward normal discharge
 *   HYDRO_BC_TRANSMISSIVE_STAGE   — copy interior stage, zero momentum
 */

#include "hydro/boundaries.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* =========================================================================
 * Boundary configuration — store per-tag BC type and parameters
 * ========================================================================= */

void hydro_domain_set_boundary(
    hydro_domain_t* domain, hydro_int boundary_tag,
    hydro_bc_type_t bc_type, const hydro_bc_params_t* bc_params)
{
    if (boundary_tag < 0 || boundary_tag >= HYDRO_MAX_BOUNDARY_TAGS) {
        fprintf(stderr, "hydro: boundary tag %lld out of range [0, %d)\n",
                (long long)boundary_tag, HYDRO_MAX_BOUNDARY_TAGS);
        return;
    }

    domain->boundary_bc_type_tag[boundary_tag] = (hydro_int)bc_type;

    if (bc_params) {
        domain->boundary_stage_tag[boundary_tag] = bc_params->stage;
        domain->boundary_xmom_tag[boundary_tag]  = bc_params->wh0;
        domain->boundary_ymom_tag[boundary_tag]  = 0.0;
    } else {
        domain->boundary_stage_tag[boundary_tag] = 0.0;
        domain->boundary_xmom_tag[boundary_tag]  = 0.0;
        domain->boundary_ymom_tag[boundary_tag]  = 0.0;
    }
}

void hydro_boundary_update_stage_time(
    hydro_domain_t* domain, hydro_int boundary_tag,
    double stage, double xmom, double ymom)
{
    if (boundary_tag < 0 || boundary_tag >= HYDRO_MAX_BOUNDARY_TAGS) return;
    domain->boundary_stage_tag[boundary_tag] = stage;
    domain->boundary_xmom_tag[boundary_tag]  = xmom;
    domain->boundary_ymom_tag[boundary_tag]  = ymom;
}

/* =========================================================================
 * Per-segment evaluators (the computational kernels)
 * ========================================================================= */

void hydro_boundary_evaluate_reflective_segment(
    hydro_domain_t* domain, hydro_int num_edges,
    const hydro_int* edge_segment, const hydro_int* vol_ids,
    const hydro_int* edge_ids)
{
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (hydro_int k = 0; k < num_edges; k++) {
        hydro_int seg_id = edge_segment[k];
        hydro_int vid    = vol_ids[k];
        hydro_int eid    = edge_ids[k];

        /* Edge normal */
        double n1 = domain->normals[vid*6 + 2*eid];
        double n2 = domain->normals[vid*6 + 2*eid + 1];

        /* Stage/bed/height: copy from interior */
        domain->stage_boundary_values[seg_id] =
            domain->stage_edge_values[3*vid + eid];
        domain->bed_boundary_values[seg_id] =
            domain->bed_edge_values[3*vid + eid];
        domain->height_boundary_values[seg_id] =
            domain->height_edge_values[3*vid + eid];

        /* Momentum: reflect — normal component flips sign */
        double q1 = domain->xmom_edge_values[3*vid + eid];
        double q2 = domain->ymom_edge_values[3*vid + eid];
        double r1 = -q1*n1 - q2*n2;
        double r2 = -q1*n2 + q2*n1;

        domain->xmom_boundary_values[seg_id] = n1*r1 - n2*r2;
        domain->ymom_boundary_values[seg_id] = n2*r1 + n1*r2;

        /* Velocity: same reflection */
        if (domain->xvelocity_edge_values && domain->yvelocity_edge_values) {
            double v1 = domain->xvelocity_edge_values[3*vid + eid];
            double v2 = domain->yvelocity_edge_values[3*vid + eid];
            double s1 = v1*n1 + v2*n2;
            double s2 = v1*n2 - v2*n1;
            domain->xvelocity_boundary_values[seg_id] = n1*s1 - n2*s2;
            domain->yvelocity_boundary_values[seg_id] = n2*s1 + n1*s2;
        }
    }
}

void hydro_boundary_evaluate_dirichlet_segment(
    hydro_domain_t* domain, hydro_int num_edges,
    const hydro_int* edge_segment, const hydro_int* vol_ids,
    const hydro_int* edge_ids, double stage)
{
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (hydro_int k = 0; k < num_edges; k++) {
        hydro_int seg_id = edge_segment[k];
        hydro_int vid    = vol_ids[k];
        hydro_int eid    = edge_ids[k];

        /* Set external stage */
        domain->stage_boundary_values[seg_id] = stage;

        /* Copy bed from interior */
        domain->bed_boundary_values[seg_id] =
            domain->bed_edge_values[3*vid + eid];

        /* Height = max(stage - bed, 0) */
        double h = stage - domain->bed_boundary_values[seg_id];
        if (h < 0) h = 0;
        domain->height_boundary_values[seg_id] = h;

        /* Zero momentum at boundary */
        domain->xmom_boundary_values[seg_id] = 0.0;
        domain->ymom_boundary_values[seg_id] = 0.0;
    }
}

void hydro_boundary_evaluate_transmissive_segment(
    hydro_domain_t* domain, hydro_int num_edges,
    const hydro_int* edge_segment, const hydro_int* vol_ids,
    const hydro_int* edge_ids,
    double external_stage, int set_stage)
{
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (hydro_int k = 0; k < num_edges; k++) {
        hydro_int seg_id = edge_segment[k];
        hydro_int vid    = vol_ids[k];
        hydro_int eid    = edge_ids[k];
        hydro_int k3i    = 3*vid + eid;

        /* Normal */
        double n1 = domain->normals[vid*6 + 2*eid];
        double n2 = domain->normals[vid*6 + 2*eid + 1];

        /* Stage: either external (set) or interior (copy) */
        if (set_stage) {
            domain->stage_boundary_values[seg_id] = external_stage;
            domain->bed_boundary_values[seg_id] =
                domain->bed_edge_values[k3i];
            double h = external_stage - domain->bed_boundary_values[seg_id];
            if (h < 0) h = 0;
            domain->height_boundary_values[seg_id] = h;
        } else {
            domain->stage_boundary_values[seg_id] =
                domain->stage_edge_values[k3i];
            domain->bed_boundary_values[seg_id] =
                domain->bed_edge_values[k3i];
            domain->height_boundary_values[seg_id] =
                domain->height_edge_values[k3i];
        }

        /* Transmissive momentum: normal component preserved, tangential zeroed */
        double q1 = domain->xmom_edge_values[k3i];
        double q2 = domain->ymom_edge_values[k3i];
        double ndotq = n1*q1 + n2*q2;

        domain->xmom_boundary_values[seg_id] = ndotq * n1;
        domain->ymom_boundary_values[seg_id] = ndotq * n2;
    }
}

void hydro_boundary_evaluate_discharge_segment(
    hydro_domain_t* domain, hydro_int num_edges,
    const hydro_int* edge_segment, const hydro_int* vol_ids,
    const hydro_int* edge_ids,
    double stage, double discharge)
{
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (hydro_int k = 0; k < num_edges; k++) {
        hydro_int seg_id = edge_segment[k];
        hydro_int vid    = vol_ids[k];
        hydro_int eid    = edge_ids[k];

        /* Edge normal (outward) → inward normal = -normal */
        double nx = domain->normals[vid*6 + 2*eid];
        double ny = domain->normals[vid*6 + 2*eid + 1];

        /* Set stage */
        domain->stage_boundary_values[seg_id] = stage;
        domain->bed_boundary_values[seg_id] =
            domain->bed_edge_values[3*vid + eid];
        double h = stage - domain->bed_boundary_values[seg_id];
        if (h < 0) h = 0;
        domain->height_boundary_values[seg_id] = h;

        /* Set momentum in inward normal direction */
        domain->xmom_boundary_values[seg_id] = -discharge * nx;
        domain->ymom_boundary_values[seg_id] = -discharge * ny;
    }
}

/* =========================================================================
 * Main boundary update — dispatches per boundary edge based on its tag
 * ========================================================================= */

void hydro_boundary_update(hydro_domain_t* domain) {
    hydro_int bl = domain->boundary_length;
    if (bl == 0) return;

    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (hydro_int bi = 0; bi < bl; bi++) {
        hydro_int tag = domain->boundary_tags[bi];
        if (tag < 0 || tag >= HYDRO_MAX_BOUNDARY_TAGS) tag = 0;

        hydro_int bc_type = domain->boundary_bc_type_tag[tag];
        hydro_int ni = domain->boundary_edges[bi];
        hydro_int k  = ni / 3;
        hydro_int ei = ni % 3;
        hydro_int k3i = 3*k + ei;

        double nx = domain->normals[2*k3i];
        double ny = domain->normals[2*k3i + 1];

        switch (bc_type) {

        case HYDRO_BC_REFLECTIVE:
        default:
            /* Reflective BC: stage/bed/height = interior, momentum reflected */
            domain->stage_boundary_values[bi]  = domain->stage_edge_values[k3i];
            domain->bed_boundary_values[bi]    = domain->bed_edge_values[k3i];
            domain->height_boundary_values[bi] = domain->height_edge_values[k3i];

            {
                double qx = domain->xmom_edge_values[k3i];
                double qy = domain->ymom_edge_values[k3i];
                double dot = qx*nx + qy*ny;
                domain->xmom_boundary_values[bi] = qx - 2.0*dot*nx;
                domain->ymom_boundary_values[bi] = qy - 2.0*dot*ny;
            }
            break;

        case HYDRO_BC_DIRICHLET:
            /* Fixed stage, zero momentum */
            {
                double stage_ext = domain->boundary_stage_tag[tag];
                domain->stage_boundary_values[bi]  = stage_ext;
                domain->bed_boundary_values[bi]    = domain->bed_edge_values[k3i];
                double h = stage_ext - domain->bed_boundary_values[bi];
                if (h < 0) h = 0;
                domain->height_boundary_values[bi] = h;
                domain->xmom_boundary_values[bi]   = 0.0;
                domain->ymom_boundary_values[bi]   = 0.0;
            }
            break;

        case HYDRO_BC_TRANSMISSIVE:
            /* Set stage from tag, copy interior momentum */
            {
                double stage_ext = domain->boundary_stage_tag[tag];
                domain->stage_boundary_values[bi]  = stage_ext;
                domain->bed_boundary_values[bi]    = domain->bed_edge_values[k3i];
                double h = stage_ext - domain->bed_boundary_values[bi];
                if (h < 0) h = 0;
                domain->height_boundary_values[bi] = h;

                /* Transmissive momentum: keep normal component, zero tangential */
                double qx = domain->xmom_edge_values[k3i];
                double qy = domain->ymom_edge_values[k3i];
                double ndotq = nx*qx + ny*qy;
                domain->xmom_boundary_values[bi] = ndotq * nx;
                domain->ymom_boundary_values[bi] = ndotq * ny;
            }
            break;

        case HYDRO_BC_TIME:
            /* Time-varying stage (updated externally), zero momentum */
            {
                double stage_ext = domain->boundary_stage_tag[tag];
                domain->stage_boundary_values[bi]  = stage_ext;
                domain->bed_boundary_values[bi]    = domain->bed_edge_values[k3i];
                double h = stage_ext - domain->bed_boundary_values[bi];
                if (h < 0) h = 0;
                domain->height_boundary_values[bi] = h;
                domain->xmom_boundary_values[bi]   = 0.0;
                domain->ymom_boundary_values[bi]   = 0.0;
            }
            break;

        case HYDRO_BC_DIRICHLET_DISCHARGE:
            /* Fixed stage + inward normal discharge */
            {
                double stage_ext = domain->boundary_stage_tag[tag];
                double wh0       = domain->boundary_xmom_tag[tag];
                domain->stage_boundary_values[bi]  = stage_ext;
                domain->bed_boundary_values[bi]    = domain->bed_edge_values[k3i];
                double h = stage_ext - domain->bed_boundary_values[bi];
                if (h < 0) h = 0;
                domain->height_boundary_values[bi] = h;
                domain->xmom_boundary_values[bi]   = -wh0 * nx;
                domain->ymom_boundary_values[bi]   = -wh0 * ny;
            }
            break;

        case HYDRO_BC_TRANSMISSIVE_STAGE:
            /* Copy interior stage, zero momentum */
            domain->stage_boundary_values[bi]  = domain->stage_edge_values[k3i];
            domain->bed_boundary_values[bi]    = domain->bed_edge_values[k3i];
            domain->height_boundary_values[bi] = domain->height_edge_values[k3i];
            domain->xmom_boundary_values[bi]   = 0.0;
            domain->ymom_boundary_values[bi]   = 0.0;
            break;
        }
    }
}
