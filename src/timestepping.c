/**
 * timestepping.c — Time integration loop
 *
 * Ported from:
 *   anuga/shallow_water/shallow_water_domain.py
 *   anuga/abstract_2d_finite_volumes/generic_domain.py:_evolve_base()
 */

#include "hydro/timestepping.h"
#include "hydro/quantity.h"
#include "hydro/fluxes.h"
#include "hydro/boundaries.h"
#include "hydro/friction.h"
#include "hydro/sww.h"
#include "hydro/config.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* ==========================================================================
 * Timestep control
 * ========================================================================== */

void hydro_update_timestep(
    hydro_domain_t* domain, double yieldstep, double finaltime)
{
    double dt;

    if (domain->flux_timestep <= 0.0) {
        dt = domain->evolve_max_timestep;
    } else {
        dt = domain->CFL * domain->flux_timestep;
    }

    if (dt > domain->evolve_max_timestep) dt = domain->evolve_max_timestep;
    if (dt < domain->evolve_min_timestep) {
        fprintf(stderr, "hydro: timestep too small: %g < %g\n",
                dt, domain->evolve_min_timestep);
        dt = domain->evolve_min_timestep;
    }

    double remaining = finaltime - domain->time;
    if (dt > remaining) dt = remaining;

    if (yieldstep > 0) {
        domain->yieldstep_counter++;
        double next_yield = domain->yieldstep_counter * yieldstep;
        double to_yield = next_yield - domain->time;
        if (dt > to_yield) dt = to_yield;
    }

    domain->timestep = dt;
}

/* ==========================================================================
 * Single-step integrators
 * ========================================================================== */

void hydro_evolve_one_euler_step(
    hydro_domain_t* domain, double yieldstep, double finaltime)
{
    /* 1. Extrapolate centroid → edges (1st or 2nd order) */
    if (domain->spatial_order == 1) {
        hydro_quantity_extrapolate_first_order(domain);
    } else {
        hydro_quantity_extrapolate_second_order_edge(domain);
    }

    /* 2. Apply boundary conditions */
    hydro_boundary_update(domain);

    /* 3. Compute fluxes */
    domain->flux_timestep = hydro_compute_fluxes_central(
        domain, domain->evolve_max_timestep);

    /* 4. Compute forcing terms (friction) */
    hydro_manning_friction_flat_semi_implicit(domain);

    /* 5. Update timestep (CFL + yield/final alignment) */
    hydro_update_timestep(domain, yieldstep, finaltime);

    /* 6. Update conserved quantities */
    hydro_quantity_update(domain, domain->timestep);

    /* 7. Fix any negative cells that resulted */
    hydro_fix_negative_cells(domain);
}

void hydro_evolve_one_rk2_step(
    hydro_domain_t* domain, double yieldstep, double finaltime)
{
    hydro_quantity_backup(domain);
    hydro_evolve_one_euler_step(domain, yieldstep, finaltime);
    hydro_quantity_saxpy(domain, 0.5, 0.5, 0.0);
}

void hydro_evolve_one_rk3_step(
    hydro_domain_t* domain, double yieldstep, double finaltime)
{
    hydro_quantity_backup(domain);
    hydro_evolve_one_euler_step(domain, yieldstep, finaltime);
    hydro_quantity_saxpy(domain, 0.25, 0.75, 0.0);

    hydro_evolve_one_euler_step(domain, yieldstep, finaltime);
    hydro_quantity_saxpy(domain, 2.0, 1.0, 3.0);
}

/* ==========================================================================
 * Main evolve loop
 * ========================================================================== */

int hydro_domain_evolve(
    hydro_domain_t* domain,
    double          finaltime,
    double          yieldstep,
    const char*     output_sww_path)
{
    hydro_sww_t* sww = NULL;

    /* Open SWW file if output requested */
    if (output_sww_path && output_sww_path[0] != '\0') {
        sww = hydro_sww_create(output_sww_path, domain, domain->starttime);
        if (!sww) {
            fprintf(stderr, "hydro: failed to create SWW file '%s'\n",
                    output_sww_path);
            return -1;
        }

        /* Compute derived quantities (height, velocity) before first step */
        hydro_quantity_update_derived(domain);

        /* Do initial extrapolation and store initial timestep */
        hydro_quantity_extrapolate_first_order(domain);
        hydro_quantity_distribute_edges_to_vertices(domain);
        hydro_sww_store_timestep(sww, domain, 0.0);
    }

    printf("hydro: evolving to t=%g with yieldstep=%g\n", finaltime, yieldstep);

    while (domain->time < finaltime) {
        /* Choose timestepping method */
        switch (domain->timestepping_method) {
        case 1:
            hydro_evolve_one_euler_step(domain, yieldstep, finaltime);
            break;
        case 3:
            hydro_evolve_one_rk3_step(domain, yieldstep, finaltime);
            break;
        case 2:
        default:
            hydro_evolve_one_rk2_step(domain, yieldstep, finaltime);
            break;
        }

        /* Advance time */
        domain->time += domain->timestep;
        domain->step++;

        /* Update derived quantities (height, velocity from conserved) */
        hydro_quantity_update_derived(domain);

        /* Re-extrapolate for SWW output: centroid→edges, then
         * edge→vertex averaging for C0-continuous vertex values */
        hydro_quantity_extrapolate_first_order(domain);
        hydro_quantity_distribute_edges_to_vertices(domain);

        /* Store to SWW at yieldstep intervals */
        if (sww && yieldstep > 0) {
            double reltime = domain->time - domain->starttime;
            if (fabs(fmod(domain->time, yieldstep)) < domain->timestep * 0.5
                || domain->time >= finaltime) {
                hydro_sww_store_timestep(sww, domain, reltime);
                printf("hydro: t=%g (step %lld, dt=%g)\n",
                       domain->time, (long long)domain->step, domain->timestep);
            }
        }
    }

    printf("hydro: simulation complete (t=%g, %lld steps)\n",
           domain->time, (long long)domain->step);

    if (sww) {
        hydro_sww_close(sww);
    }

    return 0;
}
