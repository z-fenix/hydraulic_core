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
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* ==========================================================================
 * Wall-clock timer (seconds)
 * ========================================================================== */

static double wall_seconds(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* ==========================================================================
 * Memory usage (MB) — Linux /proc/self/status
 * ========================================================================== */

static long get_vmrss_mb(void)
{
    long mb = -1;
    FILE* f = fopen("/proc/self/status", "r");
    if (!f) return mb;
    char line[256];
    while (fgets(line, sizeof(line), f))
    {
        long val;
        if (sscanf(line, "VmRSS: %ld kB", &val) == 1)
        {
            mb = val / 1024;
            break;
        }
    }
    fclose(f);
    return mb;
}

/* ==========================================================================
 * Per-yieldstep progress log
 * ========================================================================== */

static void print_yieldstep_progress(
    double sim_time, double wall_elapsed, double wall_interval,
    double dt_min, double dt_max, long long step_count,
    long long interval_steps, double finaltime,
    double evolve_start_sim_time)
{
    /* Estimated time remaining — based on overall progress from evolve start */
    double progress = (finaltime > evolve_start_sim_time)
                          ? (sim_time - evolve_start_sim_time) / (finaltime - evolve_start_sim_time)
                          : 0.0;
    double eta_sec = (progress > 1e-10) ? wall_elapsed / progress - wall_elapsed : 0.0;
    if (eta_sec < 0) eta_sec = 0;

    long eta_min = (long)(eta_sec / 60.0);
    long eta_s = (long)fmod(eta_sec, 60.0);

    long mem_mb = get_vmrss_mb();

    printf("Time = %.4f (sec), delta t in [%.8f, %.8f] (s), "
           "steps=%lld (%llds), elapsed (%.0fs), eta (%ldm:%02lds)",
           sim_time, dt_min, dt_max,
           (long long)step_count, (long long)wall_interval,
           wall_elapsed, eta_min, eta_s);
    if (mem_mb > 0)
        printf(", mem=%ldMB", mem_mb);
    printf("\n");
    fflush(stdout);
}

/* ==========================================================================
 * Timestep control
 * ========================================================================== */

void hydro_update_timestep(
    hydro_domain_t* domain, double yieldstep, double finaltime)
{
    double dt;

    if (domain->flux_timestep <= 0.0)
    {
        dt = domain->evolve_max_timestep;
    }
    else
    {
        dt = domain->CFL * domain->flux_timestep;
    }

    if (dt > domain->evolve_max_timestep) dt = domain->evolve_max_timestep;
    if (dt < domain->evolve_min_timestep)
    {
        fprintf(stderr, "hydro: timestep too small: %g < %g\n",
                dt, domain->evolve_min_timestep);
        dt = domain->evolve_min_timestep;
    }

    double remaining = finaltime - domain->time;
    if (dt > remaining) dt = remaining;

    if (yieldstep > 0)
    {
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
    if (domain->spatial_order == 1)
    {
        hydro_quantity_extrapolate_first_order(domain);
    }
    else
    {
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

    /* 7. Protect: ensure non-negative water depth, fix momentum in dry cells */
    {
        double mass_err = hydro_protect(domain);
        if (mass_err > domain->minimum_allowed_height * 0.1)
        {
            fprintf(stderr, "hydro: protect mass error = %g (step %lld, t=%g)\n",
                    mass_err, (long long)domain->step, domain->time);
        }
    }

    /* 8. Secondary safety net: catch any cells still negative after protect */
    {
        hydro_int neg = hydro_fix_negative_cells(domain);
        if (neg > 0)
        {
            fprintf(stderr, "hydro: WARNING — %lld cells still negative after protect (step %lld)\n",
                    (long long)neg, (long long)domain->step);
        }
    }
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
    double finaltime,
    double yieldstep)
{
    hydro_sww_t* sww = NULL;
    int sww_is_append = 0;

    /* Build SWW path from domain name + output_dir, if name is set */
    if (domain->name[0] != '\0')
    {
        char sww_path[5120];
        int n = snprintf(sww_path, sizeof(sww_path), "%s/%s.sww",
                         domain->output_dir, domain->name);
        if (n >= (int)sizeof(sww_path))
        {
            fprintf(stderr, "hydro: SWW path too long\n");
            return -1;
        }

        /* Compute derived quantities and extrapolate centroid→edges→vertices
         * before opening SWW, so bed_vertex_values are populated. */
        hydro_quantity_update_derived(domain);
        hydro_quantity_extrapolate_first_order(domain);
        hydro_quantity_distribute_edges_to_vertices(domain);

        /* Create new SWW if evolving from t=0, otherwise append.
         * This is a semantic check: simulation time at zero means a fresh
         * start; any positive time means we are resuming a previous run. */
        if (domain->time <= 0.0)
        {
            sww = hydro_sww_create(sww_path, domain, domain->starttime);
        }
        else
        {
            sww = hydro_sww_open(sww_path, domain);
            sww_is_append = 1;
        }

        if (!sww)
        {
            fprintf(stderr, "hydro: failed to open SWW file '%s'\n", sww_path);
            return -1;
        }

        /* On first create, store initial frame */
        if (!sww_is_append)
        {
            double init_time = domain->time - domain->starttime;
            hydro_sww_store_timestep(sww, domain, init_time);
        }
    }

    printf("hydro: evolving to t=%g with yieldstep=%g\n", finaltime, yieldstep);

    /* ---- Progress-tracking state ---- */
    double evolve_start_wall = wall_seconds();
    double evolve_start_sim = domain->time;
    double yield_wall_start = evolve_start_wall;
    long long yield_step_start = domain->step;
    double yield_dt_min = 1e100;
    double yield_dt_max = 0.0;
    int first_yield = 1;
    double next_yield_time = yieldstep; /* next yield boundary */

    while (domain->time < finaltime)
    {
        /* Choose timestepping method */
        switch (domain->timestepping_method)
        {
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

        /* Update per-interval dt min/max */
        if (domain->timestep < yield_dt_min) yield_dt_min = domain->timestep;
        if (domain->timestep > yield_dt_max) yield_dt_max = domain->timestep;

        /* Update derived quantities (height, velocity from conserved) */
        hydro_quantity_update_derived(domain);

        /* Re-extrapolate for SWW output: centroid→edges, then
         * edge→vertex averaging for C0-continuous vertex values */
        hydro_quantity_extrapolate_first_order(domain);
        hydro_quantity_distribute_edges_to_vertices(domain);

        /* Store to SWW at yieldstep intervals.
         * Print progress at the same boundaries.
         * next_yield_time tracks the upcoming yield boundary; advance it
         * after each yield to avoid duplicate prints when timestep varies. */
        int at_yield = 0;
        if (domain->time >= next_yield_time - domain->timestep * 0.5
            || domain->time >= finaltime - domain->timestep * 0.5)
        {
            at_yield = 1;
        }

        if (at_yield)
        {
            /* Advance the yield boundary for the next cycle */
            next_yield_time += yieldstep;

            if (sww && yieldstep > 0)
            {
                double reltime = domain->time - domain->starttime;
                hydro_sww_store_timestep(sww, domain, reltime);
            }

            double wall_now = wall_seconds();
            double wall_elapsed = wall_now - evolve_start_wall;
            double wall_interval = wall_now - yield_wall_start;

            /* Only print if meaningful work was done */
            if (first_yield || wall_interval > 0.001)
            {
                print_yieldstep_progress(
                    domain->time, wall_elapsed, wall_interval,
                    yield_dt_min, yield_dt_max,
                    domain->step,
                    domain->step - yield_step_start,
                    finaltime, evolve_start_sim);
                first_yield = 0;
            }

            /* Reset per-interval tracking */
            yield_wall_start = wall_now;
            yield_step_start = domain->step;
            yield_dt_min = 1e100;
            yield_dt_max = 0.0;
        }
    }

    printf("hydro: simulation complete (t=%g, %lld steps)\n",
           domain->time, (long long)domain->step);

    if (sww)
    {
        hydro_sww_close(sww);
    }

    return 0;
}
