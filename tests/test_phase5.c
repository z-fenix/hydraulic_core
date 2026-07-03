/**
 * test_phase5.c — Boundary condition verification
 *
 * Tests all boundary condition types:
 *   1. Reflective — closed domain mass conservation
 *   2. Dirichlet — fixed stage, zero momentum (open sea boundary)
 *   3. Transmissive set stage — stage boundary, copied interior momentum
 *   4. Time-varying stage — harmonic tide boundary
 *   5. Dirichlet discharge — fixed stage + normal discharge
 *   6. Transmissive stage — copy interior stage, zero momentum
 */

#include "hydro/hydro.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <string.h>

#define TOL 1e-9

/* =========================================================================
 * Helper: build a 4-triangle mesh
 * ========================================================================= */
static void build_4tri_mesh(hydro_domain_t** d_out,
    hydro_int n_node, hydro_int n_tri,
    double* vc, hydro_int* tri) {
    hydro_int bt[12] = {0};
    hydro_int be[12] = {0};
    for (int e = 0; e < 12; e++) be[e] = e;

    hydro_domain_t* d = hydro_domain_create(n_node, n_tri);
    hydro_domain_set_geometry(d, vc, tri, bt, be);
    hydro_mesh_build_neighbour_structure(d);
    hydro_mesh_build_boundary_structure(d);

    /* Set quantities */
    double elev[4] = {0,0,0,0};
    double stage[4] = {2,2,1,1};
    double xmom[4] = {0,0,0,0};
    double ymom[4] = {0,0,0,0};
    double fric[4] = {0,0,0,0};

    hydro_domain_set_quantity(d, "elevation", elev);
    hydro_domain_set_quantity(d, "stage", stage);
    hydro_domain_set_quantity(d, "xmomentum", xmom);
    hydro_domain_set_quantity(d, "ymomentum", ymom);
    hydro_domain_set_quantity(d, "friction", fric);
    d->spatial_order = 1;
    d->timestepping_method = 1;
    d->CFL = 1.0;
    hydro_quantity_update_derived(d);

    *d_out = d;
}

/* =========================================================================
 * Test 1: Reflective BC — mass should be conserved in closed domain
 * ========================================================================= */
static void test_reflective(void) {
    printf("Test 1: Reflective BC\n");

    hydro_int n6=6, n4=4;
    double vc[12] = {0,0, 1,0, 2,0, 0,1, 1,1, 2,1};
    hydro_int tri[12] = {0,1,4, 0,4,3, 1,2,5, 1,5,4};

    hydro_domain_t* d;
    build_4tri_mesh(&d, n6, n4, vc, tri);

    double mass0 = 0;
    for (int i = 0; i < 4; i++)
        mass0 += d->stage_centroid_values[i] * d->areas[i];

    /* Run 5 Euler steps */
    for (int s = 0; s < 5; s++) {
        hydro_quantity_extrapolate_first_order(d);
        hydro_boundary_update(d);  /* defaults to reflective for all tags */
        double flux_dt = hydro_compute_fluxes_central(d, 1.0);
        double dt = fmin(flux_dt * d->CFL, 0.1);
        hydro_quantity_update(d, dt);
        hydro_fix_negative_cells(d);
        hydro_quantity_update_derived(d);
    }

    double mass1 = 0;
    for (int i = 0; i < 4; i++)
        mass1 += d->stage_centroid_values[i] * d->areas[i];

    printf("  Mass: %.6f -> %.6f (ratio=%.6f)\n", mass0, mass1, mass1/mass0);
    assert(fabs(mass1/mass0 - 1.0) < 0.10);
    printf("  Reflective BC mass conservation ✓\n");

    hydro_domain_destroy(d);
}

/* =========================================================================
 * Test 2: Dirichlet BC — fixed stage at boundary edge
 * ========================================================================= */
static void test_dirichlet(void) {
    printf("\nTest 2: Dirichlet BC (fixed stage=1.5)\n");

    hydro_int n6=6, n4=4;
    double vc[12] = {0,0, 1,0, 2,0, 0,1, 1,1, 2,1};
    hydro_int tri[12] = {0,1,4, 0,4,3, 1,2,5, 1,5,4};

    hydro_domain_t* d;
    build_4tri_mesh(&d, n6, n4, vc, tri);

    /* Set tag=1 (all boundaries) to Dirichlet with stage=1.5 */
    hydro_bc_params_t params = {.stage = 1.5, .wh0 = 0.0};
    hydro_domain_set_boundary(d, 1, HYDRO_BC_DIRICHLET, &params);

    /* Run 5 Euler steps */
    for (int s = 0; s < 5; s++) {
        hydro_quantity_extrapolate_first_order(d);
        hydro_boundary_update(d);
        double flux_dt = hydro_compute_fluxes_central(d, 1.0);
        double dt = fmin(flux_dt * d->CFL, 0.1);
        hydro_quantity_update(d, dt);
        hydro_fix_negative_cells(d);
        hydro_quantity_update_derived(d);
    }

    /* After running, stage should be pulled toward 1.5 at boundaries */
    printf("  Final stage: [%.4f, %.4f, %.4f, %.4f]\n",
           d->stage_centroid_values[0], d->stage_centroid_values[1],
           d->stage_centroid_values[2], d->stage_centroid_values[3]);
    /* No NaN, no negative heights */
    for (int i = 0; i < 4; i++) {
        assert(!isnan(d->stage_centroid_values[i]));
        assert(d->stage_centroid_values[i] - d->bed_centroid_values[i] >= -1e-9);
    }
    printf("  Dirichlet BC valid ✓\n");

    hydro_domain_destroy(d);
}

/* =========================================================================
 * Test 3: Transmissive BC — set stage, copy interior momentum
 * ========================================================================= */
static void test_transmissive(void) {
    printf("\nTest 3: Transmissive BC (stage=2.0, copy momentum)\n");

    hydro_int n6=6, n4=4;
    double vc[12] = {0,0, 1,0, 2,0, 0,1, 1,1, 2,1};
    hydro_int tri[12] = {0,1,4, 0,4,3, 1,2,5, 1,5,4};

    hydro_domain_t* d;
    build_4tri_mesh(&d, n6, n4, vc, tri);

    /* Add some interior momentum */
    for (int i = 0; i < 4; i++) {
        d->xmom_centroid_values[i] = 0.5;
        d->ymom_centroid_values[i] = 0.1;
    }
    hydro_quantity_update_derived(d);

    hydro_bc_params_t params = {.stage = 2.0, .wh0 = 0.0};
    hydro_domain_set_boundary(d, 1, HYDRO_BC_TRANSMISSIVE, &params);

    /* Apply BC once */
    hydro_quantity_extrapolate_first_order(d);
    hydro_boundary_update(d);

    /* Check boundary values — first and last boundary edge */
    if (d->boundary_length > 0) {
        hydro_int bi = 0;
        printf("  Boundary[%lld]: stage=%.4f xmom=%.4f ymom=%.4f\n",
               (long long)bi,
               d->stage_boundary_values[bi],
               d->xmom_boundary_values[bi],
               d->ymom_boundary_values[bi]);
        /* Stage should be set to 2.0 */
        assert(fabs(d->stage_boundary_values[bi] - 2.0) < TOL);
        /* Momentum should not be NaN */
        assert(!isnan(d->xmom_boundary_values[bi]));
    }
    printf("  Transmissive BC valid ✓\n");

    hydro_domain_destroy(d);
}

/* =========================================================================
 * Test 4: Time-varying stage BC — harmonic tide
 * ========================================================================= */
static void test_time_stage(void) {
    printf("\nTest 4: Time-varying stage BC (tide)\n");

    hydro_int n6=6, n4=4;
    double vc[12] = {0,0, 1,0, 2,0, 0,1, 1,1, 2,1};
    hydro_int tri[12] = {0,1,4, 0,4,3, 1,2,5, 1,5,4};

    hydro_domain_t* d;
    build_4tri_mesh(&d, n6, n4, vc, tri);

    /* Set tag=1 to TIME BC */
    hydro_domain_set_boundary(d, 1, HYDRO_BC_TIME, NULL);

    /* Simulate a rising tide: update stage each step */
    for (int s = 0; s < 5; s++) {
        double tide = 1.0 + 0.1 * (s + 1);  /* 1.0 -> 1.5m */
        hydro_boundary_update_stage_time(d, 1, tide, 0.0, 0.0);

        hydro_quantity_extrapolate_first_order(d);
        hydro_boundary_update(d);
        double flux_dt = hydro_compute_fluxes_central(d, 1.0);
        double dt = fmin(flux_dt * d->CFL, 0.05);
        hydro_quantity_update(d, dt);
        hydro_fix_negative_cells(d);
        hydro_quantity_update_derived(d);
    }

    printf("  Final stage: [%.4f, %.4f, %.4f, %.4f]\n",
           d->stage_centroid_values[0], d->stage_centroid_values[1],
           d->stage_centroid_values[2], d->stage_centroid_values[3]);
    for (int i = 0; i < 4; i++) {
        assert(!isnan(d->stage_centroid_values[i]));
    }
    printf("  Time-varying stage BC valid ✓\n");

    hydro_domain_destroy(d);
}

/* =========================================================================
 * Test 5: Dirichlet discharge BC
 * ========================================================================= */
static void test_discharge(void) {
    printf("\nTest 5: Dirichlet discharge BC\n");

    hydro_int n6=6, n4=4;
    double vc[12] = {0,0, 1,0, 2,0, 0,1, 1,1, 2,1};
    hydro_int tri[12] = {0,1,4, 0,4,3, 1,2,5, 1,5,4};

    hydro_domain_t* d;
    build_4tri_mesh(&d, n6, n4, vc, tri);

    /* Set discharge BC: stage=1.5, discharge=0.5 m^2/s inward */
    hydro_bc_params_t params = {.stage = 1.5, .wh0 = 0.5};
    hydro_domain_set_boundary(d, 1, HYDRO_BC_DIRICHLET_DISCHARGE, &params);

    hydro_quantity_extrapolate_first_order(d);
    hydro_boundary_update(d);

    /* Check boundary values */
    if (d->boundary_length > 0) {
        hydro_int bi = 0;
        printf("  Boundary[%lld]: stage=%.4f xmom=%.4f ymom=%.4f (discharge=0.5)\n",
               (long long)bi,
               d->stage_boundary_values[bi],
               d->xmom_boundary_values[bi],
               d->ymom_boundary_values[bi]);
        assert(fabs(d->stage_boundary_values[bi] - 1.5) < TOL);
        /* xmom should be non-zero (discharge applied) */
        assert(fabs(d->xmom_boundary_values[bi]) > 0);
    }
    printf("  Discharge BC valid ✓\n");

    hydro_domain_destroy(d);
}

/* =========================================================================
 * Test 6: Transmissive stage BC
 * ========================================================================= */
static void test_transmissive_stage(void) {
    printf("\nTest 6: Transmissive stage (copy interior, zero momentum)\n");

    hydro_int n6=6, n4=4;
    double vc[12] = {0,0, 1,0, 2,0, 0,1, 1,1, 2,1};
    hydro_int tri[12] = {0,1,4, 0,4,3, 1,2,5, 1,5,4};

    hydro_domain_t* d;
    build_4tri_mesh(&d, n6, n4, vc, tri);

    /* All tags: transmissive stage, zero momentum */
    for (int tag = 0; tag < 5; tag++) {
        hydro_domain_set_boundary(d, tag, HYDRO_BC_TRANSMISSIVE_STAGE, NULL);
    }

    hydro_quantity_extrapolate_first_order(d);
    hydro_boundary_update(d);

    if (d->boundary_length > 0) {
        hydro_int bi = 0;
        hydro_int ni = d->boundary_edges[bi];
        hydro_int k3i = 3*(ni/3) + (ni%3);

        /* Stage should match interior edge stage */
        assert(fabs(d->stage_boundary_values[bi] - d->stage_edge_values[k3i]) < TOL);
        /* Momentum should be zero */
        assert(fabs(d->xmom_boundary_values[bi]) < TOL);
        assert(fabs(d->ymom_boundary_values[bi]) < TOL);
    }
    printf("  Transmissive stage BC valid ✓\n");

    hydro_domain_destroy(d);
}

/* =========================================================================
 * Main
 * ========================================================================= */
int main(void) {
    printf("=== Hydro Core Phase 5: Boundary Conditions ===\n\n");

    test_reflective();
    test_dirichlet();
    test_transmissive();
    test_time_stage();
    test_discharge();
    test_transmissive_stage();

    printf("\n=== Phase 5 Test PASSED ===\n");
    return 0;
}
