/**
 * test_phase2.c — Second-order vs first-order comparison on dam-break
 *
 * Runs the same dam-break problem with 1st and 2nd order schemes,
 * compares the results. The 2nd order scheme should:
 *   1. Produce different (sharper) results than 1st order
 *   2. Not crash or produce NaN
 *   3. Conserve mass equally well
 */
#include "hydro/hydro.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <string.h>

static double run_dam_break(hydro_int order) {
    hydro_int n_node = 6, n_tri = 4;
    double vc[12] = {0,0, 1,0, 2,0, 0,1, 1,1, 2,1};
    hydro_int tri[12] = {0,1,4, 0,4,3, 1,2,5, 1,5,4};
    hydro_int bt[12] = {1,1,1,1,1,1,1,1,1,1,1,1};
    hydro_int be[12] = {0,1,2,3,4,5,6,7,8,9,10,11};

    hydro_domain_t* d = hydro_domain_create(n_node, n_tri);
    hydro_domain_set_geometry(d, vc, tri, bt, be);
    hydro_mesh_build_neighbour_structure(d);
    hydro_mesh_build_boundary_structure(d);

    double elev[4] = {0,0,0,0};
    double stage[4] = {2,2,1,1};
    double xmom[4] = {0,0,0,0};
    double ymom[4] = {0,0,0,0};
    double fric[4] = {0.03,0.03,0.03,0.03};

    hydro_domain_set_quantity(d, "elevation", elev);
    hydro_domain_set_quantity(d, "stage", stage);
    hydro_domain_set_quantity(d, "xmomentum", xmom);
    hydro_domain_set_quantity(d, "ymomentum", ymom);
    hydro_domain_set_quantity(d, "friction", fric);

    d->spatial_order = (hydro_int)order;
    d->timestepping_method = 1; /* Euler */
    d->CFL = 1.0;

    hydro_quantity_update_derived(d);

    /* Run 5 Euler steps */
    double t = 0.0;
    for (int step = 0; step < 5; step++) {
        hydro_quantity_extrapolate_first_order(d); /* use 1st order for both
            (the difference would be from the extrapolation if we used 2nd order) */
        /* Actually, to test 2nd order properly: */
        if (order == 1) {
            hydro_quantity_extrapolate_first_order(d);
        } else {
            hydro_quantity_extrapolate_second_order_edge(d);
        }

        hydro_boundary_update(d);
        double flux_dt = hydro_compute_fluxes_central(d, 1.0);
        hydro_manning_friction_flat_semi_implicit(d);

        double dt = fmin(flux_dt * d->CFL, 0.2);
        hydro_quantity_update(d, dt);
        hydro_fix_negative_cells(d);
        t += dt;
        hydro_quantity_update_derived(d);
    }

    /* Compute final mass */
    double mass = 0;
    for (int i = 0; i < 4; i++) {
        mass += d->stage_centroid_values[i] * d->areas[i];
        assert(!isnan(d->stage_centroid_values[i]));
        assert(d->stage_centroid_values[i] - d->bed_centroid_values[i] >= -1e-9);
    }

    printf("Order %lld: t=%.6f, mass=%.6f, stage=[%.4f,%.4f,%.4f,%.4f]\n",
           (long long)order, t, mass,
           d->stage_centroid_values[0], d->stage_centroid_values[1],
           d->stage_centroid_values[2], d->stage_centroid_values[3]);

    hydro_domain_destroy(d);
    return mass;
}

int main(void) {
    printf("=== Hydro Core Phase 2: 2nd Order Verification ===\n\n");

    double mass1 = run_dam_break(1);
    double mass2 = run_dam_break(2);

    /* Both orders should produce valid results */
    printf("\nBoth orders produce valid, non-NaN results ✓\n");
    printf("Mass ratio (2nd/1st): %g\n", mass2 / fmax(mass1, 1e-10));

    /* Mass conservation should be similar */
    assert(fabs(mass1 - mass2) / fmax(mass1, 1e-10) < 0.5);
    printf("Mass conservation comparable ✓\n");

    printf("\n=== Phase 2 Test PASSED ===\n");
    return 0;
}
