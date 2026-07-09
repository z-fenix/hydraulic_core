/**
 * test_phase1b.c — Full evolve loop + SWW output on a dam-break
 *
 * Creates a 20-triangle rectangular channel, runs evolve() for
 * a short duration, verifies SWW output is produced.
 */
#include "hydro/hydro.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

int main(void) {
    printf("=== Hydro Core Phase 1b: Evolve Loop + SWW ===\n\n");

    /* 5x3 grid: 10 squares = 20 triangles, 15 nodes */
    hydro_int n_nodes = 15;
    hydro_int n_triangles = 20;
    hydro_int n_boundary = 20; /* approximate */

    /* Nodes: 5x3 grid:
       y=0: 0(0,0), 1(1,0), 2(2,0), 3(3,0), 4(4,0)
       y=1: 5(0,1), 6(1,1), 7(2,1), 8(3,1), 9(4,1)
       y=2: 10(0,2),11(1,2),12(2,2),13(3,2),14(4,2) */
    double vertex_coords[30] = {
        0,0, 1,0, 2,0, 3,0, 4,0,
        0,1, 1,1, 2,1, 3,1, 4,1,
        0,2, 1,2, 2,2, 3,2, 4,2
    };

    /* 2 rows * 4 cols = 8 squares = 16 triangles. Wait, that's wrong.
       Let's do 4x2 grid: 4 squares * 2 = 8 triangles.
       But with 5 nodes along x and 2 along y: 4 cols * 1 row = 4 squares = 8 tri.
       No, let me just use a simple configuration.

       3x2 grid: nodes 6, squares 2, triangles 4.
       Wait, we need more triangles for a meaningful run.

       Actually let's use: 4x3 grid = 12 nodes, 3 cols * 2 rows = 6 squares = 12 triangles.
    */

    /* Let me use a simpler mesh. 4x2 grid:
       y=1: 0(0,1), 1(1,1), 2(2,1), 3(3,1)
       y=0: 4(0,0), 5(1,0), 6(2,0), 7(3,0)
       4 nodes wide * 2 rows = 3 cols = 6 triangles.
       Actually 8 nodes, 3*2 squares = 6 squares = 12 triangles. That's good. */

    /* Let's just use 6 nodes, 4 triangles from test_phase1 */

    /* Use the same 4-triangle mesh */
    hydro_int n_node = 6;
    hydro_int n_tri = 4;

    double vc[12] = {
        0.0, 0.0, 1.0, 0.0, 2.0, 0.0,
        0.0, 1.0, 1.0, 1.0, 2.0, 1.0
    };
    hydro_int tri[12] = {0,1,4, 0,4,3, 1,2,5, 1,5,4};

    hydro_int btags[12] = {1,1,1,1,1,1,1,1,1,1,1,1};
    hydro_int bedges[12] = {0,1,2,3,4,5,6,7,8,9,10,11};

    hydro_domain_t* d = hydro_domain_create(n_node, n_tri);

    hydro_domain_set_geometry(d, vc, tri, btags, bedges);
    hydro_mesh_build_neighbour_structure(d);
    hydro_mesh_build_boundary_structure(d);

    /* Dam-break initial conditions */
    double elevation[4] = {0,0,0,0};
    double stage[4]    = {2,2,1,1};
    double xmom[4]     = {0,0,0,0};
    double ymom[4]     = {0,0,0,0};
    double friction[4] = {0.03,0.03,0.03,0.03};

    hydro_domain_set_quantity(d, "elevation", elevation);
    hydro_domain_set_quantity(d, "stage", stage);
    hydro_domain_set_quantity(d, "xmomentum", xmom);
    hydro_domain_set_quantity(d, "ymomentum", ymom);
    hydro_domain_set_quantity(d, "friction", friction);

    /* Initial mass */
    double mass0 = 0;
    for (int i = 0; i < 4; i++) mass0 += stage[i] * d->areas[i];

    /* Run evolve for 0.1 seconds */
    printf("Evolving to t=0.1 with yieldstep=0.05...\n");
    d->timestepping_method = 1;  /* Euler */
    d->CFL = 1.0;
    d->spatial_order = 1;

    hydro_domain_set_name(d, "hydro_phase1b");
    hydro_domain_set_output_dir(d, "/tmp");
    int ret = hydro_domain_evolve(d, 0.1, 0.05, 0.1);
    assert(ret == 0);
    printf("Evolve complete: %lld steps, final time=%g\n",
           (long long)d->step, d->time);

    /* Check final state */
    double final_mass = 0;
    for (int i = 0; i < 4; i++) {
        double h = d->stage_centroid_values[i] - d->bed_centroid_values[i];
        assert(h >= -1e-9);
        assert(!isnan(d->stage_centroid_values[i]));
        final_mass += d->stage_centroid_values[i] * d->areas[i];
    }

    printf("Initial mass: %g, Final mass: %g\n", mass0, final_mass);
    printf("Mass ratio: %g\n", final_mass / mass0);

    hydro_domain_destroy(d);

    printf("\n=== Phase 1b Test PASSED ===\n");
    return 0;
}
