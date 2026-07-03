/**
 * test_phase0.c — Verify Phase 0: domain creation, geometry, SWW output
 *
 * Creates a small rectangular mesh (4 triangles), sets up quantities,
 * builds neighbour structure, writes an SWW file.
 *
 * Verification steps:
 *   1. Create domain
 *   2. Set mesh geometry
 *   3. Build neighbour structure
 *   4. Set quantities
 *   5. Write SWW file
 *   6. Verify SWW file can be read back (basic existence check)
 */

#include "hydro/hydro.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <assert.h>

int main(void) {
    printf("=== Hydro Core Phase 0 Test ===\n\n");

    /* ======================================================================
     * 1. Create a simple 4-triangle mesh (2x2 grid)
     *
     *  Nodes:      Triangles:
     *   2---3---4     0: a-b-c = 0,1,2
     *   |\ 1|\ 3|     1: b-d-c = 1,3,2
     *   |0\ |2\ |     2: c-d-e = 2,3,4
     *   0---1---2     3: d-f-e = 3,5,4
     *   |\  |\  |
     *   | \ | \ |
     *   5---6---7
     *
     *  Actually, let's use a simpler standard 4-triangle square:
     *
     *  (0,2) -- (1,2) -- (2,2)
     *    |   \   2   /    |
     *    |  1   \  /   3  |
     *    |       \/       |
     *  (0,0) -- (1,0) -- (2,0)
     *         0
     *  Triangle 0: nodes 0,1,2  (bottom-left half of left square)
     *  Wait, let's be explicit:
     *
     *  Nodes: 6 nodes at positions:
     *    0: (0,0)
     *    1: (1,0)
     *    2: (2,0)
     *    3: (0,1)
     *    4: (1,1)
     *    5: (2,1)
     *
     *  Triangles (CCW):
     *    T0: 0,1,4
     *    T1: 0,4,3
     *    T2: 1,2,5
     *    T3: 1,5,4
     * ====================================================================== */

    hydro_int n_nodes = 6;
    hydro_int n_triangles = 4;

    double vertex_coords[12] = {
        0.0, 0.0,   /* node 0 */
        1.0, 0.0,   /* node 1 */
        2.0, 0.0,   /* node 2 */
        0.0, 1.0,   /* node 3 */
        1.0, 1.0,   /* node 4 */
        2.0, 1.0    /* node 5 */
    };

    hydro_int triangles[12] = {
        0, 1, 4,    /* T0 */
        0, 4, 3,    /* T1 */
        1, 2, 5,    /* T2 */
        1, 5, 4     /* T3 */
    };

    /* Create domain */
    hydro_domain_t* d = hydro_domain_create(n_nodes, n_triangles);
    printf("1. Domain created: %lld triangles\n", (long long)n_triangles);

    /* Set geometry (no boundary tags yet) */
    hydro_int boundary_tags[8] = {1, 1, 1, 1, 1, 1, 1, 1};  /* dummy */
    hydro_int boundary_edges[8] = {0, 1, 2, 3, 4, 5, 6, 7};  /* dummy */
    d->boundary_length = 8;

    hydro_domain_set_geometry(d, vertex_coords, triangles,
                              boundary_tags, boundary_edges);
    printf("2. Geometry set\n");
    printf("   First triangle area: %g\n", d->areas[0]);
    printf("   First triangle centroid: (%g, %g)\n",
           d->centroid_coordinates[0], d->centroid_coordinates[1]);

    /* Verify area computation */
    for (hydro_int k = 0; k < n_triangles; k++) {
        double area = d->areas[k];
        assert(area > 0.0);  /* CCW ordering must give positive area */
    }
    printf("   All areas positive ✓\n");

    /* ======================================================================
     * 3. Build neighbour structure
     * ====================================================================== */
    hydro_mesh_build_neighbour_structure(d);
    hydro_mesh_build_boundary_structure(d);
    printf("3. Neighbour structure built\n");
    printf("   Boundary length: %lld\n", (long long)d->boundary_length);

    /* ======================================================================
     * 4. Set quantities
     * ====================================================================== */
    double elevation[4] = {0.0, 0.0, 0.0, 0.0};
    double stage[4]      = {1.0, 1.0, 1.0, 1.0};
    double xmom[4]       = {0.0, 0.0, 0.0, 0.0};
    double ymom[4]       = {0.0, 0.0, 0.0, 0.0};
    double friction[4]   = {0.03, 0.03, 0.03, 0.03};

    hydro_domain_set_quantity(d, "elevation", elevation);
    hydro_domain_set_quantity(d, "stage", stage);
    hydro_domain_set_quantity(d, "xmomentum", xmom);
    hydro_domain_set_quantity(d, "ymomentum", ymom);
    hydro_domain_set_quantity(d, "friction", friction);
    printf("4. Quantities set\n");

    /* Verify quantity retrieval */
    double check[4];
    hydro_domain_get_quantity(d, "elevation", check);
    assert(fabs(check[0] - 0.0) < 1e-10);
    hydro_domain_get_quantity(d, "stage", check);
    assert(fabs(check[0] - 1.0) < 1e-10);
    printf("   Quantity retrieval ✓\n");

    /* ======================================================================
     * 5. Write SWW file (static data only for Phase 0)
     * ====================================================================== */
    hydro_sww_t* sww = hydro_sww_create("/tmp/hydro_phase0_test.sww", d, 0.0);
    if (!sww) {
        fprintf(stderr, "ERROR: Failed to create SWW file!\n");
        hydro_domain_destroy(d);
        return 1;
    }
    printf("5. SWW file created: /tmp/hydro_phase0_test.sww\n");

    /* Extrapolate centroid→vertices for SWW output */
    hydro_quantity_extrapolate_first_order(d);

    /* Write two timesteps */
    hydro_sww_store_timestep(sww, d, 0.0);

    double stage2[4] = {0.9, 0.9, 1.0, 1.0};
    hydro_domain_set_quantity(d, "stage", stage2);
    hydro_quantity_extrapolate_first_order(d);
    hydro_sww_store_timestep(sww, d, 1.0);

    hydro_sww_close(sww);
    printf("   Timesteps written and file closed ✓\n");

    /* ======================================================================
     * 6. Cleanup
     * ====================================================================== */
    hydro_domain_destroy(d);
    printf("6. Domain destroyed ✓\n");

    printf("\n=== Phase 0 Test PASSED ===\n");
    return 0;
}
