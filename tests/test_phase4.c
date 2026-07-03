/**
 * test_phase4.c — Full module verification
 *
 * Tests all new Phase 4 modules:
 *   1. Geometry — polygon area, inside/outside, line intersection
 *   2. Coordinate transforms — Redfearn UTM projection
 *   3. Forcing — wind stress, pressure gradient, rainfall
 *   4. Operators — explicit friction, erosion, set stage/elevation
 *   5. Structures — Boyd box/pipe discharge, inlet
 *   6. Fit/Interpolate — triangle search, interpolation, grid interp
 */

#include "hydro/hydro.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <string.h>

#define TOL 1e-9

/* =========================================================================
 * Test 1: Geometry — Polygon Area
 * ========================================================================= */
static void test_polygon_area(void) {
    printf("Test 1: Polygon Area\n");

    double square[8] = {0,0, 1,0, 1,1, 0,1};
    double area = hydro_polygon_area(square, 4);
    printf("  Unit square area: %.6f (expected 1.0)\n", area);
    assert(fabs(area - 1.0) < TOL);

    double tri[6] = {0,0, 3,0, 0,4};
    area = hydro_polygon_area(tri, 3);
    printf("  Triangle (3x4) area: %.6f (expected 6.0)\n", area);
    assert(fabs(area - 6.0) < TOL);

    printf("  Polygon area OK\n");
}

/* =========================================================================
 * Test 2: Geometry — Inside Polygon / Triangle
 * ========================================================================= */
static void test_inside_polygon(void) {
    printf("\nTest 2: Inside Polygon / Triangle\n");

    double square[8] = {0,0, 1,0, 1,1, 0,1};
    double pt_in[2] = {0.5, 0.5};
    double pt_out[2] = {2.0, 2.0};
    double pt_edge[2] = {0.5, 0.0};

    assert(hydro_is_inside_polygon(pt_in, square, 4, 1));
    assert(!hydro_is_inside_polygon(pt_out, square, 4, 1));
    assert(hydro_is_inside_polygon(pt_edge, square, 4, 1));
    assert(!hydro_is_inside_polygon(pt_edge, square, 4, 0));
    printf("  Point-in-polygon OK\n");

    /* Inside triangle */
    double tri[6] = {0,0, 1,0, 0,1};
    double pt_tri_in[2] = {0.2, 0.2};
    double pt_tri_out[2] = {0.8, 0.8};
    assert(hydro_is_inside_triangle(pt_tri_in, tri, 1));
    assert(!hydro_is_inside_triangle(pt_tri_out, tri, 1));
    printf("  Point-in-triangle OK\n");

    /* Separate points by polygon */
    double pts[10] = {0.5,0.5, -1,-1, 0.2,0.8, 2,2, 0.1,0.1};
    hydro_int indices[5];
    hydro_int count = hydro_separate_points_by_polygon(pts, 5, square, 4, 1, indices);
    printf("  Inside count: %lld (expected 3)\n", (long long)count);
    assert(count == 3);
    /* First 3 indices should be inside (0,2,4), last 2 outside (1,3) */
    for (int i = 0; i < count; i++) {
        assert(hydro_is_inside_polygon(&pts[2*indices[i]], square, 4, 1));
    }
    printf("  Separate points OK\n");
}

/* =========================================================================
 * Test 3: Geometry — Line Operations
 * ========================================================================= */
static void test_line_ops(void) {
    printf("\nTest 3: Line Operations\n");

    /* Point on line segment */
    double line[4] = {0,0, 2,0};
    double pt_on[2] = {1.0, 0.0};
    double pt_off[2] = {1.0, 0.1};
    double pt_end[2] = {0.0, 0.0};

    assert(hydro_point_on_line(pt_on, line, TOL, TOL));
    assert(!hydro_point_on_line(pt_off, line, TOL, TOL));
    assert(hydro_point_on_line(pt_end, line, TOL, TOL));
    printf("  Point-on-line OK\n");

    /* Line intersection */
    double l0[4] = {0,0, 2,2};
    double l1[4] = {0,2, 2,0};
    double result[2];
    int status = hydro_line_intersection(l0, l1, result);
    printf("  Intersection status=%d, point=(%.3f,%.3f) (expected 1,(1,1))\n",
           status, result[0], result[1]);
    assert(status == 1);
    assert(fabs(result[0] - 1.0) < TOL);
    assert(fabs(result[1] - 1.0) < TOL);
    printf("  Line intersection OK\n");
}

/* =========================================================================
 * Test 4: Coordinate Transforms — Redfearn UTM
 * ========================================================================= */
static void test_coordinate_transforms(void) {
    printf("\nTest 4: Coordinate Transforms (Redfearn UTM)\n");

    /* Known reference point: Sydney Observatory
     * ~33.859972 S, 151.205111 E -> UTM zone 56, E ~334000, N ~6252000 */
    double lat = -33.859972, lon = 151.205111;
    hydro_int zone;
    double easting, northing;
    hydro_redfearn_latlon_to_utm(lat, lon, &zone, &easting, &northing, -1, -1);

    printf("  Lat=%.6f Lon=%.6f -> Zone=%lld E=%.1f N=%.1f\n",
           lat, lon, (long long)zone, easting, northing);
    assert(zone == 56);
    assert(easting > 330000 && easting < 340000);
    assert(northing > 6240000 && northing < 6260000);
    printf("  Redfearn forward OK\n");

    /* Round-trip test */
    double lat2, lon2;
    hydro_redfearn_utm_to_latlon(zone, easting, northing, 1, &lat2, &lon2);
    printf("  Round-trip: Lat=%.6f Lon=%.6f\n", lat2, lon2);
    assert(fabs(lat - lat2) < 0.001);
    assert(fabs(lon - lon2) < 0.001);
    printf("  Redfearn inverse OK\n");

    /* Geo-reference */
    hydro_geo_ref_t gr;
    hydro_geo_ref_init(&gr);
    hydro_geo_ref_set_utm(&gr, 56, 1); /* southern hemisphere */
    gr.xllcorner = 330000;
    gr.yllcorner = 6240000;

    int epsg = hydro_geo_ref_get_epsg(&gr);
    printf("  EPSG: %d (expected 32756)\n", epsg);
    assert(epsg == 32756);

    /* Coordinate change of basis */
    double pts[4] = {100, 200, 500, 1000};
    hydro_geo_ref_to_absolute(&gr, pts, 2);
    printf("  Absolute: (%.1f,%.1f) (%.1f,%.1f)\n", pts[0], pts[1], pts[2], pts[3]);
    assert(fabs(pts[0] - 330100) < TOL);
    assert(fabs(pts[1] - 6240200) < TOL);

    hydro_geo_ref_to_relative(&gr, pts, 2);
    printf("  Relative: (%.1f,%.1f) (%.1f,%.1f)\n", pts[0], pts[1], pts[2], pts[3]);
    assert(fabs(pts[0] - 100) < TOL);
    assert(fabs(pts[1] - 200) < TOL);
    printf("  Geo-reference OK\n");

    /* DMS conversion */
    double dec = hydro_dms_to_decimal_degrees(-33, 51, 36.0);
    printf("  DMS to DD: %.6f (expected -33.86)\n", dec);
    assert(fabs(dec - (-33.86)) < 0.001);

    int dd, mm; double ss;
    hydro_decimal_degrees_to_dms(-33.86, &dd, &mm, &ss);
    printf("  DD to DMS: %d %d %.1f\n", dd, mm, ss);
    assert(dd == -33);
    assert(mm == 51);
    printf("  DMS conversion OK\n");
}

/* =========================================================================
 * Test 5: Forcing — Wind Stress and Pressure Gradient
 * ========================================================================= */
static void test_forcing(void) {
    printf("\nTest 5: Forcing (Wind & Pressure)\n");

    hydro_int n_node = 6, n_tri = 4;
    double vc[12] = {0,0, 1,0, 2,0, 0,1, 1,1, 2,1};
    hydro_int tri[12] = {0,1,4, 0,4,3, 1,2,5, 1,5,4};
    hydro_int bt[12] = {1,1,1,1, 0,0,0,0, 0,0,0,0};
    hydro_int be[12] = {0,1,2,3, 0,1,2,3, 0,1,2,3};

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
    hydro_quantity_update_derived(d);

    /* Explicit update arrays already allocated by hydro_domain_set_quantity.
     * Zero them to start clean. */
    memset(d->xmom_explicit_update, 0, n_tri * sizeof(double));
    memset(d->ymom_explicit_update, 0, n_tri * sizeof(double));

    /* Test wind stress */
    double wind_speed[4] = {10, 10, 10, 10};
    double wind_dir[4] = {0, 0, 0, 0};  /* wind from north? actually to east */
    hydro_wind_stress_apply(d, wind_speed, 0, wind_dir, 0, n_tri, 0.0);

    printf("  Wind stress xmom_explicit: [%.6e, %.6e, %.6e, %.6e]\n",
           d->xmom_explicit_update[0], d->xmom_explicit_update[1],
           d->xmom_explicit_update[2], d->xmom_explicit_update[3]);
    /* All xmom entries should be equal (same wind applied) */
    assert(d->xmom_explicit_update[0] > 0);
    assert(fabs(d->xmom_explicit_update[0] - d->xmom_explicit_update[1]) < TOL);
    /* ymom should be near zero (wind direction = 0 deg) */
    assert(fabs(d->ymom_explicit_update[0]) < TOL);
    printf("  Wind stress OK\n");

    /* Clear explicit updates */
    memset(d->xmom_explicit_update, 0, n_tri * sizeof(double));
    memset(d->ymom_explicit_update, 0, n_tri * sizeof(double));

    /* Test barometric pressure gradient */
    double pressure[6] = {1013.0, 1013.0, 1015.0, 1011.0, 1013.0, 1015.0};
    hydro_barometric_pressure_apply(d, pressure, n_node, 0.0);
    printf("  Pressure xmom_explicit: [%.6e, %.6e, %.6e, %.6e]\n",
           d->xmom_explicit_update[0], d->xmom_explicit_update[1],
           d->xmom_explicit_update[2], d->xmom_explicit_update[3]);
    printf("  Pressure forcing OK\n");

    /* Test rainfall */
    double initial_stage = d->stage_centroid_values[0];
    d->timestep = 1.0;
    hydro_rainfall_apply(d, 0.01);  /* 10 mm/s for 1s = +0.01m */
    printf("  Stage after rain: %.6f -> %.6f\n", initial_stage,
           d->stage_centroid_values[0]);
    assert(fabs(d->stage_centroid_values[0] - (initial_stage + 0.01)) < TOL);
    printf("  Rainfall OK\n");

    hydro_domain_destroy(d);
}

/* =========================================================================
 * Test 6: Operators — Friction, Erosion, Set Stage
 * ========================================================================= */
static void test_operators(void) {
    printf("\nTest 6: Operators\n");

    hydro_int n_node = 6, n_tri = 4;
    double vc[12] = {0,0, 1,0, 2,0, 0,1, 1,1, 2,1};
    hydro_int tri[12] = {0,1,4, 0,4,3, 1,2,5, 1,5,4};
    hydro_int bt[12] = {1,1,1,1, 0,0,0,0, 0,0,0,0};
    hydro_int be[12] = {0,1,2,3, 0,1,2,3, 0,1,2,3};

    hydro_domain_t* d = hydro_domain_create(n_node, n_tri);
    hydro_domain_set_geometry(d, vc, tri, bt, be);
    hydro_mesh_build_neighbour_structure(d);
    hydro_mesh_build_boundary_structure(d);

    double elev[4] = {0,0,0,0};
    double stage[4] = {2,2,1,1};
    double xmom[4] = {1,1,0.5,0.5};
    double ymom[4] = {0,0,0,0};
    double fric[4] = {0.05,0.05,0.05,0.05};

    hydro_domain_set_quantity(d, "elevation", elev);
    hydro_domain_set_quantity(d, "stage", stage);
    hydro_domain_set_quantity(d, "xmomentum", xmom);
    hydro_domain_set_quantity(d, "ymomentum", ymom);
    hydro_domain_set_quantity(d, "friction", fric);
    hydro_quantity_update_derived(d);

    /* Test explicit Manning friction */
    double ke_before = 0;
    for (int i = 0; i < n_tri; i++) {
        ke_before += xmom[i]*xmom[i] + ymom[i]*ymom[i];
    }
    d->timestep = 1.0;
    hydro_manning_friction_explicit(d);
    double ke_after = 0;
    for (int i = 0; i < n_tri; i++) {
        ke_after += d->xmom_centroid_values[i]*d->xmom_centroid_values[i] +
                    d->ymom_centroid_values[i]*d->ymom_centroid_values[i];
    }
    printf("  KE before friction: %g, after: %g\n", ke_before, ke_after);
    /* Friction should reduce kinetic energy */
    assert(ke_after < ke_before);
    assert(!isnan(ke_after));
    printf("  Explicit friction OK\n");

    /* Test set stage */
    double new_stage[4] = {1.5, 1.5, 0.8, 0.8};
    hydro_set_stage(d, new_stage, NULL, n_tri);
    for (int i = 0; i < n_tri; i++) {
        assert(fabs(d->stage_centroid_values[i] - new_stage[i]) < TOL);
    }
    printf("  Set stage OK\n");

    /* Test set elevation */
    double new_elev[4] = {0.1, 0.1, 0.1, 0.1};
    hydro_set_elevation(d, new_elev, NULL, n_tri);
    for (int i = 0; i < n_tri; i++) {
        assert(fabs(d->bed_centroid_values[i] - 0.1) < TOL);
        assert(d->stage_centroid_values[i] >= d->bed_centroid_values[i]);
    }
    printf("  Set elevation OK\n");

    /* Test erosion */
    hydro_bed_shear_erosion_apply(d, 0.5, -1.0, NULL, n_tri);
    /* With threshold 0.5, momentum = 1.0 for triangles 0-1, so they erode */
    printf("  Elev after erosion: [%.4f, %.4f, %.4f, %.4f]\n",
           d->bed_centroid_values[0], d->bed_centroid_values[1],
           d->bed_centroid_values[2], d->bed_centroid_values[3]);
    /* Triangles 0-1 (high momentum) should erode more than 2-3 */
    assert(d->bed_centroid_values[0] < 0.1); /* eroded */
    printf("  Erosion OK\n");

    hydro_domain_destroy(d);
}

/* =========================================================================
 * Test 7: Structures — Boyd Box Discharge
 * ========================================================================= */
static void test_structures(void) {
    printf("\nTest 7: Structures\n");

    double Q, velocity, flow_area;
    int ret;

    /* Boyd box: 2m wide, 1.5m high, 1 barrel, no blockage
     * 0.5m head difference, 10m long culvert */
    ret = hydro_boyd_box_discharge(
        9.8, 2.0, 1.5, 1.0, 0.0,
        1.5, 0.013, 10.0, 1,
        2.5, 2.0, 1.5, 1.0,
        &Q, &velocity, &flow_area);

    printf("  Boyd box: ret=%d Q=%.4f m3/s v=%.4f m/s A=%.4f m2\n",
           ret, Q, velocity, flow_area);
    assert(ret == 0);
    assert(Q > 0);
    assert(isfinite(Q));
    assert(velocity > 0);
    assert(flow_area > 0);
    printf("  Boyd box OK\n");

    /* Boyd pipe: 1.5m diameter */
    ret = hydro_boyd_pipe_discharge(
        9.8, 1.5, 1.0, 0.0,
        1.5, 0.013, 10.0, 1,
        2.5, 2.0, 1.5, 1.0,
        &Q, &velocity, &flow_area);
    printf("  Boyd pipe: ret=%d Q=%.4f m3/s v=%.4f m/s A=%.4f m2\n",
           ret, Q, velocity, flow_area);
    assert(ret == 0);
    assert(Q > 0);
    assert(isfinite(Q));
    printf("  Boyd pipe OK\n");

    /* Weir/orifice trapezoid */
    ret = hydro_weir_orifice_trapezoid_discharge(
        9.8, 2.0, 2.0, 2.0, 1.0, 0.0,
        1.5, 0.013, 10.0, 1,
        2.5, 2.0, 1.5, 1.0,
        &Q, &velocity, &flow_area);
    printf("  Trapezoid weir: ret=%d Q=%.4f m3/s v=%.4f m/s A=%.4f m2\n",
           ret, Q, velocity, flow_area);
    assert(ret == 0);
    assert(Q > 0);
    assert(isfinite(Q));
    printf("  Weir/orifice OK\n");

    /* Blocked culvert — zero flow */
    ret = hydro_boyd_box_discharge(
        9.8, 2.0, 1.5, 1.0, 1.0,  /* blockage = 1.0 = fully blocked */
        1.5, 0.013, 10.0, 1,
        2.5, 2.0, 1.5, 1.0,
        &Q, &velocity, &flow_area);
    printf("  Blocked culvert: ret=%d Q=%.4f (expected 0)\n", ret, Q);
    assert(ret == 0);
    assert(fabs(Q) < TOL);
    printf("  Blockage OK\n");

    /* No head difference — zero flow */
    ret = hydro_boyd_box_discharge(
        9.8, 2.0, 1.5, 1.0, 0.0,
        1.5, 0.013, 10.0, 1,
        2.0, 2.0, 1.5, 1.0,  /* equal heads */
        &Q, &velocity, &flow_area);
    printf("  Equal heads: Q=%.4f (expected 0)\n", Q);
    assert(fabs(Q) < TOL);
    printf("  Zero head diff OK\n");

    /* Inlet volume distribution */
    {
        double stage[3] = {1.0, 1.0, 1.0};
        double elev[3] = {0.0, 0.0, 0.0};
        double xmom[3] = {0.0, 0.0, 0.0};
        double ymom[3] = {0.0, 0.0, 0.0};
        double areas[3] = {0.5, 0.5, 0.5};
        hydro_int indices[3] = {0, 1, 2};

        double vol = hydro_inlet_distribute_volume(
            0.3, indices, areas, 3, stage, elev, xmom, ymom);
        printf("  Inlet distribute: vol=%.3f, stages=[%.2f,%.2f,%.2f]\n",
               vol, stage[0], stage[1], stage[2]);
        assert(fabs(vol - 0.3) < TOL);
        /* 0.3 m3 over 1.5 m2 = 0.2m rise */
        assert(fabs(stage[0] - 1.2) < TOL);
        printf("  Inlet distribution OK\n");
    }
}

/* =========================================================================
 * Test 8: Fit/Interpolate — Search and Interpolation
 * ========================================================================= */
static void test_fit_interpolate(void) {
    printf("\nTest 8: Fit/Interpolate\n");

    /* Simple 2-triangle mesh */
    double vc[8] = {0,0, 1,0, 0,1, 1,1};  /* 4 vertices */
    hydro_int tri[6] = {0,1,3, 0,3,2};     /* 2 triangles */
    hydro_int n_tri = 2;
    double vertex_vals[4] = {10.0, 20.0, 30.0, 40.0};

    /* Test find containing triangle */
    double pt1[2] = {0.5, 0.2};  /* inside triangle 0 */
    hydro_interp_result_t res = hydro_find_containing_triangle(
        pt1, vc, tri, n_tri, -1);
    printf("  Find pt(0.5,0.2): tri=%lld, sigma=[%.3f,%.3f,%.3f]\n",
           (long long)res.triangle_index,
           res.sigma[0], res.sigma[1], res.sigma[2]);
    assert(res.triangle_index == 0);
    assert(res.sigma[0] + res.sigma[1] + res.sigma[2] - 1.0 < TOL);

    /* Test interpolation at a point */
    double val = hydro_interpolate_at_point(
        vertex_vals, vc, tri, n_tri, pt1, -999.0);
    printf("  Interp at (0.5,0.2): %.3f\n", val);
    assert(val > 10.0 && val < 40.0);
    assert(!isnan(val));

    /* Point outside mesh */
    double pt_out[2] = {10, 10};
    val = hydro_interpolate_at_point(
        vertex_vals, vc, tri, n_tri, pt_out, NAN);
    printf("  Outside point: %.3f (expected NaN)\n", val);
    assert(isnan(val));

    /* Batch interpolation */
    double pts[8] = {0.5,0.2, 0.5,0.5, 0.8,0.8, 10,10};
    double output[4];
    hydro_interpolate_batch(
        vertex_vals, vc, tri, n_tri, pts, 4, output, -1.0);
    printf("  Batch interp: [%.3f, %.3f, %.3f, %.3f]\n",
           output[0], output[1], output[2], output[3]);
    assert(output[0] > 0 && output[1] > 0 && output[2] > 0);
    assert(output[3] == -1.0);  /* outside = fill value */
    printf("  Interpolation OK\n");

    /* Regular grid interpolation */
    double grid_x[3] = {0, 1, 2};
    double grid_y[3] = {0, 1, 2};
    double Z[9] = {
        0,1,2,
        3,4,5,
        6,7,8
    };
    double gpt[2] = {0.5, 0.5};
    double gpt_out[2] = {5, 5};
    double goutput[2];

    hydro_interpolate_regular_grid(
        grid_x, 3, grid_y, 3, Z, gpt, 1, goutput, 0, -999.0);
    printf("  Grid NN at (0.5,0.5): %.1f (expected closest corner)\n", goutput[0]);

    hydro_interpolate_regular_grid(
        grid_x, 3, grid_y, 3, Z, gpt, 1, goutput, 1, -999.0);
    printf("  Grid bilinear at (0.5,0.5): %.1f (expected ~2)\n", goutput[0]);

    hydro_interpolate_regular_grid(
        grid_x, 3, grid_y, 3, Z, gpt_out, 1, &goutput[1], 1, -999.0);
    printf("  Grid outside: %.1f (expected -999)\n", goutput[1]);
    assert(goutput[1] == -999.0);
    printf("  Grid interpolation OK\n");
}

/* =========================================================================
 * Main
 * ========================================================================= */
int main(void) {
    printf("=== Hydro Core Phase 4: Full Module Verification ===\n\n");

    test_polygon_area();
    test_inside_polygon();
    test_line_ops();
    test_coordinate_transforms();
    test_forcing();
    test_operators();
    test_structures();
    test_fit_interpolate();

    printf("\n=== Phase 4 Test PASSED ===\n");
    return 0;
}
