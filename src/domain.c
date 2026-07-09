/**
 * domain.c — Domain lifecycle and core operations
 *
 * Adapted from:
 *   anuga/shallow_water/sw_domain.h (struct domain)
 *   anuga/abstract_2d_finite_volumes/general_mesh.py (mesh geometry)
 *   anuga/shallow_water/sw_domain_openmp.c (computational kernels)
 */

#include "hydro/domain.h"
#include "hydro/config.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

/* ==========================================================================
 * Allocation helpers
 * ========================================================================== */

static void* hydro_alloc(size_t size)
{
    void* p = calloc(1, size);
    if (!p)
    {
        fprintf(stderr, "hydro: memory allocation failed (%zu bytes)\n", size);
        abort();
    }
    return p;
}

static double* alloc_double(hydro_int n)
{
    return (double*)hydro_alloc((size_t)n * sizeof(double));
}

static hydro_int* alloc_int(hydro_int n)
{
    return (hydro_int*)hydro_alloc((size_t)n * sizeof(hydro_int));
}

/* ==========================================================================
 * Domain lifecycle
 * ========================================================================== */

hydro_domain_t* hydro_domain_create(hydro_int n_nodes, hydro_int n_triangles)
{
    hydro_domain_t* d = (hydro_domain_t*)hydro_alloc(sizeof(hydro_domain_t));
    hydro_int n_edges = 3 * n_triangles;

    d->number_of_elements = n_triangles;
    d->number_of_nodes = n_nodes;
    d->number_of_edges = n_edges;

    /* ---- Scalar defaults (from config.h) ---- */
    d->epsilon = HYDRO_EPSILON;
    d->H0 = HYDRO_MINIMUM_ALLOWED_HEIGHT;
    d->g = HYDRO_G;
    d->evolve_max_timestep = HYDRO_MAX_TIMESTEP;
    d->evolve_min_timestep = HYDRO_MIN_TIMESTEP;
    d->minimum_allowed_height = HYDRO_MINIMUM_ALLOWED_HEIGHT;
    d->minimum_storable_height = HYDRO_MINIMUM_STORABLE_HEIGHT;
    d->maximum_allowed_speed = HYDRO_MAXIMUM_ALLOWED_SPEED;

    d->beta_w = HYDRO_BETA_W;
    d->beta_w_dry = HYDRO_BETA_W;
    d->beta_uh = HYDRO_BETA_W;
    d->beta_uh_dry = HYDRO_BETA_W;
    d->beta_vh = HYDRO_BETA_W;
    d->beta_vh_dry = HYDRO_BETA_W;

    d->CFL = HYDRO_CFL;

    d->optimise_dry_cells = HYDRO_OPTIMISE_DRY_CELLS;
    d->extrapolate_velocity_second_order = HYDRO_EXTRAPOLATE_VELOCITY_SECOND_ORDER;
    d->low_froude = HYDRO_LOW_FROUDE;
    d->timestep_fluxcalls = 0;
    d->max_flux_update_frequency = 1;
    d->number_of_riverwall_edges = 0;
    d->ncol_riverwall_hydraulic_properties = 0;

    d->spatial_order = HYDRO_DEFAULT_ORDER;
    d->timestepping_method = HYDRO_DEFAULT_TIMESTEPPING_METHOD;
    d->low_froude_mode = HYDRO_LOW_FROUDE;

    /* ---- Time state ---- */
    d->time = 0.0;
    d->relative_time = 0.0;
    d->starttime = 0.0;
    d->timestep = 0.0;
    d->flux_timestep = 0.0;
    d->yieldstep_counter = 0;
    d->output_frequency = 1; /* write SWW every yieldstep by default */
    d->step = 0;

    /* ---- Geo-reference defaults ---- */
    d->xllcorner = 0.0;
    d->yllcorner = 0.0;
    d->zone = -1;
    strcpy(d->datum, "wgs84");
    strcpy(d->projection, "UTM");
    strcpy(d->units, "m");

    /* ---- SWW output defaults ---- */
    d->name[0] = '\0'; /* empty = no SWW output */
    strcpy(d->output_dir, ".");

    /* ---- Mesh geometry (allocated now, filled by set_geometry) ---- */
    d->vertex_coordinates = alloc_double(n_edges * 2);
    d->edge_coordinates = alloc_double(n_edges * 2);
    d->centroid_coordinates = alloc_double(n_triangles * 2);
    d->normals = alloc_double(n_edges * 2);
    d->edgelengths = alloc_double(n_edges);
    d->radii = alloc_double(n_triangles);
    d->areas = alloc_double(n_triangles);

    /* ---- Mesh topology ---- */
    d->triangles = alloc_int(n_edges); /* M*3 */
    d->neighbours = alloc_int(n_edges);
    d->neighbour_edges = alloc_int(n_edges);
    d->surrogate_neighbours = alloc_int(n_edges);
    d->edge_flux_type = alloc_int(n_edges);
    d->edge_river_wall_counter = alloc_int(n_edges);
    d->tri_full_flag = alloc_int(n_triangles);
    d->already_computed_flux = alloc_int(n_edges);
    d->number_of_boundaries = alloc_int(n_triangles);

    for (hydro_int i = 0; i < n_triangles; i++)
    {
        d->tri_full_flag[i] = 1; /* all cells are "full" in serial */
    }

    /* ---- Work arrays ---- */
    d->max_speed = alloc_double(n_triangles);
    d->edge_timestep = alloc_double(n_edges);
    d->edge_qr_stage = alloc_double(n_edges);
    d->edge_qr_xmom = alloc_double(n_edges);
    d->edge_qr_ymom = alloc_double(n_edges);
    d->edge_zr = alloc_double(n_edges);
    d->edge_h_left = alloc_double(n_edges);
    d->edge_hre = alloc_double(n_edges);
    d->edge_h_right = alloc_double(n_edges);
    d->edge_z_half = alloc_double(n_edges);
    d->x_centroid_work = alloc_double(n_triangles);
    d->y_centroid_work = alloc_double(n_triangles);
    d->boundary_flux_sum = alloc_double(3); /* max_time_substeps=3 */

    d->flux_update_frequency = alloc_int(n_triangles);
    d->update_next_flux = alloc_int(n_triangles);
    d->update_extrapolation = alloc_int(n_triangles);
    d->allow_timestep_increase = alloc_int(n_triangles);

    /* ---- Boundary metadata (sized during set_geometry) ---- */
    d->boundary_length = 0;
    d->boundary_tags = NULL;
    d->boundary_edges = NULL;
    d->boundary_tag_map = NULL;

    /* Initialize per-tag BC configuration: default = reflective (type=1) */
    for (int i = 0; i < HYDRO_MAX_BOUNDARY_TAGS; i++)
    {
        d->boundary_bc_type_tag[i] = 1; /* HYDRO_BC_REFLECTIVE */
        d->boundary_stage_tag[i] = 0.0;
        d->boundary_xmom_tag[i] = 0.0;
        d->boundary_ymom_tag[i] = 0.0;
        d->boundary_time_series[i].times = NULL;
        d->boundary_time_series[i].q_values = NULL;
        d->boundary_time_series[i].n_points = 0;
        d->boundary_time_series[i].default_stage = 0.1;
        d->boundary_time_series[i].total_width = 0.0;
        d->boundary_time_series[i].mean_bed = 0.0;
    }

    d->geo_structure_indices = NULL;
    d->geo_structure_values = NULL;

    /* ---- Boundary values (sized during set_geometry) ---- */
    d->stage_boundary_values = NULL;
    d->xmom_boundary_values = NULL;
    d->ymom_boundary_values = NULL;
    d->bed_boundary_values = NULL;
    d->height_boundary_values = NULL;
    d->xvelocity_boundary_values = NULL;
    d->yvelocity_boundary_values = NULL;

    /* ---- Quantity arrays (allocated on set_quantity) ---- */
    d->stage_centroid_values = NULL;
    d->xmom_centroid_values = NULL;
    d->ymom_centroid_values = NULL;
    d->bed_centroid_values = NULL;
    d->height_centroid_values = NULL;
    d->friction_centroid_values = NULL;
    d->xvelocity_centroid_values = NULL;
    d->yvelocity_centroid_values = NULL;

    d->stage_edge_values = NULL;
    d->xmom_edge_values = NULL;
    d->ymom_edge_values = NULL;
    d->bed_edge_values = NULL;
    d->height_edge_values = NULL;
    d->xvelocity_edge_values = NULL;
    d->yvelocity_edge_values = NULL;

    d->stage_vertex_values = NULL;
    d->xmom_vertex_values = NULL;
    d->ymom_vertex_values = NULL;
    d->bed_vertex_values = NULL;
    d->height_vertex_values = NULL;

    d->stage_explicit_update = NULL;
    d->xmom_explicit_update = NULL;
    d->ymom_explicit_update = NULL;
    d->stage_semi_implicit_update = NULL;
    d->xmom_semi_implicit_update = NULL;
    d->ymom_semi_implicit_update = NULL;

    d->stage_backup_values = NULL;
    d->xmom_backup_values = NULL;
    d->ymom_backup_values = NULL;

    d->riverwall_elevation = NULL;
    d->riverwall_rowIndex = NULL;
    d->riverwall_hydraulic_properties = NULL;

    return d;
}

void hydro_domain_destroy(hydro_domain_t* d)
{
    if (!d) return;

    free(d->vertex_coordinates);
    free(d->edge_coordinates);
    free(d->centroid_coordinates);
    free(d->normals);
    free(d->edgelengths);
    free(d->radii);
    free(d->areas);

    free(d->triangles);
    free(d->neighbours);
    free(d->neighbour_edges);
    free(d->surrogate_neighbours);
    free(d->edge_flux_type);
    free(d->edge_river_wall_counter);
    free(d->tri_full_flag);
    free(d->already_computed_flux);
    free(d->number_of_boundaries);

    free(d->max_speed);
    free(d->edge_timestep);
    free(d->edge_qr_stage);
    free(d->edge_qr_xmom);
    free(d->edge_qr_ymom);
    free(d->edge_zr);
    free(d->edge_h_left);
    free(d->edge_hre);
    free(d->edge_h_right);
    free(d->edge_z_half);
    free(d->x_centroid_work);
    free(d->y_centroid_work);
    free(d->boundary_flux_sum);

    free(d->flux_update_frequency);
    free(d->update_next_flux);
    free(d->update_extrapolation);
    free(d->allow_timestep_increase);

    /* Free per-tag time-series data */
    for (int i = 0; i < HYDRO_MAX_BOUNDARY_TAGS; i++)
    {
        free(d->boundary_time_series[i].times);
        free(d->boundary_time_series[i].q_values);
    }

    free(d->boundary_tags);
    free(d->boundary_edges);
    free(d->boundary_tag_map);

    free(d->stage_boundary_values);
    free(d->xmom_boundary_values);
    free(d->ymom_boundary_values);
    free(d->bed_boundary_values);
    free(d->height_boundary_values);
    free(d->xvelocity_boundary_values);
    free(d->yvelocity_boundary_values);

    free(d->geo_structure_indices);
    free(d->geo_structure_values);

    free(d->stage_centroid_values);
    free(d->xmom_centroid_values);
    free(d->ymom_centroid_values);
    free(d->bed_centroid_values);
    free(d->height_centroid_values);
    free(d->friction_centroid_values);
    free(d->xvelocity_centroid_values);
    free(d->yvelocity_centroid_values);

    free(d->stage_edge_values);
    free(d->xmom_edge_values);
    free(d->ymom_edge_values);
    free(d->bed_edge_values);
    free(d->height_edge_values);
    free(d->xvelocity_edge_values);
    free(d->yvelocity_edge_values);

    free(d->stage_vertex_values);
    free(d->xmom_vertex_values);
    free(d->ymom_vertex_values);
    free(d->bed_vertex_values);
    free(d->height_vertex_values);

    free(d->stage_explicit_update);
    free(d->xmom_explicit_update);
    free(d->ymom_explicit_update);
    free(d->stage_semi_implicit_update);
    free(d->xmom_semi_implicit_update);
    free(d->ymom_semi_implicit_update);

    free(d->stage_backup_values);
    free(d->xmom_backup_values);
    free(d->ymom_backup_values);

    free(d->riverwall_elevation);
    free(d->riverwall_rowIndex);
    free(d->riverwall_hydraulic_properties);

    free(d);
}

/* ==========================================================================
 * Mesh Geometry Computation
 *
 * Ported from anuga/abstract_2d_finite_volumes/general_mesh.py
 * ========================================================================== */

void hydro_domain_set_geometry(
    hydro_domain_t* d,
    const double* vertex_coords,
    const hydro_int* triangles,
    const hydro_int* boundary_tags,
    const hydro_int* boundary_edges_in)
{
    hydro_int N = d->number_of_elements;
    hydro_int n_edges = d->number_of_edges;
    hydro_int i, k3;

    (void)boundary_tags; /* stored via hydro_domain_set_boundary_tag_map */
    (void)boundary_edges_in; /* stored via hydro_domain_set_boundary_tag_map */

    /* Copy triangle connectivity */
    for (i = 0; i < n_edges; i++)
    {
        d->triangles[i] = triangles[i];
    }

    /* ---- Build vertex_coordinates (flattened per-triangle [x0,y0,x1,y1,x2,y2]) ---- */
    for (i = 0; i < N; i++)
    {
        k3 = 3 * i;
        hydro_int n0 = triangles[k3];
        hydro_int n1 = triangles[k3 + 1];
        hydro_int n2 = triangles[k3 + 2];

        d->vertex_coordinates[2 * k3] = vertex_coords[2 * n0];
        d->vertex_coordinates[2 * k3 + 1] = vertex_coords[2 * n0 + 1];

        d->vertex_coordinates[2 * k3 + 2] = vertex_coords[2 * n1];
        d->vertex_coordinates[2 * k3 + 3] = vertex_coords[2 * n1 + 1];

        d->vertex_coordinates[2 * k3 + 4] = vertex_coords[2 * n2];
        d->vertex_coordinates[2 * k3 + 5] = vertex_coords[2 * n2 + 1];
    }

    /* ---- Compute areas, centroids, normals, edgelengths, radii ---- */
    for (i = 0; i < N; i++)
    {
        k3 = 3 * i;

        double x0 = d->vertex_coordinates[2 * k3];
        double y0 = d->vertex_coordinates[2 * k3 + 1];
        double x1 = d->vertex_coordinates[2 * k3 + 2];
        double y1 = d->vertex_coordinates[2 * k3 + 3];
        double x2 = d->vertex_coordinates[2 * k3 + 4];
        double y2 = d->vertex_coordinates[2 * k3 + 5];

        /* Area (CCW ordering gives positive area) */
        double area = -((x1 * y0 - x0 * y1) + (x2 * y1 - x1 * y2)
            + (x0 * y2 - x2 * y0)) / 2.0;
        d->areas[i] = area;

        /* Centroid */
        d->centroid_coordinates[2 * i] = (x0 + x1 + x2) / 3.0;
        d->centroid_coordinates[2 * i + 1] = (y0 + y1 + y2) / 3.0;

        /* Edge 0 (opposite vertex 0): from v1 to v2 */
        double xn0 = x2 - x1;
        double yn0 = y2 - y1;
        double l0 = sqrt(xn0 * xn0 + yn0 * yn0);
        xn0 /= l0;
        yn0 /= l0;
        d->normals[2 * k3] = yn0;
        d->normals[2 * k3 + 1] = -xn0;
        d->edgelengths[k3] = l0;

        /* Edge midpoint 0 */
        d->edge_coordinates[2 * k3] = (x1 + x2) / 2.0;
        d->edge_coordinates[2 * k3 + 1] = (y1 + y2) / 2.0;

        /* Edge 1 (opposite vertex 1): from v2 to v0 */
        double xn1 = x0 - x2;
        double yn1 = y0 - y2;
        double l1 = sqrt(xn1 * xn1 + yn1 * yn1);
        xn1 /= l1;
        yn1 /= l1;
        d->normals[2 * k3 + 2] = yn1;
        d->normals[2 * k3 + 3] = -xn1;
        d->edgelengths[k3 + 1] = l1;

        /* Edge midpoint 1 */
        d->edge_coordinates[2 * k3 + 2] = (x2 + x0) / 2.0;
        d->edge_coordinates[2 * k3 + 3] = (y2 + y0) / 2.0;

        /* Edge 2 (opposite vertex 2): from v0 to v1 */
        double xn2 = x1 - x0;
        double yn2 = y1 - y0;
        double l2 = sqrt(xn2 * xn2 + yn2 * yn2);
        xn2 /= l2;
        yn2 /= l2;
        d->normals[2 * k3 + 4] = yn2;
        d->normals[2 * k3 + 5] = -xn2;
        d->edgelengths[k3 + 2] = l2;

        /* Edge midpoint 2 */
        d->edge_coordinates[2 * k3 + 4] = (x0 + x1) / 2.0;
        d->edge_coordinates[2 * k3 + 5] = (y0 + y1) / 2.0;

        /* Radii: distance from centroid to closest edge midpoint */
        double cx = d->centroid_coordinates[2 * i];
        double cy = d->centroid_coordinates[2 * i + 1];

        double xm0 = (x1 + x2) / 2.0;
        double ym0 = (y1 + y2) / 2.0;
        double xm1 = (x2 + x0) / 2.0;
        double ym1 = (y2 + y0) / 2.0;
        double xm2 = (x0 + x1) / 2.0;
        double ym2 = (y0 + y1) / 2.0;

        double d0 = sqrt((cx - xm0) * (cx - xm0) + (cy - ym0) * (cy - ym0));
        double d1 = sqrt((cx - xm1) * (cx - xm1) + (cy - ym1) * (cy - ym1));
        double d2 = sqrt((cx - xm2) * (cx - xm2) + (cy - ym2) * (cy - ym2));

        double min_d = d0;
        if (d1 < min_d) min_d = d1;
        if (d2 < min_d) min_d = d2;
        d->radii[i] = min_d;
    }
}

/* ==========================================================================
 * Boundary tag map — stores user-provided per-edge boundary tags
 * ========================================================================== */

void hydro_domain_set_boundary_tag_map(
    hydro_domain_t* d,
    const hydro_int* boundary_edges_in,
    const hydro_int* boundary_tags_in,
    hydro_int count)
{
    hydro_int n_edges = d->number_of_edges;
    if (!d->boundary_tag_map)
    {
        d->boundary_tag_map = (hydro_int*)calloc((size_t)n_edges, sizeof(hydro_int));
    }
    for (hydro_int i = 0; i < count; i++)
    {
        hydro_int be = boundary_edges_in[i];
        if (be >= 0 && be < n_edges)
        {
            d->boundary_tag_map[be] = boundary_tags_in[i];
        }
    }
}

/* ==========================================================================
 * Quantity management
 * ========================================================================== */

static double** hydro_domain_get_centroid_ptr(hydro_domain_t* d, const char* name)
{
    if (strcmp(name, "stage") == 0) return &d->stage_centroid_values;
    if (strcmp(name, "xmomentum") == 0) return &d->xmom_centroid_values;
    if (strcmp(name, "ymomentum") == 0) return &d->ymom_centroid_values;
    if (strcmp(name, "elevation") == 0) return &d->bed_centroid_values;
    if (strcmp(name, "height") == 0) return &d->height_centroid_values;
    if (strcmp(name, "friction") == 0) return &d->friction_centroid_values;
    if (strcmp(name, "xvelocity") == 0) return &d->xvelocity_centroid_values;
    if (strcmp(name, "yvelocity") == 0) return &d->yvelocity_centroid_values;
    return NULL;
}

void hydro_domain_set_quantity(
    hydro_domain_t* d, const char* name, const double* values)
{
    hydro_int N = d->number_of_elements;
    double** dst_ptr = hydro_domain_get_centroid_ptr(d, name);

    if (!dst_ptr)
    {
        fprintf(stderr, "hydro: unknown quantity '%s'\n", name);
        return;
    }

    /* Allocate if not already */
    if (*dst_ptr == NULL)
    {
        *dst_ptr = alloc_double(N);
    }

    /* Also allocate edge and vertex arrays for conserved quantities */
    if (strcmp(name, "stage") == 0 || strcmp(name, "xmomentum") == 0
        || strcmp(name, "ymomentum") == 0)
    {
        hydro_int n_edges = d->number_of_edges;
        const char* names[] = {"stage", "xmomentum", "ymomentum"};
        for (int qi = 0; qi < 3; qi++)
        {
            if (strcmp(name, names[qi]) == 0)
            {
                /* Allocate if needed */
                if (qi == 0)
                {
                    if (!d->stage_edge_values) d->stage_edge_values = alloc_double(n_edges);
                    if (!d->stage_vertex_values) d->stage_vertex_values = alloc_double(n_edges);
                    if (!d->stage_explicit_update) d->stage_explicit_update = alloc_double(N);
                    if (!d->stage_semi_implicit_update) d->stage_semi_implicit_update = alloc_double(N);
                    if (!d->stage_backup_values) d->stage_backup_values = alloc_double(N);
                }
                else if (qi == 1)
                {
                    if (!d->xmom_edge_values) d->xmom_edge_values = alloc_double(n_edges);
                    if (!d->xmom_vertex_values) d->xmom_vertex_values = alloc_double(n_edges);
                    if (!d->xmom_explicit_update) d->xmom_explicit_update = alloc_double(N);
                    if (!d->xmom_semi_implicit_update) d->xmom_semi_implicit_update = alloc_double(N);
                    if (!d->xmom_backup_values) d->xmom_backup_values = alloc_double(N);
                }
                else
                {
                    if (!d->ymom_edge_values) d->ymom_edge_values = alloc_double(n_edges);
                    if (!d->ymom_vertex_values) d->ymom_vertex_values = alloc_double(n_edges);
                    if (!d->ymom_explicit_update) d->ymom_explicit_update = alloc_double(N);
                    if (!d->ymom_semi_implicit_update) d->ymom_semi_implicit_update = alloc_double(N);
                    if (!d->ymom_backup_values) d->ymom_backup_values = alloc_double(N);
                }
            }
        }
    }

    /* Also allocate edge/vertex arrays for auxiliary quantities */
    if (strcmp(name, "elevation") == 0)
    {
        hydro_int n_edges = d->number_of_edges;
        if (!d->bed_edge_values) d->bed_edge_values = alloc_double(n_edges);
        if (!d->bed_vertex_values) d->bed_vertex_values = alloc_double(n_edges);
    }
    if (strcmp(name, "height") == 0)
    {
        hydro_int n_edges = d->number_of_edges;
        if (!d->height_edge_values) d->height_edge_values = alloc_double(n_edges);
        if (!d->height_vertex_values) d->height_vertex_values = alloc_double(n_edges);
    }

    /* Copy values */
    for (hydro_int i = 0; i < N; i++)
    {
        (*dst_ptr)[i] = values[i];
    }
}

void hydro_domain_set_parameter(
    hydro_domain_t* d, const char* name, double value)
{
    if (strcmp(name, "g") == 0) d->g = value;
    else if (strcmp(name, "CFL") == 0) d->CFL = value;
    else if (strcmp(name, "evolve_max_timestep") == 0) d->evolve_max_timestep = value;
    else if (strcmp(name, "evolve_min_timestep") == 0) d->evolve_min_timestep = value;
    else if (strcmp(name, "minimum_allowed_height") == 0) d->minimum_allowed_height = value;
    else if (strcmp(name, "minimum_storable_height") == 0) d->minimum_storable_height = value;
    else if (strcmp(name, "maximum_allowed_speed") == 0) d->maximum_allowed_speed = value;
    else if (strcmp(name, "spatial_order") == 0) d->spatial_order = (hydro_int)value;
    else if (strcmp(name, "timestepping_method") == 0) d->timestepping_method = (hydro_int)value;
    else if (strcmp(name, "beta_w") == 0) d->beta_w = value;
    else if (strcmp(name, "optimise_dry_cells") == 0) d->optimise_dry_cells = (hydro_int)value;
    else if (strcmp(name, "xllcorner") == 0) d->xllcorner = value;
    else if (strcmp(name, "yllcorner") == 0) d->yllcorner = value;
    else if (strcmp(name, "zone") == 0) d->zone = (hydro_int)value;
    else
    {
        fprintf(stderr, "hydro: unknown parameter '%s'\n", name);
    }
}

void hydro_domain_set_name(hydro_domain_t* d, const char* name)
{
    if (name)
    {
        strncpy(d->name, name, sizeof(d->name) - 1);
        d->name[sizeof(d->name) - 1] = '\0';
    }
    else
    {
        d->name[0] = '\0';
    }
}

void hydro_domain_set_output_dir(hydro_domain_t* d, const char* dir)
{
    if (dir && dir[0] != '\0')
    {
        strncpy(d->output_dir, dir, sizeof(d->output_dir) - 1);
        d->output_dir[sizeof(d->output_dir) - 1] = '\0';
    }
    else
    {
        strcpy(d->output_dir, ".");
    }
}

void hydro_domain_get_quantity(
    const hydro_domain_t* d, const char* name, double* values)
{
    double* src = NULL;
    /* Look up the centroid array */
    if (strcmp(name, "stage") == 0) src = d->stage_centroid_values;
    else if (strcmp(name, "xmomentum") == 0) src = d->xmom_centroid_values;
    else if (strcmp(name, "ymomentum") == 0) src = d->ymom_centroid_values;
    else if (strcmp(name, "elevation") == 0) src = d->bed_centroid_values;
    else if (strcmp(name, "height") == 0) src = d->height_centroid_values;
    else if (strcmp(name, "friction") == 0) src = d->friction_centroid_values;

    if (src && values)
    {
        for (hydro_int i = 0; i < d->number_of_elements; i++)
        {
            values[i] = src[i];
        }
    }
}

double hydro_domain_get_time(const hydro_domain_t* d)
{
    return d->time;
}
