/**
 * sww.c — SWW file writer using libnetcdf C API
 *
 * Produces byte-compatible SWW (NetCDF) files matching ANUGA output.
 *
 * Adapted from:
 *   anuga/file/sww.py (Write_sww class)
 *   anuga/file/sts.py (Write_sts class)
 */

#include "hydro/sww.h"
#include "hydro/config.h"
#include <netcdf.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#define NCERR(e) do { \
    if ((e) != NC_NOERR) { \
        fprintf(stderr, "hydro SWW: NetCDF error: %s\n", nc_strerror(e)); \
    } \
} while(0)

struct hydro_sww_t {
    int ncid;
    int timestep_id;
    int time_var_id;
    int stage_var_id, xmom_var_id, ymom_var_id;
    int stage_range_id, xmom_range_id, ymom_range_id;
    hydro_int n_points;
    hydro_int n_volumes;
    hydro_int n_triangle_vertices;
    int n_timesteps;
    double starttime;
};

/* NetCDF types matching ANUGA: float32 for quantities, float64 for time, int32 for volumes */
#define SWW_FLOAT  NC_FLOAT
#define SWW_DOUBLE NC_DOUBLE
#define SWW_INT    NC_INT

static void write_global_attr(int ncid, const char* name, const char* value) {
    nc_put_att_text(ncid, NC_GLOBAL, name, strlen(value), value);
}

static void write_global_attr_int(int ncid, const char* name, int value) {
    nc_put_att_int(ncid, NC_GLOBAL, name, NC_INT, 1, &value);
}

static void write_global_attr_double(int ncid, const char* name, double value) {
    nc_put_att_double(ncid, NC_GLOBAL, name, NC_DOUBLE, 1, &value);
}

hydro_sww_t* hydro_sww_create(
    const char* path,
    const hydro_domain_t* domain,
    double starttime)
{
    hydro_sww_t* sww;
    int ncid, ret;

    /* Determine output path (append .sww if not present) */
    char filepath[4096];
    snprintf(filepath, sizeof(filepath), "%s", path);
    size_t len = strlen(filepath);
    if (len < 4 || strcmp(filepath + len - 4, ".sww") != 0) {
        snprintf(filepath + len, sizeof(filepath) - len, ".sww");
    }

    /* Create NetCDF file (64-bit offset format, matching ANUGA) */
    ret = nc_create(filepath, NC_CLOBBER | NC_64BIT_OFFSET, &ncid);
    if (ret != NC_NOERR) {
        NCERR(ret);
        return NULL;
    }

    sww = (hydro_sww_t*)calloc(1, sizeof(hydro_sww_t));
    if (!sww) {
        nc_close(ncid);
        return NULL;
    }
    sww->ncid = ncid;
    sww->starttime = starttime;
    sww->n_timesteps = 0;

    hydro_int N = domain->number_of_elements;
    hydro_int n_vertices = 3 * N;  /* non-unique: 3 vertices per triangle */
    hydro_int n_points = n_vertices;

    sww->n_volumes = N;
    sww->n_triangle_vertices = domain->number_of_nodes;
    sww->n_points = n_points;

    /* ======================================================================
     * Global attributes (matching ANUGA SWW metadata)
     * ====================================================================== */
    write_global_attr(ncid, "institution", HYDRO_INSTITUTION);
    write_global_attr(ncid, "description",
        "Output from hydro_core SWW writer");
    write_global_attr(ncid, "smoothing", "Yes");
    write_global_attr(ncid, "vertices_are_stored_uniquely", "False");
    write_global_attr_int(ncid, "order", domain->spatial_order);
    write_global_attr_double(ncid, "starttime", starttime);
    write_global_attr(ncid, "timezone", "UTC");
    write_global_attr(ncid, "revision_number", "None");
    write_global_attr(ncid, "anuga_version", "hydro_core_0.1.0");

    /* ======================================================================
     * Dimensions
     * ====================================================================== */
    int dim_volumes, dim_tri_vertices, dim_vertices, dim_range;
    int dim_points;

    nc_def_dim(ncid, "number_of_volumes", (size_t)N, &dim_volumes);
    nc_def_dim(ncid, "number_of_triangle_vertices",
               (size_t)domain->number_of_nodes, &dim_tri_vertices);
    nc_def_dim(ncid, "number_of_vertices", 3, &dim_vertices);
    nc_def_dim(ncid, "numbers_in_range", 2, &dim_range);
    nc_def_dim(ncid, "number_of_points", (size_t)n_points, &dim_points);

    int dim_timesteps_unlim;
    nc_def_dim(ncid, "number_of_timesteps", NC_UNLIMITED, &dim_timesteps_unlim);

    /* ======================================================================
     * Variables: geometry
     * ====================================================================== */
    int x_var_id, y_var_id, vol_var_id;

    nc_def_var(ncid, "x", SWW_FLOAT, 1, &dim_points, &x_var_id);
    nc_def_var(ncid, "y", SWW_FLOAT, 1, &dim_points, &y_var_id);

    int vol_dims[2] = {dim_volumes, dim_vertices};
    nc_def_var(ncid, "volumes", SWW_INT, 2, vol_dims, &vol_var_id);

    /* ======================================================================
     * Variables: static quantities
     * ====================================================================== */
    int elev_var_id, elev_range_id;
    int fric_var_id, fric_range_id;

    nc_def_var(ncid, "elevation", SWW_FLOAT, 1, &dim_points, &elev_var_id);
    nc_def_var(ncid, "elevation_range", SWW_FLOAT, 1, &dim_range,
               &elev_range_id);

    nc_def_var(ncid, "friction", SWW_FLOAT, 1, &dim_points, &fric_var_id);
    nc_def_var(ncid, "friction_range", SWW_FLOAT, 1, &dim_range,
               &fric_range_id);

    /* ======================================================================
     * Variables: dynamic quantities (time-varying)
     * ====================================================================== */
    int stage_dims[2] = {dim_timesteps_unlim, dim_points};
    int range_dims[1] = {dim_range};

    nc_def_var(ncid, "stage", SWW_FLOAT, 2, stage_dims, &sww->stage_var_id);
    nc_def_var(ncid, "stage_range", SWW_FLOAT, 1, range_dims,
               &sww->stage_range_id);

    nc_def_var(ncid, "xmomentum", SWW_FLOAT, 2, stage_dims,
               &sww->xmom_var_id);
    nc_def_var(ncid, "xmomentum_range", SWW_FLOAT, 1, range_dims,
               &sww->xmom_range_id);

    nc_def_var(ncid, "ymomentum", SWW_FLOAT, 2, stage_dims,
               &sww->ymom_var_id);
    nc_def_var(ncid, "ymomentum_range", SWW_FLOAT, 1, range_dims,
               &sww->ymom_range_id);

    /* Time variable */
    int time_dims[1] = {dim_timesteps_unlim};
    nc_def_var(ncid, "time", SWW_DOUBLE, 1, time_dims, &sww->time_var_id);

    /* End define mode */
    ret = nc_enddef(ncid);
    if (ret != NC_NOERR) {
        NCERR(ret);
        free(sww);
        nc_close(ncid);
        return NULL;
    }

    /* ======================================================================
     * Write geometry data
     * ====================================================================== */

    /* Build per-vertex x, y arrays (non-unique: 3 * N entries) */
    float* x_data = (float*)malloc((size_t)n_points * sizeof(float));
    float* y_data = (float*)malloc((size_t)n_points * sizeof(float));

    for (hydro_int k = 0; k < N; k++) {
        hydro_int k3 = 3 * k;
        for (int v = 0; v < 3; v++) {
            hydro_int idx = k3 + v;
            x_data[idx] = (float)domain->vertex_coordinates[2 * k3 + 2 * v];
            y_data[idx] = (float)domain->vertex_coordinates[2 * k3 + 2 * v + 1];
        }
    }

    nc_put_var_float(ncid, x_var_id, x_data);
    nc_put_var_float(ncid, y_var_id, y_data);
    free(x_data);
    free(y_data);

    /* Write volumes (triangle connectivity) */
    int* vol_data = (int*)malloc(3 * (size_t)N * sizeof(int));
    for (hydro_int k = 0; k < N; k++) {
        hydro_int k3 = 3 * k;
        vol_data[k3]     = (int)domain->triangles[k3];
        vol_data[k3 + 1] = (int)domain->triangles[k3 + 1];
        vol_data[k3 + 2] = (int)domain->triangles[k3 + 2];
    }
    size_t vol_start[2] = {0, 0};
    size_t vol_count[2] = {(size_t)N, 3};
    nc_put_vara_int(ncid, vol_var_id, vol_start, vol_count, vol_data);
    free(vol_data);

    /* ======================================================================
     * Write static quantities (elevation, friction)
     * ====================================================================== */

    /* Elevation at vertices */
    float* elev_data = (float*)malloc((size_t)n_points * sizeof(float));
    float elev_min = (float)HYDRO_MAX_FLOAT;
    float elev_max = -(float)HYDRO_MAX_FLOAT;

    if (domain->bed_vertex_values) {
        for (hydro_int i = 0; i < n_points; i++) {
            float val = (float)domain->bed_vertex_values[i];
            elev_data[i] = val;
            if (val < elev_min) elev_min = val;
            if (val > elev_max) elev_max = val;
        }
    } else if (domain->bed_centroid_values) {
        /* If vertex values not available, use centroid values for each vertex */
        for (hydro_int k = 0; k < N; k++) {
            float val = (float)domain->bed_centroid_values[k];
            for (int v = 0; v < 3; v++) {
                elev_data[3*k + v] = val;
            }
            if (val < elev_min) elev_min = val;
            if (val > elev_max) elev_max = val;
        }
    } else {
        /* No elevation set — write zeros */
        elev_min = 0.0f;
        elev_max = 0.0f;
    }

    nc_put_var_float(ncid, elev_var_id, elev_data);
    float elev_range[2] = {elev_min, elev_max};
    nc_put_var_float(ncid, elev_range_id, elev_range);

    /* Friction at vertices */
    float* fric_data = (float*)malloc((size_t)n_points * sizeof(float));
    float fric_min = (float)HYDRO_MAX_FLOAT;
    float fric_max = -(float)HYDRO_MAX_FLOAT;

    if (domain->friction_centroid_values) {
        for (hydro_int k = 0; k < N; k++) {
            float val = (float)domain->friction_centroid_values[k];
            for (int v = 0; v < 3; v++) {
                fric_data[3*k + v] = val;
            }
            if (val < fric_min) fric_min = val;
            if (val > fric_max) fric_max = val;
        }
    } else {
        /* Default friction */
        for (hydro_int i = 0; i < n_points; i++) {
            fric_data[i] = (float)HYDRO_MANNING_DEFAULT;
        }
        fric_min = fric_max = (float)HYDRO_MANNING_DEFAULT;
    }

    nc_put_var_float(ncid, fric_var_id, fric_data);
    float fric_range[2] = {fric_min, fric_max};
    nc_put_var_float(ncid, fric_range_id, fric_range);
    free(fric_data);
    free(elev_data);

    /* ======================================================================
     * Initialise dynamic quantity ranges
     * ====================================================================== */
    float init_range[2] = {(float)HYDRO_MAX_FLOAT, -(float)HYDRO_MAX_FLOAT};
    nc_put_var_float(ncid, sww->stage_range_id, init_range);
    nc_put_var_float(ncid, sww->xmom_range_id, init_range);
    nc_put_var_float(ncid, sww->ymom_range_id, init_range);

    nc_sync(ncid);

    return sww;
}

int hydro_sww_store_timestep(
    hydro_sww_t* sww,
    const hydro_domain_t* domain,
    double time)
{
    if (!sww) return -1;

    hydro_int N = domain->number_of_elements;
    hydro_int n_points = sww->n_points;
    int ncid = sww->ncid;
    int ti = sww->n_timesteps;

    /* Write time value */
    double time_val = time;
    size_t time_start[1] = {(size_t)ti};
    size_t time_count[1] = {1};
    nc_put_vara_double(ncid, sww->time_var_id, time_start, time_count, &time_val);

    /* Build and write stage, xmomentum, ymomentum at vertices */
    float* data = (float*)malloc((size_t)n_points * sizeof(float));
    float q_min, q_max;

    /* ---- Stage ---- */
    q_min = (float)HYDRO_MAX_FLOAT;
    q_max = -(float)HYDRO_MAX_FLOAT;
    if (domain->stage_vertex_values) {
        for (hydro_int i = 0; i < n_points; i++) {
            float val = (float)domain->stage_vertex_values[i];
            data[i] = val;
            if (val < q_min) q_min = val;
            if (val > q_max) q_max = val;
        }
    } else if (domain->stage_centroid_values) {
        for (hydro_int k = 0; k < N; k++) {
            float val = (float)domain->stage_centroid_values[k];
            for (int v = 0; v < 3; v++) {
                data[3*k + v] = val;
            }
            if (val < q_min) q_min = val;
            if (val > q_max) q_max = val;
        }
    }

    size_t q_start[2] = {(size_t)ti, 0};
    size_t q_count[2] = {1, (size_t)n_points};
    nc_put_vara_float(ncid, sww->stage_var_id, q_start, q_count, data);

    /* Update stage range */
    float range_cur[2];
    nc_get_var_float(ncid, sww->stage_range_id, range_cur);
    if (q_min < range_cur[0]) range_cur[0] = q_min;
    if (q_max > range_cur[1]) range_cur[1] = q_max;
    nc_put_var_float(ncid, sww->stage_range_id, range_cur);

    /* ---- Xmomentum ---- */
    q_min = (float)HYDRO_MAX_FLOAT;
    q_max = -(float)HYDRO_MAX_FLOAT;
    if (domain->xmom_vertex_values) {
        for (hydro_int i = 0; i < n_points; i++) {
            float val = (float)domain->xmom_vertex_values[i];
            data[i] = val;
            if (val < q_min) q_min = val;
            if (val > q_max) q_max = val;
        }
    } else if (domain->xmom_centroid_values) {
        for (hydro_int k = 0; k < N; k++) {
            float val = (float)domain->xmom_centroid_values[k];
            for (int v = 0; v < 3; v++) {
                data[3*k + v] = val;
            }
            if (val < q_min) q_min = val;
            if (val > q_max) q_max = val;
        }
    }
    nc_put_vara_float(ncid, sww->xmom_var_id, q_start, q_count, data);

    nc_get_var_float(ncid, sww->xmom_range_id, range_cur);
    if (q_min < range_cur[0]) range_cur[0] = q_min;
    if (q_max > range_cur[1]) range_cur[1] = q_max;
    nc_put_var_float(ncid, sww->xmom_range_id, range_cur);

    /* ---- Ymomentum ---- */
    q_min = (float)HYDRO_MAX_FLOAT;
    q_max = -(float)HYDRO_MAX_FLOAT;
    if (domain->ymom_vertex_values) {
        for (hydro_int i = 0; i < n_points; i++) {
            float val = (float)domain->ymom_vertex_values[i];
            data[i] = val;
            if (val < q_min) q_min = val;
            if (val > q_max) q_max = val;
        }
    } else if (domain->ymom_centroid_values) {
        for (hydro_int k = 0; k < N; k++) {
            float val = (float)domain->ymom_centroid_values[k];
            for (int v = 0; v < 3; v++) {
                data[3*k + v] = val;
            }
            if (val < q_min) q_min = val;
            if (val > q_max) q_max = val;
        }
    }
    nc_put_vara_float(ncid, sww->ymom_var_id, q_start, q_count, data);

    nc_get_var_float(ncid, sww->ymom_range_id, range_cur);
    if (q_min < range_cur[0]) range_cur[0] = q_min;
    if (q_max > range_cur[1]) range_cur[1] = q_max;
    nc_put_var_float(ncid, sww->ymom_range_id, range_cur);

    free(data);
    nc_sync(ncid);

    sww->n_timesteps++;
    return 0;
}

int hydro_sww_close(hydro_sww_t* sww) {
    if (!sww) return -1;
    int ret = nc_close(sww->ncid);
    if (ret != NC_NOERR) {
        NCERR(ret);
    }
    free(sww);
    return ret;
}
