/**
 * sww.c — SWW file writer using libnetcdf C API
 *
 * Produces byte-compatible SWW (NetCDF) files matching ANUGA output.
 * Stores vertices uniquely with per-vertex averaged quantities.
 *
 * Adapted from:
 *   anuga/file/sww.py (Write_sww class)
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
    hydro_int n_points;            /* number of UNIQUE vertices */
    hydro_int n_volumes;
    int n_timesteps;
    double starttime;

    /* Vertex deduplication tables (built on create, used on store) */
    hydro_int n_expanded;          /* 3 * n_volumes */
    hydro_int* exp_to_unique;      /* [n_expanded] maps expanded idx → unique idx */
    hydro_int* unique_count;       /* [n_points] count of expanded entries per unique */
};

/* NetCDF types matching ANUGA: float32 for quantities, float64 for time, int32 for volumes */
#define SWW_FLOAT  NC_FLOAT
#define SWW_DOUBLE NC_DOUBLE
#define SWW_INT    NC_INT

/* Hash entry for vertex deduplication */
typedef struct {
    double x, y;
    hydro_int unique_id;
    int used;
} vert_hash_entry_t;

static hydro_int vert_hash(double x, double y, hydro_int size) {
    unsigned long long h = 14695981039346656037ULL;
    unsigned char* p = (unsigned char*)&x;
    for (int i = 0; i < 8; i++) { h ^= p[i]; h *= 1099511628211ULL; }
    p = (unsigned char*)&y;
    for (int i = 0; i < 8; i++) { h ^= p[i]; h *= 1099511628211ULL; }
    return (hydro_int)(h % (unsigned long long)size);
}

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

    char filepath[4096];
    snprintf(filepath, sizeof(filepath), "%s", path);
    size_t len = strlen(filepath);
    if (len < 4 || strcmp(filepath + len - 4, ".sww") != 0) {
        snprintf(filepath + len, sizeof(filepath) - len, ".sww");
    }

    ret = nc_create(filepath, NC_CLOBBER | NC_64BIT_OFFSET, &ncid);
    if (ret != NC_NOERR) { NCERR(ret); return NULL; }

    sww = (hydro_sww_t*)calloc(1, sizeof(hydro_sww_t));
    if (!sww) { nc_close(ncid); return NULL; }
    sww->ncid = ncid;
    sww->starttime = starttime;
    sww->n_timesteps = 0;

    hydro_int N = domain->number_of_elements;
    hydro_int n_exp = 3 * N;

    /* ==================================================================
     * Deduplicate expanded vertex_coordinates → unique vertices
     * ================================================================== */
    hydro_int hash_size = n_exp * 2;
    vert_hash_entry_t* hash = (vert_hash_entry_t*)calloc((size_t)hash_size,
        sizeof(vert_hash_entry_t));

    sww->exp_to_unique = (hydro_int*)malloc((size_t)n_exp * sizeof(hydro_int));
    hydro_int n_unique = 0;

    for (hydro_int ei = 0; ei < n_exp; ei++) {
        double x = domain->vertex_coordinates[2*ei];
        double y = domain->vertex_coordinates[2*ei + 1];
        hydro_int hi = vert_hash(x, y, hash_size);

        /* Linear probe */
        while (hash[hi].used) {
            if (fabs(hash[hi].x - x) < 1e-10 && fabs(hash[hi].y - y) < 1e-10) {
                sww->exp_to_unique[ei] = hash[hi].unique_id;
                break;
            }
            hi = (hi + 1) % hash_size;
        }
        if (!hash[hi].used) {
            hash[hi].x = x; hash[hi].y = y;
            hash[hi].unique_id = n_unique;
            hash[hi].used = 1;
            sww->exp_to_unique[ei] = n_unique;
            n_unique++;
        }
    }
    free(hash);

    /* Count how many expanded entries per unique vertex */
    sww->unique_count = (hydro_int*)calloc((size_t)n_unique, sizeof(hydro_int));
    for (hydro_int ei = 0; ei < n_exp; ei++) {
        sww->unique_count[sww->exp_to_unique[ei]]++;
    }

    sww->n_points = n_unique;
    sww->n_volumes = N;
    sww->n_expanded = n_exp;

    /* ==================================================================
     * Global attributes
     * ================================================================== */
    write_global_attr(ncid, "institution", HYDRO_INSTITUTION);
    write_global_attr(ncid, "description",
        "Output from hydro_core SWW writer");
    write_global_attr(ncid, "smoothing", "Yes");
    write_global_attr(ncid, "vertices_are_stored_uniquely", "False");
    write_global_attr_int(ncid, "order", domain->spatial_order);
    write_global_attr_double(ncid, "starttime", starttime);
    write_global_attr(ncid, "timezone", "UTC");
    write_global_attr(ncid, "revision_number", "None");
    write_global_attr(ncid, "version", "hydro_core_0.1.0");

    write_global_attr_double(ncid, "xllcorner", domain->xllcorner);
    write_global_attr_double(ncid, "yllcorner", domain->yllcorner);
    {
        char buf[32];
        snprintf(buf, sizeof(buf), "%lld", (long long)domain->zone);
        write_global_attr(ncid, "zone", buf);
    }
    write_global_attr(ncid, "datum", domain->datum);
    write_global_attr(ncid, "projection", domain->projection);
    write_global_attr(ncid, "units", domain->units);
    {
        char buf[32];
        int fe = 500000, fn = (starttime != starttime) ? 0 : 10000000;
        snprintf(buf, sizeof(buf), "%d", fe);
        write_global_attr(ncid, "false_easting", buf);
        snprintf(buf, sizeof(buf), "%d", fn);
        write_global_attr(ncid, "false_northing", buf);
    }
    write_global_attr(ncid, "hemisphere", "southern");

    /* ==================================================================
     * Dimensions — matching ANUGA exactly
     * ================================================================== */
    int dim_volumes, dim_vertices, dim_range, dim_points;

    nc_def_dim(ncid, "number_of_volumes", (size_t)N, &dim_volumes);
    nc_def_dim(ncid, "number_of_vertices", 3, &dim_vertices);
    nc_def_dim(ncid, "numbers_in_range", 2, &dim_range);
    nc_def_dim(ncid, "number_of_points", (size_t)n_unique, &dim_points);

    int dim_timesteps_unlim;
    nc_def_dim(ncid, "number_of_timesteps", NC_UNLIMITED, &dim_timesteps_unlim);

    /* ==================================================================
     * Variables: geometry
     * ================================================================== */
    int x_var_id, y_var_id, vol_var_id;

    nc_def_var(ncid, "x", SWW_FLOAT, 1, &dim_points, &x_var_id);
    nc_def_var(ncid, "y", SWW_FLOAT, 1, &dim_points, &y_var_id);

    int vol_dims[2] = {dim_volumes, dim_vertices};
    nc_def_var(ncid, "volumes", SWW_INT, 2, vol_dims, &vol_var_id);

    /* ==================================================================
     * Variables: static quantities (per unique vertex)
     * ================================================================== */
    int elev_var_id, elev_range_id, fric_var_id, fric_range_id;

    nc_def_var(ncid, "elevation", SWW_FLOAT, 1, &dim_points, &elev_var_id);
    nc_def_var(ncid, "elevation_range", SWW_FLOAT, 1, &dim_range, &elev_range_id);
    nc_def_var(ncid, "friction", SWW_FLOAT, 1, &dim_points, &fric_var_id);
    nc_def_var(ncid, "friction_range", SWW_FLOAT, 1, &dim_range, &fric_range_id);

    /* ==================================================================
     * Variables: dynamic quantities
     * ================================================================== */
    int stage_dims[2] = {dim_timesteps_unlim, dim_points};
    int range_dims[1] = {dim_range};

    nc_def_var(ncid, "stage", SWW_FLOAT, 2, stage_dims, &sww->stage_var_id);
    nc_def_var(ncid, "stage_range", SWW_FLOAT, 1, range_dims, &sww->stage_range_id);
    nc_def_var(ncid, "xmomentum", SWW_FLOAT, 2, stage_dims, &sww->xmom_var_id);
    nc_def_var(ncid, "xmomentum_range", SWW_FLOAT, 1, range_dims, &sww->xmom_range_id);
    nc_def_var(ncid, "ymomentum", SWW_FLOAT, 2, stage_dims, &sww->ymom_var_id);
    nc_def_var(ncid, "ymomentum_range", SWW_FLOAT, 1, range_dims, &sww->ymom_range_id);

    int time_dims[1] = {dim_timesteps_unlim};
    nc_def_var(ncid, "time", SWW_DOUBLE, 1, time_dims, &sww->time_var_id);

    ret = nc_enddef(ncid);
    if (ret != NC_NOERR) { NCERR(ret); free(sww); nc_close(ncid); return NULL; }

    /* ==================================================================
     * Write unique x, y coordinates
     * ================================================================== */
    float* x_data = (float*)malloc((size_t)n_unique * sizeof(float));
    float* y_data = (float*)malloc((size_t)n_unique * sizeof(float));
    /* Initialise to 0 for vertices that might not be populated */
    for (hydro_int i = 0; i < n_unique; i++) { x_data[i] = 0.0f; y_data[i] = 0.0f; }

    for (hydro_int ei = 0; ei < n_exp; ei++) {
        hydro_int ui = sww->exp_to_unique[ei];
        x_data[ui] = (float)domain->vertex_coordinates[2*ei];
        y_data[ui] = (float)domain->vertex_coordinates[2*ei + 1];
    }
    nc_put_var_float(ncid, x_var_id, x_data);
    nc_put_var_float(ncid, y_var_id, y_data);
    free(x_data); free(y_data);

    /* ==================================================================
     * Write volumes (remapped to unique indices)
     * ================================================================== */
    int* vol_data = (int*)malloc(3 * (size_t)N * sizeof(int));
    for (hydro_int k = 0; k < N; k++) {
        hydro_int k3 = 3 * k;
        for (int v = 0; v < 3; v++) {
            vol_data[k3 + v] = (int)sww->exp_to_unique[k3 + v];
        }
    }
    size_t vol_start[2] = {0, 0};
    size_t vol_count[2] = {(size_t)N, 3};
    nc_put_vara_int(ncid, vol_var_id, vol_start, vol_count, vol_data);
    free(vol_data);

    /* ==================================================================
     * Write elevation (averaged to unique vertices)
     * ================================================================== */
    float* elev_data = (float*)calloc((size_t)n_unique, sizeof(float));
    if (domain->bed_vertex_values) {
        for (hydro_int ei = 0; ei < n_exp; ei++) {
            hydro_int ui = sww->exp_to_unique[ei];
            elev_data[ui] += (float)domain->bed_vertex_values[ei];
        }
    } else if (domain->bed_centroid_values) {
        for (hydro_int k = 0; k < N; k++) {
            float val = (float)domain->bed_centroid_values[k];
            for (int v = 0; v < 3; v++) {
                hydro_int ui = sww->exp_to_unique[3*k + v];
                elev_data[ui] += val;
            }
        }
    }
    /* Average */
    for (hydro_int i = 0; i < n_unique; i++) {
        if (sww->unique_count[i] > 0) elev_data[i] /= (float)sww->unique_count[i];
    }
    nc_put_var_float(ncid, elev_var_id, elev_data);

    float elev_min = elev_data[0], elev_max = elev_data[0];
    for (hydro_int i = 1; i < n_unique; i++) {
        if (elev_data[i] < elev_min) elev_min = elev_data[i];
        if (elev_data[i] > elev_max) elev_max = elev_data[i];
    }
    float elev_range[2] = {elev_min, elev_max};
    nc_put_var_float(ncid, elev_range_id, elev_range);
    free(elev_data);

    /* ==================================================================
     * Write friction (averaged to unique vertices)
     * ================================================================== */
    float* fric_data = (float*)calloc((size_t)n_unique, sizeof(float));
    if (domain->friction_centroid_values) {
        for (hydro_int k = 0; k < N; k++) {
            float val = (float)domain->friction_centroid_values[k];
            for (int v = 0; v < 3; v++) {
                hydro_int ui = sww->exp_to_unique[3*k + v];
                fric_data[ui] += val;
            }
        }
    } else {
        for (hydro_int i = 0; i < n_unique; i++) {
            fric_data[i] = (float)HYDRO_MANNING_DEFAULT;
        }
    }
    for (hydro_int i = 0; i < n_unique; i++) {
        if (sww->unique_count[i] > 0) fric_data[i] /= (float)sww->unique_count[i];
    }
    nc_put_var_float(ncid, fric_var_id, fric_data);

    float fric_min = fric_data[0], fric_max = fric_data[0];
    for (hydro_int i = 1; i < n_unique; i++) {
        if (fric_data[i] < fric_min) fric_min = fric_data[i];
        if (fric_data[i] > fric_max) fric_max = fric_data[i];
    }
    float fric_range[2] = {fric_min, fric_max};
    nc_put_var_float(ncid, fric_range_id, fric_range);
    free(fric_data);

    /* Initialise dynamic quantity ranges */
    float init_range[2] = {(float)HYDRO_MAX_FLOAT, -(float)HYDRO_MAX_FLOAT};
    nc_put_var_float(ncid, sww->stage_range_id, init_range);
    nc_put_var_float(ncid, sww->xmom_range_id, init_range);
    nc_put_var_float(ncid, sww->ymom_range_id, init_range);

    nc_sync(ncid);
    return sww;
}

/* ==========================================================================
 * Open existing SWW for append
 * ========================================================================== */

hydro_sww_t* hydro_sww_open(
    const char* path,
    const hydro_domain_t* domain)
{
    hydro_sww_t* sww;
    int ncid, ret;
    size_t n_timesteps_existing;

    ret = nc_open(path, NC_WRITE, &ncid);
    if (ret != NC_NOERR) { NCERR(ret); return NULL; }

    sww = (hydro_sww_t*)calloc(1, sizeof(hydro_sww_t));
    if (!sww) { nc_close(ncid); return NULL; }
    sww->ncid = ncid;

    /* Read existing timestep count */
    int time_dim_id;
    nc_inq_dimid(ncid, "number_of_timesteps", &time_dim_id);
    nc_inq_dimlen(ncid, time_dim_id, &n_timesteps_existing);
    sww->n_timesteps = (int)n_timesteps_existing;

    /* Get variable IDs */
    nc_inq_varid(ncid, "time",            &sww->time_var_id);
    nc_inq_varid(ncid, "stage",           &sww->stage_var_id);
    nc_inq_varid(ncid, "stage_range",     &sww->stage_range_id);
    nc_inq_varid(ncid, "xmomentum",       &sww->xmom_var_id);
    nc_inq_varid(ncid, "xmomentum_range", &sww->xmom_range_id);
    nc_inq_varid(ncid, "ymomentum",       &sww->ymom_var_id);
    nc_inq_varid(ncid, "ymomentum_range", &sww->ymom_range_id);

    /* Read starttime */
    {
        double st = 0.0;
        nc_get_att_double(ncid, NC_GLOBAL, "starttime", &st);
        sww->starttime = st;
    }

    /* Rebuild vertex deduplication tables (same logic as create) */
    hydro_int N = domain->number_of_elements;
    hydro_int n_exp = 3 * N;
    hydro_int hash_size = n_exp * 2;
    vert_hash_entry_t* hash = (vert_hash_entry_t*)calloc((size_t)hash_size,
        sizeof(vert_hash_entry_t));

    sww->exp_to_unique = (hydro_int*)malloc((size_t)n_exp * sizeof(hydro_int));
    hydro_int n_unique = 0;

    for (hydro_int ei = 0; ei < n_exp; ei++) {
        double x = domain->vertex_coordinates[2*ei];
        double y = domain->vertex_coordinates[2*ei + 1];
        hydro_int hi = vert_hash(x, y, hash_size);

        while (hash[hi].used) {
            if (fabs(hash[hi].x - x) < 1e-10 && fabs(hash[hi].y - y) < 1e-10) {
                sww->exp_to_unique[ei] = hash[hi].unique_id;
                break;
            }
            hi = (hi + 1) % hash_size;
        }
        if (!hash[hi].used) {
            hash[hi].x = x; hash[hi].y = y;
            hash[hi].unique_id = n_unique;
            hash[hi].used = 1;
            sww->exp_to_unique[ei] = n_unique;
            n_unique++;
        }
    }
    free(hash);

    sww->unique_count = (hydro_int*)calloc((size_t)n_unique, sizeof(hydro_int));
    for (hydro_int ei = 0; ei < n_exp; ei++) {
        sww->unique_count[sww->exp_to_unique[ei]]++;
    }

    sww->n_points   = n_unique;
    sww->n_volumes  = N;
    sww->n_expanded = n_exp;

    printf("hydro: opened SWW '%s' for append (n_timesteps=%d, n_points=%lld)\n",
           path, sww->n_timesteps, (long long)n_unique);

    return sww;
}

/* ==========================================================================
 * Average expanded per-vertex data to unique vertices
 * ========================================================================== */
static void avg_to_unique(const hydro_sww_t* sww, const double* expanded,
                           float* unique_out) {
    hydro_int n_exp = sww->n_expanded;
    hydro_int n_uniq = sww->n_points;

    /* Reset */
    for (hydro_int i = 0; i < n_uniq; i++) unique_out[i] = 0.0f;

    if (expanded) {
        for (hydro_int ei = 0; ei < n_exp; ei++) {
            unique_out[sww->exp_to_unique[ei]] += (float)expanded[ei];
        }
    }
    for (hydro_int i = 0; i < n_uniq; i++) {
        if (sww->unique_count[i] > 0)
            unique_out[i] /= (float)sww->unique_count[i];
    }
}

int hydro_sww_store_timestep(
    hydro_sww_t* sww,
    const hydro_domain_t* domain,
    double time)
{
    if (!sww) return -1;

    hydro_int n_uniq = sww->n_points;
    int ncid = sww->ncid;
    int ti = sww->n_timesteps;

    /* Write time value */
    double time_val = time;
    size_t time_start[1] = {(size_t)ti};
    size_t time_count[1] = {1};
    nc_put_vara_double(ncid, sww->time_var_id, time_start, time_count, &time_val);

    float* uniq = (float*)malloc((size_t)n_uniq * sizeof(float));
    size_t q_start[2] = {(size_t)ti, 0};
    size_t q_count[2] = {1, (size_t)n_uniq};

    /* ---- Stage ---- */
    avg_to_unique(sww, domain->stage_vertex_values, uniq);
    nc_put_vara_float(ncid, sww->stage_var_id, q_start, q_count, uniq);
    /* Update range */
    float range_cur[2];
    nc_get_var_float(ncid, sww->stage_range_id, range_cur);
    for (hydro_int i = 0; i < n_uniq; i++) {
        if (uniq[i] < range_cur[0]) range_cur[0] = uniq[i];
        if (uniq[i] > range_cur[1]) range_cur[1] = uniq[i];
    }
    nc_put_var_float(ncid, sww->stage_range_id, range_cur);

    /* ---- Xmomentum ---- */
    avg_to_unique(sww, domain->xmom_vertex_values, uniq);
    nc_put_vara_float(ncid, sww->xmom_var_id, q_start, q_count, uniq);
    nc_get_var_float(ncid, sww->xmom_range_id, range_cur);
    for (hydro_int i = 0; i < n_uniq; i++) {
        if (uniq[i] < range_cur[0]) range_cur[0] = uniq[i];
        if (uniq[i] > range_cur[1]) range_cur[1] = uniq[i];
    }
    nc_put_var_float(ncid, sww->xmom_range_id, range_cur);

    /* ---- Ymomentum ---- */
    avg_to_unique(sww, domain->ymom_vertex_values, uniq);
    nc_put_vara_float(ncid, sww->ymom_var_id, q_start, q_count, uniq);
    nc_get_var_float(ncid, sww->ymom_range_id, range_cur);
    for (hydro_int i = 0; i < n_uniq; i++) {
        if (uniq[i] < range_cur[0]) range_cur[0] = uniq[i];
        if (uniq[i] > range_cur[1]) range_cur[1] = uniq[i];
    }
    nc_put_var_float(ncid, sww->ymom_range_id, range_cur);

    free(uniq);
    nc_sync(ncid);

    sww->n_timesteps++;
    return 0;
}

int hydro_sww_close(hydro_sww_t* sww) {
    if (!sww) return -1;
    int ret = nc_close(sww->ncid);
    if (ret != NC_NOERR) { NCERR(ret); }
    free(sww->exp_to_unique);
    free(sww->unique_count);
    free(sww);
    return ret;
}
