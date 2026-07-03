#ifndef HYDRO_DOMAIN_H
#define HYDRO_DOMAIN_H

#include "types.h"
#include "config.h"

/* ==========================================================================
 * hydro_domain_t — the core computational domain
 *
 * All arrays are flat, contiguous, and owned by the domain.
 * Structure-of-Arrays (SoA) layout for cache efficiency.
 * ========================================================================== */

typedef struct {
    /* ---- Scalar parameters (set once, read during evolve) ---- */
    hydro_int number_of_elements;       /* number of triangles */
    hydro_int number_of_nodes;          /* number of unique vertices */
    hydro_int number_of_edges;          /* 3 * number_of_elements */
    hydro_int boundary_length;          /* number of boundary edges */
    hydro_int number_of_riverwall_edges;
    hydro_int optimise_dry_cells;
    hydro_int extrapolate_velocity_second_order;
    hydro_int low_froude;
    hydro_int timestep_fluxcalls;
    hydro_int max_flux_update_frequency;
    hydro_int ncol_riverwall_hydraulic_properties;

    /* Physical/numerical parameters */
    double epsilon;
    double H0;                          /* H0 = minimum_allowed_height */
    double g;                           /* gravity */
    double evolve_max_timestep;
    double evolve_min_timestep;
    double minimum_allowed_height;
    double maximum_allowed_speed;

    /* Limiter beta parameters */
    double beta_w;
    double beta_w_dry;
    double beta_uh;
    double beta_uh_dry;
    double beta_vh;
    double beta_vh_dry;

    /* CFL factor */
    double CFL;

    /* ---- Mesh geometry (read-only after setup) ---- */
    /* vertex_coordinates: [3*n_elements * 2] flattened, x0,y0,x1,y1,x2,y2 */
    double* vertex_coordinates;
    /* edge_coordinates: [3*n_elements * 2] flattened, edge midpoint x,y */
    double* edge_coordinates;
    /* centroid_coordinates: [n_elements * 2] */
    double* centroid_coordinates;
    /* normals: [3*n_elements * 2] flattened outward edge normals */
    double* normals;
    /* edgelengths: [3*n_elements] */
    double* edgelengths;
    /* radii: [n_elements] inscribed circle radii */
    double* radii;
    /* areas: [n_elements] triangle areas */
    double* areas;

    /* ---- Mesh topology (read-only after setup) ---- */
    /* triangles: [n_elements * 3] vertex indices per triangle */
    hydro_int* triangles;
    /* neighbours: [3*n_elements] neighbour triangle id, -1 = boundary */
    hydro_int* neighbours;
    /* neighbour_edges: [3*n_elements] which edge of neighbour connects back */
    hydro_int* neighbour_edges;
    /* surrogate_neighbours: [3*n_elements] like neighbours but self-ref at boundary */
    hydro_int* surrogate_neighbours;
    /* edge_flux_type: [3*n_elements] 0=normal, 1=riverwall, etc. */
    hydro_int* edge_flux_type;
    /* edge_river_wall_counter: [3*n_elements] */
    hydro_int* edge_river_wall_counter;
    /* tri_full_flag: [n_elements] 1=full cell, 0=ghost */
    hydro_int* tri_full_flag;
    /* already_computed_flux: [3*n_elements] */
    hydro_int* already_computed_flux;
    /* number_of_boundaries: [n_elements] boundary edge count per triangle */
    hydro_int* number_of_boundaries;

    /* ---- Conserved quantities (edge values) ---- */
    double* stage_edge_values;          /* [3*n_elements] */
    double* xmom_edge_values;           /* [3*n_elements] */
    double* ymom_edge_values;           /* [3*n_elements] */

    /* ---- Auxiliary quantities (edge values) ---- */
    double* bed_edge_values;            /* [3*n_elements] */
    double* height_edge_values;         /* [3*n_elements] */
    double* xvelocity_edge_values;      /* [3*n_elements] */
    double* yvelocity_edge_values;      /* [3*n_elements] */

    /* ---- Conserved quantities (centroid values — primary state) ---- */
    double* stage_centroid_values;      /* [n_elements] */
    double* xmom_centroid_values;       /* [n_elements] */
    double* ymom_centroid_values;       /* [n_elements] */

    /* ---- Auxiliary quantities (centroid values) ---- */
    double* bed_centroid_values;        /* [n_elements] */
    double* height_centroid_values;     /* [n_elements] */
    double* friction_centroid_values;   /* [n_elements] */
    double* xvelocity_centroid_values;  /* [n_elements] */
    double* yvelocity_centroid_values;  /* [n_elements] */

    /* ---- Conserved quantities (vertex values) ---- */
    double* stage_vertex_values;        /* [3*n_elements] */
    double* xmom_vertex_values;         /* [3*n_elements] */
    double* ymom_vertex_values;         /* [3*n_elements] */

    /* ---- Auxiliary quantities (vertex values) ---- */
    double* bed_vertex_values;          /* [3*n_elements] */
    double* height_vertex_values;       /* [3*n_elements] */

    /* ---- Boundary values (length = boundary_length) ---- */
    double* stage_boundary_values;
    double* xmom_boundary_values;
    double* ymom_boundary_values;
    double* bed_boundary_values;
    double* height_boundary_values;
    double* xvelocity_boundary_values;
    double* yvelocity_boundary_values;

    /* ---- Explicit and semi-implicit updates (per quantity, per element) ---- */
    double* stage_explicit_update;      /* [n_elements] */
    double* xmom_explicit_update;       /* [n_elements] */
    double* ymom_explicit_update;       /* [n_elements] */

    double* stage_semi_implicit_update; /* [n_elements] */
    double* xmom_semi_implicit_update;  /* [n_elements] */
    double* ymom_semi_implicit_update;  /* [n_elements] */

    /* ---- Work and pre-compute arrays ---- */
    double* max_speed;                  /* [n_elements], diagnostic */
    double* edge_timestep;              /* [3*n_elements] */
    double* edge_qr_stage;             /* [3*n_elements] pre-computed right stage */
    double* edge_qr_xmom;              /* [3*n_elements] pre-computed right xmom  */
    double* edge_qr_ymom;              /* [3*n_elements] pre-computed right ymom  */
    double* edge_zr;                   /* [3*n_elements] pre-computed right bed   */
    double* edge_h_left;               /* [3*n_elements] pre-computed left depth  */
    double* edge_hre;                 /* [3*n_elements] pre-computed right edge h  */
    double* edge_h_right;              /* [3*n_elements] pre-computed right depth */
    double* edge_z_half;               /* [3*n_elements] pre-computed max bed     */
    double* x_centroid_work;            /* [n_elements] */
    double* y_centroid_work;            /* [n_elements] */

    /* Boundary flux tracking */
    double* boundary_flux_sum;          /* [max_time_substeps] */

    /* Flux update frequency control */
    hydro_int* flux_update_frequency;   /* [n_elements] */
    hydro_int* update_next_flux;        /* [n_elements] */
    hydro_int* update_extrapolation;    /* [n_elements] */
    hydro_int* allow_timestep_increase; /* [n_elements] */

    /* ---- Backup arrays for RK multi-stage ---- */
    double* stage_backup_values;        /* [n_elements] */
    double* xmom_backup_values;         /* [n_elements] */
    double* ymom_backup_values;         /* [n_elements] */

    /* ---- Riverwall data ---- */
    double* riverwall_elevation;
    hydro_int* riverwall_rowIndex;
    double* riverwall_hydraulic_properties;

    /* ---- Time state ---- */
    double time;                        /* current simulation time */
    double relative_time;               /* time relative to starttime */
    double starttime;                   /* absolute start time (epoch seconds) */
    double timestep;                    /* current timestep */
    double flux_timestep;               /* CFL-constrained timestep */
    hydro_int yieldstep_counter;        /* steps since last yield */
    hydro_int step;                     /* total step count */

    /* ---- Geo-reference (for SWW output) ---- */
    double xllcorner;                   /* X origin of local coords in UTM */
    double yllcorner;                   /* Y origin of local coords in UTM */
    hydro_int zone;                     /* UTM zone (default -1 = none) */
    char datum[32];                     /* e.g. "wgs84" */
    char projection[32];                /* e.g. "UTM" */
    char units[16];                     /* e.g. "m" */

    /* ---- SWW output configuration ---- */
    char name[256];                     /* domain name (used for SWW filename) */
    char output_dir[1024];              /* output directory for SWW files */

    /* ---- Flow algorithm ---- */
    hydro_int spatial_order;            /* 1 or 2 */
    hydro_int timestepping_method;      /* 1=Euler, 2=RK2, 3=RK3 */
    hydro_int low_froude_mode;          /* 0=off, 1=type1, 2=type2 */

    /* ---- Boundary metadata ---- */
    hydro_int* boundary_tags;           /* [boundary_length], tag for each boundary edge */
    hydro_int* boundary_edges;          /* [boundary_length], edge index for each boundary */
    hydro_int* boundary_tag_map;        /* [3*n_elements], per-edge tag (0=interior, >0=boundary) */

    /* Per-tag BC configuration (tag 0 = default reflective) */
#define HYDRO_MAX_BOUNDARY_TAGS 128
    hydro_int boundary_bc_type_tag[128]; /* BC type for each tag (index=tag) */
    double   boundary_stage_tag[128];    /* stage value for Dirichlet/time BC */
    double   boundary_xmom_tag[128];     /* x-momentum for Dirichlet BC */
    double   boundary_ymom_tag[128];     /* y-momentum for Dirichlet BC */

    /* ---- Kinematic viscosity geo-structure (built once, reused) ---- */
    hydro_int* geo_structure_indices;   /* [3*n_elements] column indices */
    double*    geo_structure_values;    /* [3*n_elements] geometric coeffs */

} hydro_domain_t;

/* ==========================================================================
 * Edge Data (used during flux computation)
 * ========================================================================== */

typedef struct {
    double ql[3], qr[3];                /* left/right conserved quantities [w, uh, vh] */
    double zl, zr;                      /* left/right bed elevation */
    double hle, hre;                    /* left/right edge heights */
    double h_left, h_right;             /* water depth adjusted to max bed */
    double hc, zc, hc_n, zc_n;          /* centroid values */
    double z_half;                      /* max(zl, zr, zwall) */
    double normal_x, normal_y;          /* edge normal components */
    double length;                      /* edge length */
    hydro_int n;                        /* neighbour triangle index (-1 = boundary) */
    hydro_int ki, ki2;                  /* edge flat index, 2*ki */
    int is_boundary;                    /* boundary flag */
    int is_riverwall;                   /* riverwall flag */
    hydro_int riverwall_index;          /* index into riverwall data */
} hydro_edge_data_t;

/* ==========================================================================
 * Core API
 * ========================================================================== */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Create a new computational domain.
 *
 * @param n_nodes     Number of unique vertices in the mesh
 * @param n_triangles Number of triangles in the mesh
 * @return            Newly allocated domain (must be freed with hydro_domain_destroy)
 */
hydro_domain_t* hydro_domain_create(hydro_int n_nodes, hydro_int n_triangles);

/**
 * Destroy a domain and free all associated memory.
 */
void hydro_domain_destroy(hydro_domain_t* domain);

/**
 * Set the mesh geometry and compute all derived quantities.
 *
 * @param domain         The domain
 * @param vertex_coords  Flat array of vertex coordinates [n_nodes * 2]
 * @param triangles      Triangle vertex indices [n_tri * 3]
 * @param boundary_tags  Tag for each boundary edge [boundary_length],
 *                       positive integer = boundary tag, 0 = interior
 * @param boundary_edges Flat array of edge indices [boundary_length]
 */
void hydro_domain_set_geometry(
    hydro_domain_t* domain,
    const double*   vertex_coords,
    const hydro_int* triangles,
    const hydro_int* boundary_tags,
    const hydro_int* boundary_edges);

/**
 * Set boundary tag lookup table from user-provided per-edge tags.
 * boundary_edges_in: flat edge indices [count]
 * boundary_tags_in: corresponding tags [count]
 * count: number of boundary edges
 */
void hydro_domain_set_boundary_tag_map(
    hydro_domain_t* domain,
    const hydro_int* boundary_edges_in,
    const hydro_int* boundary_tags_in,
    hydro_int       count);

/**
 * Set quantity values at triangle centroids.
 * Allocates storage if not already allocated, copies given values.
 *
 * @param domain   The domain
 * @param name     Quantity name: "elevation", "stage", "xmomentum",
 *                 "ymomentum", "friction"
 * @param values   Centroid values [n_triangles]
 */
void hydro_domain_set_quantity(
    hydro_domain_t* domain,
    const char*     name,
    const double*   values);

/**
 * Set a scalar parameter on the domain.
 *
 * @param domain  The domain
 * @param name    Parameter name
 * @param value   Parameter value
 */
void hydro_domain_set_parameter(
    hydro_domain_t* domain,
    const char*     name,
    double          value);

/**
 * Set the domain name (used for SWW output filename: {output_dir}/{name}.sww).
 * Set to empty string or NULL to disable SWW output.
 */
void hydro_domain_set_name(hydro_domain_t* domain, const char* name);

/**
 * Set the output directory for SWW files.
 * Defaults to "." if not set.
 */
void hydro_domain_set_output_dir(hydro_domain_t* domain, const char* dir);

/**
 * Get quantity centroid values (for post-simulation inspection).
 *
 * @param domain  The domain
 * @param name    Quantity name
 * @param values  Output array [n_triangles] (caller pre-allocated)
 */
void hydro_domain_get_quantity(
    const hydro_domain_t* domain,
    const char*           name,
    double*               values);

/**
 * Get the current simulation time.
 */
double hydro_domain_get_time(const hydro_domain_t* domain);

#ifdef __cplusplus
}
#endif

#endif /* HYDRO_DOMAIN_H */
