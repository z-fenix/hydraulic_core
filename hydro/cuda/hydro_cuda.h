/**
 * hydro_cuda.h — C interface for hydro_core CUDA kernels
 *
 * These functions are implemented in hydro/cuda/hydro_cuda.cu
 * and called from pybind11 bindings or C code.
 */

#ifndef HYDRO_CUDA_H
#define HYDRO_CUDA_H

#include "hydro/types.h"
#include "hydro/domain.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Compute fluxes on GPU.
 *
 * All arrays must already be on the GPU (cupy-managed).
 * Returns CUDA error code (0 = success).
 */
hydro_int cuda_compute_fluxes(
    double* timestep_k_array,
    double* boundary_flux_sum_k_array,
    double* max_speed,
    double* stage_explicit_update,
    double* xmom_explicit_update,
    double* ymom_explicit_update,
    double* stage_centroid_values,
    double* height_centroid_values,
    double* bed_centroid_values,
    double* stage_edge_values,
    double* xmom_edge_values,
    double* ymom_edge_values,
    double* bed_edge_values,
    double* height_edge_values,
    double* stage_boundary_values,
    double* xmom_boundary_values,
    double* ymom_boundary_values,
    double* areas,
    double* normals,
    double* edgelengths,
    double* radii,
    int64_t* tri_full_flag,
    int64_t* neighbours,
    int64_t* neighbour_edges,
    int64_t* edge_flux_type,
    int64_t* edge_river_wall_counter,
    double* riverwall_elevation,
    int64_t* riverwall_rowIndex,
    double* riverwall_hydraulic_properties,
    int64_t number_of_elements,
    int64_t substep_count,
    int64_t ncol_riverwall,
    double epsilon,
    double g,
    int64_t low_froude,
    double limiting_threshold,
    double* timestep_out);

/**
 * Extrapolation: 4 sequential kernels.
 */
void cuda_extrapolate_loop1(
    double* stage_centroid_values,
    double* xmom_centroid_values,
    double* ymom_centroid_values,
    double* height_centroid_values,
    double* bed_centroid_values,
    double* x_centroid_work,
    double* y_centroid_work,
    double minimum_allowed_height,
    int64_t number_of_elements,
    int64_t extrapolate_velocity_second_order);

void cuda_extrapolate_loop2(
    double* stage_edge_values,
    double* xmom_edge_values,
    double* ymom_edge_values,
    double* height_edge_values,
    double* bed_edge_values,
    double* stage_centroid_values,
    double* xmom_centroid_values,
    double* ymom_centroid_values,
    double* height_centroid_values,
    double* bed_centroid_values,
    double* x_centroid_work,
    double* y_centroid_work,
    int64_t* number_of_boundaries,
    double* centroid_coordinates,
    double* edge_coordinates,
    int64_t* surrogate_neighbours,
    double beta_w_dry, double beta_w,
    double beta_uh_dry, double beta_uh,
    double beta_vh_dry, double beta_vh,
    double minimum_allowed_height,
    int64_t number_of_elements,
    int64_t extrapolate_velocity_second_order);

void cuda_extrapolate_loop3(
    double* xmom_centroid_values,
    double* ymom_centroid_values,
    double* x_centroid_work,
    double* y_centroid_work,
    int64_t extrapolate_velocity_second_order,
    int64_t number_of_elements);

void cuda_extrapolate_loop4(
    double* stage_edge_values,
    double* xmom_edge_values,
    double* ymom_edge_values,
    double* height_edge_values,
    double* bed_edge_values,
    double* stage_vertex_values,
    double* height_vertex_values,
    double* xmom_vertex_values,
    double* ymom_vertex_values,
    double* bed_vertex_values,
    int64_t number_of_elements);

/**
 * Update conserved quantities: Q_new = (Q_old + dt*E) / (1 - dt*SI)
 */
void cuda_update_sw(
    int64_t number_of_elements,
    double timestep,
    double* centroid_values,
    double* explicit_update,
    double* semi_implicit_update);

/**
 * Fix negative depth cells: set stage=bed, momentum=0.
 */
void cuda_fix_negative_cells(
    int64_t number_of_elements,
    int64_t* tri_full_flag,
    double* stage_centroid_values,
    double* bed_centroid_values,
    double* xmom_centroid_values,
    double* ymom_centroid_values,
    int64_t* num_negative_cells);

/**
 * Protect against infinitesimal and negative heights.
 */
void cuda_protect(
    double domain_minimum_allowed_height,
    int64_t number_of_elements,
    double* stage_centroid_values,
    double* bed_centroid_values,
    double* xmom_centroid_values,
    double* areas,
    double* stage_vertex_values);

/**
 * Manning friction forcing (flat bed).
 */
void cuda_manning_friction_flat(
    double g, double eps, int64_t N,
    double* w, double* zv,
    double* uh, double* vh,
    double* eta, double* xmom, double* ymom);

/**
 * Manning friction forcing (sloped bed).
 */
void cuda_manning_friction_sloped(
    double g, double eps, int64_t N,
    double* x, double* w, double* zv,
    double* uh, double* vh,
    double* eta, double* xmom_update, double* ymom_update);

#ifdef __cplusplus
}
#endif

#endif /* HYDRO_CUDA_H */
