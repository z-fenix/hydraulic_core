/**
 * gpu_bindings.cpp — pybind11 bindings for hydro_core CUDA GPU module
 *
 * Exposes GPU kernel launchers and memory management to Python.
 * GPU arrays are managed by cupy on the Python side; this module
 * only launches pre-compiled CUDA kernels from hydro_cuda.cu.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cstdint>
#include <stdexcept>

namespace py = pybind11;

/* Forward declarations from hydro_cuda.cu */
extern "C" {
    int64_t cuda_compute_fluxes(
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

    void cuda_update_sw(
        int64_t number_of_elements,
        double timestep,
        double* centroid_values,
        double* explicit_update,
        double* semi_implicit_update);

    void cuda_fix_negative_cells(
        int64_t number_of_elements,
        int64_t* tri_full_flag,
        double* stage_centroid_values,
        double* bed_centroid_values,
        double* xmom_centroid_values,
        double* ymom_centroid_values,
        int64_t* num_negative_cells);

    void cuda_protect(
        double domain_minimum_allowed_height,
        int64_t number_of_elements,
        double* stage_centroid_values,
        double* bed_centroid_values,
        double* xmom_centroid_values,
        double* areas,
        double* stage_vertex_values);

    void cuda_manning_friction_flat(
        double g, double eps, int64_t N,
        double* w, double* zv,
        double* uh, double* vh,
        double* eta, double* xmom, double* ymom);

    void cuda_manning_friction_sloped(
        double g, double eps, int64_t N,
        double* x, double* w, double* zv,
        double* uh, double* vh,
        double* eta, double* xmom_update, double* ymom_update);
}

/* ==========================================================================
 * Helper: get raw GPU pointer from cupy array
 *
 * The Python side passes cupy arrays. We extract the device pointer
 * via the __cuda_array_interface__ or .data.ptr.
 * ========================================================================== */

static double* get_gpu_ptr(py::object arr) {
    try {
        auto ptr_val = arr.attr("data").attr("ptr")();
        return (double*)py::cast<intptr_t>(ptr_val);
    } catch (const std::exception&) {
        throw std::runtime_error("Cannot get GPU pointer from array. "
            "Ensure the object is a cupy array with __cuda_array_interface__.");
    }
}

static int64_t* get_gpu_ptr_int64(py::object arr) {
    try {
        auto ptr_val = arr.attr("data").attr("ptr")();
        return (int64_t*)py::cast<intptr_t>(ptr_val);
    } catch (const std::exception&) {
        throw std::runtime_error("Cannot get GPU int64 pointer from array.");
    }
}

/* ==========================================================================
 * GPU interface class — manages all GPU state
 * ========================================================================== */

class GPUInterface {
public:
    GPUInterface(py::object domain_obj) {
        /* Extract domain properties from Python domain object */
        n_elements_ = py::cast<int64_t>(domain_obj.attr("n_tri"));

        /* Extract scalar parameters */
        epsilon_ = py::cast<double>(domain_obj.attr("epsilon"));
        g_ = py::cast<double>(domain_obj.attr("g"));
        minimum_allowed_height_ = py::cast<double>(domain_obj.attr("minimum_allowed_height"));
        low_froude_ = py::cast<int64_t>(domain_obj.attr("low_froude"));
        extrapolate_velocity_second_order_ = py::cast<int64_t>(domain_obj.attr("extrapolate_velocity_second_order"));
        limiting_threshold_ = py::cast<double>(domain_obj.attr("limiting_threshold"));
        ncol_riverwall_ = py::cast<int64_t>(domain_obj.attr("ncol_riverwall_hydraulic_properties"));

        /* Extract beta parameters */
        beta_w_ = py::cast<double>(domain_obj.attr("beta_w"));
        beta_w_dry_ = py::cast<double>(domain_obj.attr("beta_w_dry"));
        beta_uh_ = py::cast<double>(domain_obj.attr("beta_uh"));
        beta_uh_dry_ = py::cast<double>(domain_obj.attr("beta_uh_dry"));
        beta_vh_ = py::cast<double>(domain_obj.attr("beta_vh"));
        beta_vh_dry_ = py::cast<double>(domain_obj.attr("beta_vh_dry"));

        /* Store reference to domain object for array access */
        domain_obj_ = domain_obj;
    }

    /**
     * Launch flux computation kernel.
     * All arrays are cupy arrays passed as Python objects.
     */
    double compute_fluxes(
        py::object timestep_k_array,
        py::object boundary_flux_sum_k_array,
        py::object max_speed,
        py::object stage_explicit_update,
        py::object xmom_explicit_update,
        py::object ymom_explicit_update,
        py::object stage_centroid_values,
        py::object height_centroid_values,
        py::object bed_centroid_values,
        py::object stage_edge_values,
        py::object xmom_edge_values,
        py::object ymom_edge_values,
        py::object bed_edge_values,
        py::object height_edge_values,
        py::object stage_boundary_values,
        py::object xmom_boundary_values,
        py::object ymom_boundary_values,
        py::object areas,
        py::object normals,
        py::object edgelengths,
        py::object radii,
        py::object tri_full_flag,
        py::object neighbours,
        py::object neighbour_edges,
        py::object edge_flux_type,
        py::object edge_river_wall_counter,
        py::object riverwall_elevation,
        py::object riverwall_rowIndex,
        py::object riverwall_hydraulic_properties,
        int64_t substep_count)
    {
        double timestep_out = 1.0e100;

        cuda_compute_fluxes(
            get_gpu_ptr(timestep_k_array),
            get_gpu_ptr(boundary_flux_sum_k_array),
            get_gpu_ptr(max_speed),
            get_gpu_ptr(stage_explicit_update),
            get_gpu_ptr(xmom_explicit_update),
            get_gpu_ptr(ymom_explicit_update),
            get_gpu_ptr(stage_centroid_values),
            get_gpu_ptr(height_centroid_values),
            get_gpu_ptr(bed_centroid_values),
            get_gpu_ptr(stage_edge_values),
            get_gpu_ptr(xmom_edge_values),
            get_gpu_ptr(ymom_edge_values),
            get_gpu_ptr(bed_edge_values),
            get_gpu_ptr(height_edge_values),
            get_gpu_ptr(stage_boundary_values),
            get_gpu_ptr(xmom_boundary_values),
            get_gpu_ptr(ymom_boundary_values),
            get_gpu_ptr(areas),
            get_gpu_ptr(normals),
            get_gpu_ptr(edgelengths),
            get_gpu_ptr(radii),
            get_gpu_ptr_int64(tri_full_flag),
            get_gpu_ptr_int64(neighbours),
            get_gpu_ptr_int64(neighbour_edges),
            get_gpu_ptr_int64(edge_flux_type),
            get_gpu_ptr_int64(edge_river_wall_counter),
            get_gpu_ptr(riverwall_elevation),
            get_gpu_ptr_int64(riverwall_rowIndex),
            get_gpu_ptr(riverwall_hydraulic_properties),
            n_elements_,
            substep_count,
            ncol_riverwall_,
            epsilon_,
            g_,
            low_froude_,
            limiting_threshold_,
            &timestep_out);

        return timestep_out;
    }

    /**
     * Launch all 4 extrapolation kernels.
     */
    void extrapolate_second_order(
        py::object stage_centroid_values,
        py::object xmom_centroid_values,
        py::object ymom_centroid_values,
        py::object height_centroid_values,
        py::object bed_centroid_values,
        py::object x_centroid_work,
        py::object y_centroid_work,
        py::object stage_edge_values,
        py::object xmom_edge_values,
        py::object ymom_edge_values,
        py::object height_edge_values,
        py::object bed_edge_values,
        py::object number_of_boundaries,
        py::object centroid_coordinates,
        py::object edge_coordinates,
        py::object surrogate_neighbours,
        py::object stage_vertex_values,
        py::object height_vertex_values,
        py::object xmom_vertex_values,
        py::object ymom_vertex_values,
        py::object bed_vertex_values)
    {
        /* Loop 1: depth + velocity conversion */
        cuda_extrapolate_loop1(
            get_gpu_ptr(stage_centroid_values),
            get_gpu_ptr(xmom_centroid_values),
            get_gpu_ptr(ymom_centroid_values),
            get_gpu_ptr(height_centroid_values),
            get_gpu_ptr(bed_centroid_values),
            get_gpu_ptr(x_centroid_work),
            get_gpu_ptr(y_centroid_work),
            minimum_allowed_height_,
            n_elements_,
            extrapolate_velocity_second_order_);

        /* Loop 2: 2nd order extrapolation + TVD limiting */
        cuda_extrapolate_loop2(
            get_gpu_ptr(stage_edge_values),
            get_gpu_ptr(xmom_edge_values),
            get_gpu_ptr(ymom_edge_values),
            get_gpu_ptr(height_edge_values),
            get_gpu_ptr(bed_edge_values),
            get_gpu_ptr(stage_centroid_values),
            get_gpu_ptr(xmom_centroid_values),
            get_gpu_ptr(ymom_centroid_values),
            get_gpu_ptr(height_centroid_values),
            get_gpu_ptr(bed_centroid_values),
            get_gpu_ptr(x_centroid_work),
            get_gpu_ptr(y_centroid_work),
            get_gpu_ptr_int64(number_of_boundaries),
            get_gpu_ptr(centroid_coordinates),
            get_gpu_ptr(edge_coordinates),
            get_gpu_ptr_int64(surrogate_neighbours),
            beta_w_dry_, beta_w_,
            beta_uh_dry_, beta_uh_,
            beta_vh_dry_, beta_vh_,
            minimum_allowed_height_,
            n_elements_,
            extrapolate_velocity_second_order_);

        /* Loop 3: velocity to momentum */
        cuda_extrapolate_loop3(
            get_gpu_ptr(xmom_centroid_values),
            get_gpu_ptr(ymom_centroid_values),
            get_gpu_ptr(x_centroid_work),
            get_gpu_ptr(y_centroid_work),
            extrapolate_velocity_second_order_,
            n_elements_);

        /* Loop 4: edge to vertex */
        cuda_extrapolate_loop4(
            get_gpu_ptr(stage_edge_values),
            get_gpu_ptr(xmom_edge_values),
            get_gpu_ptr(ymom_edge_values),
            get_gpu_ptr(height_edge_values),
            get_gpu_ptr(bed_edge_values),
            get_gpu_ptr(stage_vertex_values),
            get_gpu_ptr(height_vertex_values),
            get_gpu_ptr(xmom_vertex_values),
            get_gpu_ptr(ymom_vertex_values),
            get_gpu_ptr(bed_vertex_values),
            n_elements_);
    }

    /**
     * Update conserved quantities + fix negative cells.
     */
    int64_t update_conserved(
        double timestep,
        py::object stage_centroid_values,
        py::object xmom_centroid_values,
        py::object ymom_centroid_values,
        py::object stage_explicit_update,
        py::object xmom_explicit_update,
        py::object ymom_explicit_update,
        py::object stage_semi_implicit_update,
        py::object xmom_semi_implicit_update,
        py::object ymom_semi_implicit_update,
        py::object tri_full_flag,
        py::object bed_centroid_values,
        py::object num_negative_cells)
    {
        /* Update stage */
        cuda_update_sw(
            n_elements_, timestep,
            get_gpu_ptr(stage_centroid_values),
            get_gpu_ptr(stage_explicit_update),
            get_gpu_ptr(stage_semi_implicit_update));

        /* Update xmom */
        cuda_update_sw(
            n_elements_, timestep,
            get_gpu_ptr(xmom_centroid_values),
            get_gpu_ptr(xmom_explicit_update),
            get_gpu_ptr(xmom_semi_implicit_update));

        /* Update ymom */
        cuda_update_sw(
            n_elements_, timestep,
            get_gpu_ptr(ymom_centroid_values),
            get_gpu_ptr(ymom_explicit_update),
            get_gpu_ptr(ymom_semi_implicit_update));

        /* Fix negative cells */
        cuda_fix_negative_cells(
            n_elements_,
            get_gpu_ptr_int64(tri_full_flag),
            get_gpu_ptr(stage_centroid_values),
            get_gpu_ptr(bed_centroid_values),
            get_gpu_ptr(xmom_centroid_values),
            get_gpu_ptr(ymom_centroid_values),
            get_gpu_ptr_int64(num_negative_cells));

        return (int64_t)0;  /* num_negative_cells is on GPU; caller reads via cupy */
    }

    /**
     * Protect against infinitesimal heights.
     */
    void protect(
        py::object stage_centroid_values,
        py::object bed_centroid_values,
        py::object xmom_centroid_values,
        py::object areas,
        py::object stage_vertex_values)
    {
        cuda_protect(
            minimum_allowed_height_,
            n_elements_,
            get_gpu_ptr(stage_centroid_values),
            get_gpu_ptr(bed_centroid_values),
            get_gpu_ptr(xmom_centroid_values),
            get_gpu_ptr(areas),
            get_gpu_ptr(stage_vertex_values));
    }

    /**
     * Manning friction (flat bed).
     */
    void manning_friction_flat(
        py::object stage_centroid_values,
        py::object bed_vertex_values,
        py::object xmom_centroid_values,
        py::object ymom_centroid_values,
        py::object friction_centroid_values,
        py::object xmom_semi_implicit_update,
        py::object ymom_semi_implicit_update)
    {
        cuda_manning_friction_flat(
            g_, epsilon_, n_elements_,
            get_gpu_ptr(stage_centroid_values),
            get_gpu_ptr(bed_vertex_values),
            get_gpu_ptr(xmom_centroid_values),
            get_gpu_ptr(ymom_centroid_values),
            get_gpu_ptr(friction_centroid_values),
            get_gpu_ptr(xmom_semi_implicit_update),
            get_gpu_ptr(ymom_semi_implicit_update));
    }

private:
    int64_t n_elements_;
    double epsilon_;
    double g_;
    double minimum_allowed_height_;
    int64_t low_froude_;
    int64_t extrapolate_velocity_second_order_;
    double limiting_threshold_;
    int64_t ncol_riverwall_;
    double beta_w_;
    double beta_w_dry_;
    double beta_uh_;
    double beta_uh_dry_;
    double beta_vh_;
    double beta_vh_dry_;

    py::object domain_obj_;
};

/* ==========================================================================
 * Module definition
 * ========================================================================== */

PYBIND11_MODULE(_gpu, m) {
    m.doc() = "Hydro Core — CUDA GPU acceleration module";

    py::class_<GPUInterface>(m, "GPUInterface")
        .def(py::init<py::object>())
        .def("compute_fluxes", &GPUInterface::compute_fluxes,
             py::arg("timestep_k_array"),
             py::arg("boundary_flux_sum_k_array"),
             py::arg("max_speed"),
             py::arg("stage_explicit_update"),
             py::arg("xmom_explicit_update"),
             py::arg("ymom_explicit_update"),
             py::arg("stage_centroid_values"),
             py::arg("height_centroid_values"),
             py::arg("bed_centroid_values"),
             py::arg("stage_edge_values"),
             py::arg("xmom_edge_values"),
             py::arg("ymom_edge_values"),
             py::arg("bed_edge_values"),
             py::arg("height_edge_values"),
             py::arg("stage_boundary_values"),
             py::arg("xmom_boundary_values"),
             py::arg("ymom_boundary_values"),
             py::arg("areas"),
             py::arg("normals"),
             py::arg("edgelengths"),
             py::arg("radii"),
             py::arg("tri_full_flag"),
             py::arg("neighbours"),
             py::arg("neighbour_edges"),
             py::arg("edge_flux_type"),
             py::arg("edge_river_wall_counter"),
             py::arg("riverwall_elevation"),
             py::arg("riverwall_rowIndex"),
             py::arg("riverwall_hydraulic_properties"),
             py::arg("substep_count") = 0,
             "Launch flux computation kernel on GPU")

        .def("extrapolate_second_order", &GPUInterface::extrapolate_second_order,
             py::arg("stage_centroid_values"),
             py::arg("xmom_centroid_values"),
             py::arg("ymom_centroid_values"),
             py::arg("height_centroid_values"),
             py::arg("bed_centroid_values"),
             py::arg("x_centroid_work"),
             py::arg("y_centroid_work"),
             py::arg("stage_edge_values"),
             py::arg("xmom_edge_values"),
             py::arg("ymom_edge_values"),
             py::arg("height_edge_values"),
             py::arg("bed_edge_values"),
             py::arg("number_of_boundaries"),
             py::arg("centroid_coordinates"),
             py::arg("edge_coordinates"),
             py::arg("surrogate_neighbours"),
             py::arg("stage_vertex_values"),
             py::arg("height_vertex_values"),
             py::arg("xmom_vertex_values"),
             py::arg("ymom_vertex_values"),
             py::arg("bed_vertex_values"),
             "Launch all 4 extrapolation kernels")

        .def("update_conserved", &GPUInterface::update_conserved,
             py::arg("timestep"),
             py::arg("stage_centroid_values"),
             py::arg("xmom_centroid_values"),
             py::arg("ymom_centroid_values"),
             py::arg("stage_explicit_update"),
             py::arg("xmom_explicit_update"),
             py::arg("ymom_explicit_update"),
             py::arg("stage_semi_implicit_update"),
             py::arg("xmom_semi_implicit_update"),
             py::arg("ymom_semi_implicit_update"),
             py::arg("tri_full_flag"),
             py::arg("bed_centroid_values"),
             py::arg("num_negative_cells"),
             "Update conserved quantities + fix negative cells")

        .def("protect", &GPUInterface::protect,
             py::arg("stage_centroid_values"),
             py::arg("bed_centroid_values"),
             py::arg("xmom_centroid_values"),
             py::arg("areas"),
             py::arg("stage_vertex_values"),
             "Protect against infinitesimal heights")

        .def("manning_friction_flat", &GPUInterface::manning_friction_flat,
             py::arg("stage_centroid_values"),
             py::arg("bed_vertex_values"),
             py::arg("xmom_centroid_values"),
             py::arg("ymom_centroid_values"),
             py::arg("friction_centroid_values"),
             py::arg("xmom_semi_implicit_update"),
             py::arg("ymom_semi_implicit_update"),
             "Manning friction forcing (flat bed)");
}
