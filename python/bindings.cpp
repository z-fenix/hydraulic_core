/**
 * bindings.cpp — pybind11 Python bindings for hydro_core
 *
 * Replaces the ctypes-based hydro/_core.py with a native extension.
 * Exposes hydro_domain_t as a Python Domain class with numpy integration.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "hydro/hydro.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace py = pybind11;

/* ==========================================================================
 * Domain class — wraps hydro_domain_t* with RAII
 * ========================================================================== */

class Domain {
public:
    /* --- Construction --- */
    Domain(py::array_t<double> vertices,
           py::array_t<int64_t> triangles,
           py::array_t<int64_t> boundary_tags = py::array_t<int64_t>{},
           py::array_t<int64_t> boundary_edges = py::array_t<int64_t>{})
    {
        auto v = vertices.unchecked<2>();
        auto t = triangles.unchecked<2>();
        hydro_int n_uniq = (hydro_int)v.shape(0);
        n_tri_ = (hydro_int)t.shape(0);
        n_edges_ = 3 * n_tri_;

        handle_ = hydro_domain_create(n_uniq, n_tri_);
        if (!handle_)
            throw std::runtime_error("hydro_domain_create failed");

        /* Set geometry */
        hydro_domain_set_geometry(
            handle_,
            vertices.data(),
            triangles.data(),
            boundary_tags.size() ? boundary_tags.data() : nullptr,
            boundary_edges.size() ? boundary_edges.data() : nullptr);

        hydro_mesh_build_neighbour_structure(handle_);

        /* Set boundary tag map if provided */
        if (boundary_tags.size() > 0 && boundary_edges.size() > 0) {
            hydro_domain_set_boundary_tag_map(
                handle_,
                boundary_edges.data(),
                boundary_tags.data(),
                (hydro_int)boundary_edges.size());
        }

        hydro_mesh_build_boundary_structure(handle_);

        /* Default parameters */
        hydro_domain_set_parameter(handle_, "CFL", 1.0);
        hydro_domain_set_parameter(handle_, "spatial_order", 1);
        hydro_domain_set_parameter(handle_, "timestepping_method", 1);
    }

    ~Domain() {
        if (handle_) {
            hydro_domain_destroy(handle_);
            handle_ = nullptr;
        }
    }

    /* Non-copyable */
    Domain(const Domain&) = delete;
    Domain& operator=(const Domain&) = delete;
    Domain(Domain&& other) noexcept : handle_(other.handle_), n_tri_(other.n_tri_), n_edges_(other.n_edges_) {
        other.handle_ = nullptr;
    }

    /* --- Quantities --- */
    void set_quantity(const std::string& name, py::array_t<double> values) {
        auto arr = values.unchecked<1>();
        if ((hydro_int)arr.shape(0) != n_tri_)
            throw std::runtime_error("set_quantity: expected " +
                std::to_string(n_tri_) + " values, got " +
                std::to_string(arr.shape(0)));
        hydro_domain_set_quantity(handle_, name.c_str(), values.data());
    }

    py::array_t<double> get_quantity(const std::string& name) {
        auto result = py::array_t<double>(n_tri_);
        hydro_domain_get_quantity(handle_, name.c_str(), result.mutable_data());
        return result;
    }

    /* --- Convenience setters/getters --- */
    void set_elevation(py::array_t<double> v) { set_quantity("elevation", v); }
    void set_stage(py::array_t<double> v)      { set_quantity("stage", v); }
    void set_xmomentum(py::array_t<double> v)  { set_quantity("xmomentum", v); }
    void set_ymomentum(py::array_t<double> v)  { set_quantity("ymomentum", v); }
    void set_friction(py::array_t<double> v)   { set_quantity("friction", v); }

    py::array_t<double> get_stage()     { return get_quantity("stage"); }
    py::array_t<double> get_elevation() { return get_quantity("elevation"); }
    py::array_t<double> get_xmomentum() { return get_quantity("xmomentum"); }
    py::array_t<double> get_ymomentum() { return get_quantity("ymomentum"); }

    py::array_t<double> get_height() {
        auto s = get_stage();
        auto e = get_elevation();
        auto h = py::array_t<double>(n_tri_);
        for (py::ssize_t i = 0; i < n_tri_; i++)
            h.mutable_data()[i] = s.data()[i] - e.data()[i];
        return h;
    }

    /* --- Parameters --- */
    void set_parameter(const std::string& name, double value) {
        hydro_domain_set_parameter(handle_, name.c_str(), value);
    }

    /* --- Naming / output --- */
    void set_name(const std::string& name) {
        hydro_domain_set_name(handle_, name.c_str());
    }

    void set_output_dir(const std::string& dir) {
        hydro_domain_set_output_dir(handle_, dir.c_str());
    }

    /* --- Boundary conditions --- */
    void set_boundary(int tag, int bc_type,
                      double stage = 0.0, double discharge = 0.0) {
        hydro_bc_params_t params;
        params.stage = stage;
        params.wh0 = discharge;
        hydro_domain_set_boundary(handle_, (hydro_int)tag,
                                   (hydro_bc_type_t)bc_type, &params);
    }

    /**
     * Set time-series boundary data for a tag.
     *
     * @param tag           Boundary tag (e.g. 1 for left/inflow)
     * @param times         1-D array of time values (seconds)
     * @param q_values      1-D array of discharge values (m^3/s)
     * @param default_stage Fallback stage when Q is out of CSV range
     */
    void set_time_series_boundary(
        int tag,
        py::array_t<double> times,
        py::array_t<double> q_values,
        double default_stage = 0.1)
    {
        auto t_arr = times.unchecked<1>();
        auto q_arr = q_values.unchecked<1>();
        hydro_int n = (hydro_int)t_arr.shape(0);
        if (n != (hydro_int)q_arr.shape(0))
            throw std::runtime_error(
                "set_time_series_boundary: times and q_values must have same length");
        if (n <= 0)
            throw std::runtime_error(
                "set_time_series_boundary: need at least 1 data point");

        hydro_boundary_set_time_series(
            handle_, (hydro_int)tag,
            times.data(), q_values.data(),
            (int)n, default_stage);
    }

    /**
     * Update time-series boundary values at current simulation time.
     * The effective channel width is auto-derived from boundary edge geometry.
     *
     * @param tag           Boundary tag
     * @param current_time  Current simulation time (pass domain.get_time())
     */
    void update_time_series_boundary(
        int tag,
        double current_time)
    {
        hydro_boundary_update_time_series(
            handle_, (hydro_int)tag, current_time);
    }

    /* --- Evolution --- */
    void evolve(double finaltime, double yieldstep = 1.0) {
        hydro_quantity_update_derived(handle_);
        int ret = hydro_domain_evolve(handle_, finaltime, yieldstep);
        if (ret != 0)
            throw std::runtime_error("hydro_domain_evolve failed (code " +
                std::to_string(ret) + ")");
    }

    double get_time() const {
        return hydro_domain_get_time(handle_);
    }

    void close() { /* no-op: destructor handles cleanup */ }

    std::string repr() const {
        return "HydroDomain(n_tri=" + std::to_string(n_tri_) + ")";
    }

    /* --- Accessors --- */
    long long n_tri() const { return (long long)n_tri_; }

private:
    hydro_domain_t* handle_ = nullptr;
    hydro_int n_tri_ = 0;
    hydro_int n_edges_ = 0;
};

/* ==========================================================================
 * Module definition
 * ========================================================================== */

PYBIND11_MODULE(_core, m) {
    m.doc() = "Hydro Core — shallow water equation solver (native extension)";

    /* ---- Constants ---- */
    m.attr("G")     = 9.8;
    m.attr("EPSILON") = 1.0e-6;
    m.attr("MINIMUM_ALLOWED_HEIGHT") = 1.0e-05;
    m.attr("MINIMUM_STORABLE_HEIGHT") = 1.0e-03;
    m.attr("MAXIMUM_ALLOWED_SPEED")  = 1000.0;
    m.attr("DEFAULT_MANNING") = 0.03;
    m.attr("EVOLVE_MAX_TIMESTEP") = 1000.0;
    m.attr("EVOLVE_MIN_TIMESTEP") = 1.0e-6;

    m.attr("EULER") = 1;
    m.attr("RK2")   = 2;
    m.attr("RK3")   = 3;

    /* ---- Boundary condition types ---- */
    m.attr("HYDRO_BC_NONE")          = 0;
    m.attr("HYDRO_BC_REFLECTIVE")    = 1;
    m.attr("HYDRO_BC_DIRICHLET")     = 2;
    m.attr("HYDRO_BC_TRANSMISSIVE")  = 3;
    m.attr("HYDRO_BC_TIME")          = 4;
    m.attr("HYDRO_BC_DIRICHLET_DISCHARGE") = 5;
    m.attr("HYDRO_BC_TRANSMISSIVE_STAGE")   = 6;
    m.attr("HYDRO_BC_TIME_SERIES")    = 7;

    /* ---- Domain class ---- */
    py::class_<Domain>(m, "Domain")
        .def(py::init<py::array_t<double>, py::array_t<int64_t>,
                      py::array_t<int64_t>, py::array_t<int64_t>>(),
             py::arg("vertices"),
             py::arg("triangles"),
             py::arg("boundary_tags") = py::array_t<int64_t>{},
             py::arg("boundary_edges") = py::array_t<int64_t>{},
             R"(Shallow-water simulation domain.

Parameters
----------
vertices : ndarray (N, 2) float64
    Unique vertex coordinates in metres.
triangles : ndarray (M, 3) int64
    Triangle vertex indices (CCW ordering).
boundary_tags : ndarray (B,) int64, optional
    Per-edge boundary tag: positive = boundary, 0 = interior.
boundary_edges : ndarray (B,) int64, optional
    Flat edge indices of boundary edges.
)")

        /* Quantities */
        .def("set_quantity", &Domain::set_quantity,
             py::arg("name"), py::arg("values"))
        .def("get_quantity", &Domain::get_quantity,
             py::arg("name"))

        .def("set_elevation", &Domain::set_elevation)
        .def("set_stage",     &Domain::set_stage)
        .def("set_xmomentum", &Domain::set_xmomentum)
        .def("set_ymomentum", &Domain::set_ymomentum)
        .def("set_friction",  &Domain::set_friction)
        .def("get_stage",     &Domain::get_stage)
        .def("get_elevation", &Domain::get_elevation)
        .def("get_xmomentum", &Domain::get_xmomentum)
        .def("get_ymomentum", &Domain::get_ymomentum)
        .def("get_height",    &Domain::get_height)

        /* Parameters */
        .def("set_parameter", &Domain::set_parameter,
             py::arg("name"), py::arg("value"))

        /* Naming / output */
        .def("set_name",       &Domain::set_name)
        .def("set_output_dir", &Domain::set_output_dir)

        /* Boundaries */
        .def("set_boundary", &Domain::set_boundary,
             py::arg("tag"), py::arg("bc_type"),
             py::arg("stage") = 0.0,
             py::arg("discharge") = 0.0)
        .def("set_time_series_boundary", &Domain::set_time_series_boundary,
             py::arg("tag"), py::arg("times"), py::arg("q_values"),
             py::arg("default_stage") = 0.1,
             "Set time-series inflow boundary data for a tag.")
        .def("update_time_series_boundary", &Domain::update_time_series_boundary,
             py::arg("tag"), py::arg("current_time"),
             "Update time-series boundary values at current simulation time.")

        /* Evolution */
        .def("evolve", &Domain::evolve,
             py::arg("finaltime"), py::arg("yieldstep") = 1.0)
        .def("get_time", &Domain::get_time)

        /* Accessors */
        .def_property_readonly("n_tri", &Domain::n_tri)
        .def("close", &Domain::close)
        .def("__repr__", &Domain::repr);
}
