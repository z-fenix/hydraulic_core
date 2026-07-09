"""
gpu.py — GPU orchestration layer for hydro_core

Provides GPU_interface class that manages:
1. Cupy GPU array allocation (copies of all hydro_domain_t arrays)
2. Kernel launch coordination
3. CPU ↔ GPU data synchronization

Requires: cupy, hydro._gpu (pre-compiled pybind11 module)
"""

import numpy as np

try:
    import cupy as cp
except ImportError:
    cp = None

try:
    from hydro._gpu import GPUInterface as _GPUInterface
except ImportError:
    _GPUInterface = None


class GPUError(RuntimeError):
    """Raised when GPU operations fail."""
    pass


def _require_cupy():
    """Ensure cupy is available."""
    if cp is None:
        raise GPUError(
            "cupy is not installed. Install it with: pip install cupy-cuda12x "
            "(or the appropriate version for your CUDA installation)"
        )


def _require_gpu_module():
    """Ensure the _gpu pybind11 module is available."""
    if _GPUInterface is None:
        raise GPUError(
            "hydro._gpu module not found. "
            "Build with: cmake -DHYDRO_BUILD_GPU=ON ..."
        )


class GPU_interface:
    """
    GPU orchestration layer for hydro_core.

    Manages GPU memory allocation, kernel launches, and CPU↔GPU data transfers.

    Usage:
        gpu = GPU_interface(domain)
        gpu.allocate()

        # Inside the evolve loop:
        gpu.transfer_to_gpu()
        gpu.extrapolate_second_order()
        gpu.compute_fluxes(substep_count=0)
        gpu.manning_friction_flat()
        gpu.update_conserved(timestep)
        gpu.protect()
        gpu.transfer_to_cpu()
    """

    def __init__(self, domain):
        """
        Initialize GPU interface for a hydro_core domain.

        Parameters
        ----------
        domain : hydro.Domain
            The CPU domain object. All arrays are accessed via numpy.
        """
        _require_cupy()
        _require_gpu_module()

        self.domain = domain
        self.n_elements = domain.n_tri
        self.n_edges = 3 * self.n_elements
        self.gpu_arrays_allocated = False

        # Extract domain parameters
        self._extract_parameters(domain)

        # Create GPU interface wrapper
        self._gpu = _GPUInterface(domain)

    def _extract_parameters(self, domain):
        """Extract scalar parameters from the domain object."""
        # Physical and numerical parameters
        self.epsilon = getattr(domain, 'epsilon', 1.0e-6)
        self.g = getattr(domain, 'g', 9.8)
        self.minimum_allowed_height = getattr(domain, 'minimum_allowed_height', 1.0e-5)
        self.low_froude = getattr(domain, 'low_froude', 0)
        self.extrapolate_velocity_second_order = getattr(
            domain, 'extrapolate_velocity_second_order', 0)
        self.limiting_threshold = getattr(domain, 'limiting_threshold', 10.0 * self.minimum_allowed_height)
        self.ncol_riverwall_hydraulic_properties = getattr(
            domain, 'ncol_riverwall_hydraulic_properties', 5)

        # Beta parameters
        self.beta_w = getattr(domain, 'beta_w', 1.0)
        self.beta_w_dry = getattr(domain, 'beta_w_dry', 0.0)
        self.beta_uh = getattr(domain, 'beta_uh', 1.0)
        self.beta_uh_dry = getattr(domain, 'beta_uh_dry', 0.0)
        self.beta_vh = getattr(domain, 'beta_vh', 1.0)
        self.beta_vh_dry = getattr(domain, 'beta_vh_dry', 0.0)

    def allocate(self):
        """
        Allocate all GPU arrays.

        Creates cupy device copies of every array in the domain.
        Must be called before any kernel launches.
        """
        _require_cupy()

        n = self.n_elements
        n_edges = self.n_edges
        n_bdry = getattr(self.domain, 'boundary_length', 0)

        # Transient arrays (updated each timestep)
        # Timestep reduction arrays
        self.gpu_timestep_array = cp.zeros(n, dtype=cp.float64)
        self.gpu_local_boundary_flux_sum = cp.zeros(n, dtype=cp.float64)

        # Max speed (InOut)
        self.gpu_max_speed = cp.array(getattr(self.domain, 'max_speed', np.zeros(n)))

        # Explicit updates (InOut)
        self.gpu_stage_explicit_update = cp.array(self.domain.stage_explicit_update)
        self.gpu_xmom_explicit_update = cp.array(self.domain.xmom_explicit_update)
        self.gpu_ymom_explicit_update = cp.array(self.domain.ymom_explicit_update)

        # Semi-implicit updates
        self.gpu_stage_semi_implicit_update = cp.array(self.domain.stage_semi_implicit_update)
        self.gpu_xmom_semi_implicit_update = cp.array(self.domain.xmom_semi_implicit_update)
        self.gpu_ymom_semi_implicit_update = cp.array(self.domain.ymom_semi_implicit_update)

        # Centroid values
        self.gpu_stage_centroid_values = cp.array(self.domain.stage_centroid_values)
        self.gpu_xmom_centroid_values = cp.array(self.domain.xmom_centroid_values)
        self.gpu_ymom_centroid_values = cp.array(self.domain.ymom_centroid_values)
        self.gpu_height_centroid_values = cp.array(self.domain.height_centroid_values)
        self.gpu_bed_centroid_values = cp.array(self.domain.bed_centroid_values)
        self.gpu_friction_centroid_values = cp.array(self.domain.friction_centroid_values)

        # Edge values
        self.gpu_stage_edge_values = cp.array(self.domain.stage_edge_values)
        self.gpu_xmom_edge_values = cp.array(self.domain.xmom_edge_values)
        self.gpu_ymom_edge_values = cp.array(self.domain.ymom_edge_values)
        self.gpu_bed_edge_values = cp.array(self.domain.bed_edge_values)
        self.gpu_height_edge_values = cp.array(self.domain.height_edge_values)

        # Boundary values
        self.gpu_stage_boundary_values = cp.zeros(n_bdry, dtype=cp.float64)
        self.gpu_xmom_boundary_values = cp.zeros(n_bdry, dtype=cp.float64)
        self.gpu_ymom_boundary_values = cp.zeros(n_bdry, dtype=cp.float64)

        # Vertex values
        self.gpu_stage_vertex_values = cp.zeros(n_edges, dtype=cp.float64)
        self.gpu_height_vertex_values = cp.zeros(n_edges, dtype=cp.float64)
        self.gpu_xmom_vertex_values = cp.zeros(n_edges, dtype=cp.float64)
        self.gpu_ymom_vertex_values = cp.zeros(n_edges, dtype=cp.float64)
        self.gpu_bed_vertex_values = cp.zeros(n_edges, dtype=cp.float64)

        # Work arrays
        self.gpu_x_centroid_work = cp.zeros(n, dtype=cp.float64)
        self.gpu_y_centroid_work = cp.zeros(n, dtype=cp.float64)

        # Negative cell counter
        self.gpu_num_negative_cells = cp.array([0], dtype=cp.int64)

        # Static mesh arrays (don't change during simulation)
        self.gpu_areas = cp.array(self.domain.areas)
        self.gpu_normals = cp.array(self.domain.normals)
        self.gpu_edgelengths = cp.array(self.domain.edgelengths)
        self.gpu_radii = cp.array(self.domain.radii)
        self.gpu_tri_full_flag = cp.array(self.domain.tri_full_flag, dtype=cp.int64)
        self.gpu_neighbours = cp.array(self.domain.neighbours, dtype=cp.int64)
        self.gpu_neighbour_edges = cp.array(self.domain.neighbour_edges, dtype=cp.int64)
        self.gpu_edge_flux_type = cp.array(self.domain.edge_flux_type, dtype=cp.int64)
        self.gpu_edge_river_wall_counter = cp.array(
            self.domain.edge_river_wall_counter, dtype=cp.int64)
        self.gpu_centroid_coordinates = cp.array(self.domain.centroid_coordinates)
        self.gpu_edge_coordinates = cp.array(self.domain.edge_coordinates)
        self.gpu_surrogate_neighbours = cp.array(
            self.domain.surrogate_neighbours, dtype=cp.int64)
        self.gpu_number_of_boundaries = cp.array(
            self.domain.number_of_boundaries, dtype=cp.int64)

        # Riverwall data
        if hasattr(self.domain, 'riverwall_elevation'):
            self.gpu_riverwall_elevation = cp.array(self.domain.riverwall_elevation)
        else:
            self.gpu_riverwall_elevation = cp.zeros(0, dtype=cp.float64)

        if hasattr(self.domain, 'riverwall_rowIndex'):
            self.gpu_riverwall_rowIndex = cp.array(
                self.domain.riverwall_rowIndex, dtype=cp.int64)
        else:
            self.gpu_riverwall_rowIndex = cp.zeros(0, dtype=cp.int64)

        if hasattr(self.domain, 'riverwall_hydraulic_properties'):
            self.gpu_riverwall_hydraulic_properties = cp.array(
                self.domain.riverwall_hydraulic_properties)
        else:
            self.gpu_riverwall_hydraulic_properties = cp.zeros(0, dtype=cp.float64)

        # Vertex coordinates (for sloped bed Manning)
        self.gpu_vertex_coordinates = cp.array(self.domain.vertex_coordinates)

        self.gpu_arrays_allocated = True

    # ------------------------------------------------------------------
    # CPU → GPU transfer methods
    # ------------------------------------------------------------------

    def transfer_to_gpu(self):
        """Transfer all transient data from CPU to GPU before kernel launches."""
        n = self.n_elements

        # Centroid values
        self.gpu_stage_centroid_values.set(self.domain.stage_centroid_values)
        self.gpu_xmom_centroid_values.set(self.domain.xmom_centroid_values)
        self.gpu_ymom_centroid_values.set(self.domain.ymom_centroid_values)
        self.gpu_height_centroid_values.set(self.domain.height_centroid_values)
        self.gpu_bed_centroid_values.set(self.domain.bed_centroid_values)

        # Edge values
        self.gpu_stage_edge_values.set(self.domain.stage_edge_values)
        self.gpu_xmom_edge_values.set(self.domain.xmom_edge_values)
        self.gpu_ymom_edge_values.set(self.domain.ymom_edge_values)
        self.gpu_bed_edge_values.set(self.domain.bed_edge_values)
        self.gpu_height_edge_values.set(self.domain.height_edge_values)

        # Boundary values
        self.gpu_stage_boundary_values.set(self.domain.stage_boundary_values)
        self.gpu_xmom_boundary_values.set(self.domain.xmom_boundary_values)
        self.gpu_ymom_boundary_values.set(self.domain.ymom_boundary_values)

        # Updates
        self.gpu_stage_explicit_update.set(self.domain.stage_explicit_update)
        self.gpu_xmom_explicit_update.set(self.domain.xmom_explicit_update)
        self.gpu_ymom_explicit_update.set(self.domain.ymom_explicit_update)
        self.gpu_stage_semi_implicit_update.set(self.domain.stage_semi_implicit_update)
        self.gpu_xmom_semi_implicit_update.set(self.domain.xmom_semi_implicit_update)
        self.gpu_ymom_semi_implicit_update.set(self.domain.ymom_semi_implicit_update)

        # Max speed
        self.gpu_max_speed.set(self.domain.max_speed)

    # ------------------------------------------------------------------
    # GPU → CPU transfer methods
    # ------------------------------------------------------------------

    def transfer_to_cpu(self):
        """Transfer results from GPU back to CPU arrays."""
        # Centroid values
        cp.asnumpy(self.gpu_stage_centroid_values, out=self.domain.stage_centroid_values)
        cp.asnumpy(self.gpu_xmom_centroid_values, out=self.domain.xmom_centroid_values)
        cp.asnumpy(self.gpu_ymom_centroid_values, out=self.domain.ymom_centroid_values)
        cp.asnumpy(self.gpu_height_centroid_values, out=self.domain.height_centroid_values)
        cp.asnumpy(self.gpu_bed_centroid_values, out=self.domain.bed_centroid_values)

        # Edge values
        cp.asnumpy(self.gpu_stage_edge_values, out=self.domain.stage_edge_values)
        cp.asnumpy(self.gpu_xmom_edge_values, out=self.domain.xmom_edge_values)
        cp.asnumpy(self.gpu_ymom_edge_values, out=self.domain.ymom_edge_values)
        cp.asnumpy(self.gpu_bed_edge_values, out=self.domain.bed_edge_values)
        cp.asnumpy(self.gpu_height_edge_values, out=self.domain.height_edge_values)

        # Vertex values
        cp.asnumpy(self.gpu_stage_vertex_values, out=self.domain.stage_vertex_values)
        cp.asnumpy(self.gpu_height_vertex_values, out=self.domain.height_vertex_values)
        cp.asnumpy(self.gpu_xmom_vertex_values, out=self.domain.xmom_vertex_values)
        cp.asnumpy(self.gpu_ymom_vertex_values, out=self.domain.ymom_vertex_values)
        cp.asnumpy(self.gpu_bed_vertex_values, out=self.domain.bed_vertex_values)

        # Updates
        cp.asnumpy(self.gpu_stage_explicit_update, out=self.domain.stage_explicit_update)
        cp.asnumpy(self.gpu_xmom_explicit_update, out=self.domain.xmom_explicit_update)
        cp.asnumpy(self.gpu_ymom_explicit_update, out=self.domain.ymom_explicit_update)
        cp.asnumpy(self.gpu_stage_semi_implicit_update, out=self.domain.stage_semi_implicit_update)
        cp.asnumpy(self.gpu_xmom_semi_implicit_update, out=self.domain.xmom_semi_implicit_update)
        cp.asnumpy(self.gpu_ymom_semi_implicit_update, out=self.domain.ymom_semi_implicit_update)

        # Diagnostics
        cp.asnumpy(self.gpu_max_speed, out=self.domain.max_speed)
        cp.asnumpy(self.gpu_num_negative_cells, out=self.domain.num_negative_cells
                    if hasattr(self.domain, 'num_negative_cells') else np.zeros(1, dtype=np.int64))

    # ------------------------------------------------------------------
    # Kernel launch methods
    # ------------------------------------------------------------------

    def compute_fluxes(self, substep_count=0):
        """
        Compute fluxes on GPU.

        Parameters
        ----------
        substep_count : int
            Current substep within RK multi-stage method (0 = update timestep).

        Returns
        -------
        float
            Global minimum timestep from GPU reduction.
        """
        if not self.gpu_arrays_allocated:
            raise GPUError("Call allocate() before compute_fluxes()")

        timestep_out = self._gpu.compute_fluxes(
            self.gpu_timestep_array,
            self.gpu_local_boundary_flux_sum,
            self.gpu_max_speed,
            self.gpu_stage_explicit_update,
            self.gpu_xmom_explicit_update,
            self.gpu_ymom_explicit_update,
            self.gpu_stage_centroid_values,
            self.gpu_height_centroid_values,
            self.gpu_bed_centroid_values,
            self.gpu_stage_edge_values,
            self.gpu_xmom_edge_values,
            self.gpu_ymom_edge_values,
            self.gpu_bed_edge_values,
            self.gpu_height_edge_values,
            self.gpu_stage_boundary_values,
            self.gpu_xmom_boundary_values,
            self.gpu_ymom_boundary_values,
            self.gpu_areas,
            self.gpu_normals,
            self.gpu_edgelengths,
            self.gpu_radii,
            self.gpu_tri_full_flag,
            self.gpu_neighbours,
            self.gpu_neighbour_edges,
            self.gpu_edge_flux_type,
            self.gpu_edge_river_wall_counter,
            self.gpu_riverwall_elevation,
            self.gpu_riverwall_rowIndex,
            self.gpu_riverwall_hydraulic_properties,
            substep_count,
        )

        # GPU-side reduction
        gpu_min_timestep = float(self.gpu_timestep_array.min())
        self.domain.boundary_flux_sum[substep_count] = float(
            self.gpu_local_boundary_flux_sum.sum())

        if substep_count == 0:
            return gpu_min_timestep
        return timestep_out

    def extrapolate_second_order(self):
        """Run all 4 extrapolation kernels."""
        self._gpu.extrapolate_second_order(
            self.gpu_stage_centroid_values,
            self.gpu_xmom_centroid_values,
            self.gpu_ymom_centroid_values,
            self.gpu_height_centroid_values,
            self.gpu_bed_centroid_values,
            self.gpu_x_centroid_work,
            self.gpu_y_centroid_work,
            self.gpu_stage_edge_values,
            self.gpu_xmom_edge_values,
            self.gpu_ymom_edge_values,
            self.gpu_height_edge_values,
            self.gpu_bed_edge_values,
            self.gpu_number_of_boundaries,
            self.gpu_centroid_coordinates,
            self.gpu_edge_coordinates,
            self.gpu_surrogate_neighbours,
            self.gpu_stage_vertex_values,
            self.gpu_height_vertex_values,
            self.gpu_xmom_vertex_values,
            self.gpu_ymom_vertex_values,
            self.gpu_bed_vertex_values,
        )

    def update_conserved(self, timestep):
        """
        Update conserved quantities and fix negative cells.

        Parameters
        ----------
        timestep : float
            Current timestep size.
        """
        self._gpu.update_conserved(
            timestep,
            self.gpu_stage_centroid_values,
            self.gpu_xmom_centroid_values,
            self.gpu_ymom_centroid_values,
            self.gpu_stage_explicit_update,
            self.gpu_xmom_explicit_update,
            self.gpu_ymom_explicit_update,
            self.gpu_stage_semi_implicit_update,
            self.gpu_xmom_semi_implicit_update,
            self.gpu_ymom_semi_implicit_update,
            self.gpu_tri_full_flag,
            self.gpu_bed_centroid_values,
            self.gpu_num_negative_cells,
        )

    def protect(self):
        """Protect against infinitesimal and negative heights."""
        self._gpu.protect(
            self.gpu_stage_centroid_values,
            self.gpu_bed_centroid_values,
            self.gpu_xmom_centroid_values,
            self.gpu_areas,
            self.gpu_stage_vertex_values,
        )

    def manning_friction_flat(self):
        """Compute Manning friction forcing (flat bed)."""
        self._gpu.manning_friction_flat(
            self.gpu_stage_centroid_values,
            self.gpu_bed_vertex_values,
            self.gpu_xmom_centroid_values,
            self.gpu_ymom_centroid_values,
            self.gpu_friction_centroid_values,
            self.gpu_xmom_semi_implicit_update,
            self.gpu_ymom_semi_implicit_update,
        )

    def get_num_negative_cells(self):
        """Return count of negative depth cells fixed on GPU."""
        return int(cp.asnumpy(self.gpu_num_negative_cells).sum())
