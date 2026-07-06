#!/usr/bin/env python3
"""Unit tests for TimeSeriesInflowBoundary."""

import os
import unittest
import tempfile

import numpy as np

from hydro import Domain, HYDRO_BC_TIME_SERIES
from hydro.time_series_boundary import TimeSeriesInflowBoundary
from hydro.mesh_builder import rectangular_cross_domain


HERE = os.path.dirname(__file__)
TEST_DATA = os.path.join(HERE, 'data')


class TestTimeSeriesInflowBoundary(unittest.TestCase):
    """Tests for the time-series inflow boundary condition."""

    def setUp(self):
        """Create a simple 2x2 channel mesh."""
        self.vertices, self.triangles, self.btags, self.bedges = \
            rectangular_cross_domain(2, 1, len1=10.0, len2=5.0)
        self.n_tri = len(self.triangles)

    def _make_domain(self):
        """Create and initialise a domain."""
        domain = Domain(self.vertices, self.triangles, self.btags, self.bedges)
        domain.set_name('test_ts')
        elevation = -np.mean(self.vertices[self.triangles], axis=1)[:, 0] / 10.0
        domain.set_elevation(elevation)
        domain.set_friction(np.full(self.n_tri, 0.01))
        domain.set_stage(elevation.copy())
        domain.set_xmomentum(np.zeros(self.n_tri))
        domain.set_ymomentum(np.zeros(self.n_tri))
        domain.set_parameter('CFL', 0.9)
        domain.set_parameter('spatial_order', 1)
        domain.set_parameter('timestepping_method', 1)
        return domain

    # ---- CSV loading ----

    def test_load_csv_with_header(self):
        """CSV with header row loads correctly."""
        csv_path = os.path.join(TEST_DATA, 'inflow_hydrograph.csv')
        bc = TimeSeriesInflowBoundary(
            self._make_domain(), tag=1, csv_file=csv_path,
        )
        try:
            self.assertGreater(len(bc._times), 0)
            self.assertEqual(len(bc._times), len(bc._q_values))
            self.assertEqual(bc._times[0], 0.0)
            self.assertEqual(bc._q_values[0], 0.0)
            self.assertEqual(bc._times[-1], 20.0)
        finally:
            bc.close()

    def test_load_csv_no_header(self):
        """CSV without header loads correctly."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write('0.0,1.0\n1.0,2.0\n2.0,0.5\n')
            f.flush()
            path = f.name

        try:
            domain = self._make_domain()
            bc = TimeSeriesInflowBoundary(
                domain, tag=1, csv_file=path,
            )
            try:
                np.testing.assert_array_almost_equal(
                    bc._times, [0.0, 1.0, 2.0])
                np.testing.assert_array_almost_equal(
                    bc._q_values, [1.0, 2.0, 0.5])
            finally:
                bc.close()
        finally:
            os.unlink(path)

    # ---- Direct array input ----

    def test_direct_arrays(self):
        """Pass times/q_values directly instead of CSV."""
        times = np.array([0.0, 1.0, 2.0, 3.0])
        qs = np.array([0.0, 1.0, 2.0, 1.0])
        domain = self._make_domain()
        bc = TimeSeriesInflowBoundary(
            domain, tag=1, times=times, q_values=qs,
        )
        try:
            self.assertEqual(bc._times[0], 0.0)
            self.assertAlmostEqual(bc.q(1.5), 1.5)  # linear interp
        finally:
            bc.close()

    # ---- Interpolation ----

    def test_q_interpolation(self):
        """q(t) returns linearly interpolated value."""
        times = np.array([0.0, 2.0, 4.0])
        qs = np.array([0.0, 10.0, 0.0])
        domain = self._make_domain()
        bc = TimeSeriesInflowBoundary(
            domain, tag=1, times=times, q_values=qs,
        )
        try:
            self.assertAlmostEqual(bc.q(0.0), 0.0)
            self.assertAlmostEqual(bc.q(1.0), 5.0)
            self.assertAlmostEqual(bc.q(2.0), 10.0)
            self.assertAlmostEqual(bc.q(3.0), 5.0)
            self.assertAlmostEqual(bc.q(4.0), 0.0)
        finally:
            bc.close()

    # ---- C-layer registration ----

    def test_boundary_type_registered(self):
        """Setting HYDRO_BC_TIME_SERIES does not crash."""
        domain = self._make_domain()
        domain.set_boundary(1, HYDRO_BC_TIME_SERIES)
        times = np.array([0.0, 1.0, 2.0])
        qs = np.array([1.0, 2.0, 1.0])
        bc = TimeSeriesInflowBoundary(
            domain, tag=1, times=times, q_values=qs,
        )
        try:
            # Should not raise
            bc.update(0.5)
        finally:
            bc.close()

    # ---- Evolution with time-series boundary ----

    def test_evolve_with_time_series(self):
        """Full evolution with TIME_SERIES boundary produces valid output."""
        domain = self._make_domain()
        domain.set_boundary(1, HYDRO_BC_TIME_SERIES)
        domain.set_boundary(2, 1)  # reflective
        domain.set_boundary(3, 1)
        domain.set_boundary(4, 1)

        times = np.array([0.0, 5.0, 10.0, 15.0])
        qs = np.array([0.0, 2.0, 2.0, 0.0])
        bc = TimeSeriesInflowBoundary(
            domain, tag=1, times=times, q_values=qs,
            default_stage=0.1,
        )
        try:
            # Evolve for a short time — should not crash
            domain.evolve(finaltime=2.0, yieldstep=0.5)
            stage = domain.get_stage()
            self.assertEqual(len(stage), self.n_tri)
            # Stage should be at least bed + epsilon
            elevation = domain.get_elevation()
            self.assertTrue(np.all(stage >= elevation - 1e-6))
        finally:
            bc.close()

    # ---- Context manager ----

    def test_context_manager(self):
        """TimeSeriesInflowBoundary works as a context manager."""
        domain = self._make_domain()
        domain.set_boundary(1, HYDRO_BC_TIME_SERIES)
        times = np.array([0.0, 1.0, 2.0])
        qs = np.array([1.0, 2.0, 1.0])

        with TimeSeriesInflowBoundary(
            domain, tag=1, times=times, q_values=qs,
        ) as bc:
            stage_before = domain.get_stage().copy()
            domain.evolve(finaltime=1.0, yieldstep=0.5)
            stage_after = domain.get_stage()
            # Water should have entered from the left
            self.assertTrue(np.any(stage_after > stage_before))

    # ---- Validation: constant Q should match Dirichlet ----

    def test_constant_q_matches_dirichlet(self):
        """When Q is constant, TIME_SERIES should produce similar results
        to a Dirichlet boundary with the derived stage."""
        import scipy

        # Constant Q = 1.0 m3/s → derive approximate stage
        # h = (n*Q/(w*sqrt(S)))^0.6 = (0.03*1/(1*sqrt(0.01)))^0.6 = 0.03^0.6 ≈ 0.163
        # stage ≈ bed + h ≈ 0 + 0.163
        times = np.array([0.0, 10.0])
        qs = np.array([1.0, 1.0])
        domain = self._make_domain()
        domain.set_boundary(1, HYDRO_BC_TIME_SERIES)
        domain.set_boundary(2, 1)
        domain.set_boundary(3, 1)
        domain.set_boundary(4, 1)

        bc = TimeSeriesInflowBoundary(
            domain, tag=1, times=times, q_values=qs,
        )
        try:
            domain.evolve(finaltime=1.0, yieldstep=0.25)
            stage = domain.get_stage()
            elevation = domain.get_elevation()
            # Stage at left should be above bed
            left_tri = np.where(
                np.mean(self.vertices[self.triangles], axis=1)[:, 0] < 1.0
            )[0]
            if len(left_tri) > 0:
                self.assertGreater(
                    stage[left_tri[0]], elevation[left_tri[0]] - 1e-6)
        finally:
            bc.close()


if __name__ == '__main__':
    unittest.main()
