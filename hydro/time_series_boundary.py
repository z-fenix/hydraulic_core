#!/usr/bin/env python3
"""
Time-series inflow boundary driven by CSV flow rate data.

Ported from ANUGA's scenario/setup_boundary_conditions.py pattern.

Usage
-----
>>> import numpy as np
>>> from hydro import Domain, HYDRO_BC_TIME_SERIES
>>> from hydro.time_series_boundary import TimeSeriesInflowBoundary
>>>
>>> domain = Domain(vertices, triangles, btags, bedges)
>>> domain.set_boundary(1, HYDRO_BC_TIME_SERIES)  # left = inflow
>>>
>>> bc = TimeSeriesInflowBoundary(domain, tag=1, csv_file='inflow.csv')
>>> domain.evolve(finaltime=10.0, yieldstep=0.1)
>>> bc.close()

The boundary reads Q(t) from a CSV file (two columns: time, discharge),
interpolates at each yield step, and derives stage from Q using Manning's
equation.  The effective channel width is auto-derived from the boundary
edge geometry (sum of edge lengths for edges with the given tag).
"""

import numpy as np
from scipy.interpolate import interp1d

from hydro._core import HYDRO_BC_TIME_SERIES


class TimeSeriesInflowBoundary:
    """Apply a time-varying flow rate Q(t) as an inflow boundary condition.

    Reads Q(t) from a CSV file (two columns: time, discharge),
    interpolates at each yield step, and derives stage from the
    discharge using Manning's equation.  The effective channel width
    is auto-derived from the boundary edge geometry.

    Parameters
    ----------
    domain : Domain
        The hydro_core Domain object.
    tag : int
        Boundary tag (1=left, etc.).
    csv_file : str, optional
        Path to CSV with columns: time, discharge (m3/s).
        First row is treated as a header if it contains non-numeric data.
    times : ndarray, optional
        Direct time array (alternative to csv_file).
    q_values : ndarray, optional
        Direct discharge array (alternative to csv_file).
    default_stage : float, default 0.1
        Fallback stage (m) when Q is out of CSV range or Q <= 0.
    """

    def __init__(
        self,
        domain: Domain,
        tag: int,
        csv_file: str | None = None,
        times: np.ndarray | None = None,
        q_values: np.ndarray | None = None,
        default_stage: float = 0.1,
    ):
        self._domain = domain
        self._tag = tag
        self._default_stage = default_stage

        if csv_file is not None:
            self._load_csv(csv_file)
        elif times is not None and q_values is not None:
            self._times = np.asarray(times, dtype=np.float64)
            self._q_values = np.asarray(q_values, dtype=np.float64)
        else:
            raise ValueError(
                "Provide either csv_file or both times and q_values"
            )

        # Sanity-check arrays
        if len(self._times) < 2:
            raise ValueError("Need at least 2 time-series points")
        if len(self._times) != len(self._q_values):
            raise ValueError("times and q_values must have same length")
        if not np.all(np.diff(self._times) >= 0):
            raise ValueError("times must be non-decreasing")

        # Register with C layer
        domain.set_time_series_boundary(
            tag, self._times, self._q_values, default_stage=default_stage
        )

        # Build scipy interpolator for fast Python-side lookup
        self._interp = interp1d(
            self._times, self._q_values,
            kind='linear', fill_value='extrapolate',
        )
        self._t_min = self._times[0]
        self._t_max = self._times[-1]

    # ------------------------------------------------------------------
    # CSV loading
    # ------------------------------------------------------------------

    def _load_csv(self, path: str) -> None:
        """Load time, Q from CSV (skip header if non-numeric)."""
        raw_str = np.loadtxt(path, delimiter=',', dtype=str)
        if raw_str.ndim == 1:
            raw_str = raw_str.reshape(-1, 1)

        # If single column, transpose to (n, 2)
        if raw_str.shape[1] == 1:
            raw_str = np.hstack([raw_str, raw_str])

        # Detect header: if any cell in first row is non-numeric, skip it
        has_header = False
        for val in raw_str[0]:
            try:
                float(val)
            except (ValueError, TypeError):
                has_header = True
                break

        raw = raw_str[1:] if has_header else raw_str
        raw = raw.astype(np.float64)

        self._times = raw[:, 0]
        self._q_values = raw[:, 1]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def q(self, t: float) -> float:
        """Interpolated discharge at time t (m3/s)."""
        assert self._interp is not None
        return float(self._interp(t))

    def update(self, t: float | None = None) -> None:
        """Update boundary values at the given time.

        If t is None, uses the domain's current time.
        Delegates to the C layer which auto-derives channel width
        from boundary edge geometry.
        """
        if t is None:
            t = self._domain.get_time()

        self._domain.update_time_series_boundary(self._tag, t)

    def close(self) -> None:
        """Clean up resources."""
        self._interp = None

    # ------------------------------------------------------------------
    # Context manager protocol
    # ------------------------------------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
