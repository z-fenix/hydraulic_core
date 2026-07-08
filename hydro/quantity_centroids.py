#!/usr/bin/env python3
"""
Triangle-centroid quantity extraction from SWW files.

Ported from ANUGA's ``plot_utils.get_centroids``.  Reads vertex-stored
variables from a hydro SWW (NetCDF) file and computes triangle-centroid
quantities by averaging over the three vertices of each triangle.

Usage
-----
>>> from hydro import QuantityCentroids
>>> qc = QuantityCentroids('output.sww')
>>> print(qc.time, qc.stage.shape)  # (nt, n_tri)
>>> print(qc.xvel[0, :5])           # x-velocity at first 5 triangles

Parameters
----------
velocity_extrapolation : bool
    If ``True``, compute centroid velocities by first averaging vertex
    velocities, then deriving centroid momenta from
    ``momentum = velocity * height``.
    If ``False``, average vertex momenta to centroids, then divide by
    centroid height to obtain velocities.

Dependencies
------------
Requires the optional ``netCDF4`` package (``pip install hydro_core[netcdf]``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import netCDF4
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "netCDF4 is required for QuantityCentroids. "
        "Install it with: pip install netCDF4"
    )


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _resolve_indices(time_slices, n_timesteps: int) -> list[int] | str:
    """Map a ``time_slices`` argument to integer indices or a sentinel.

    Sentinels
    ---------
    'all'  → list(range(n_timesteps))
    'last' → [n_timesteps - 1]
    'max'  → the string 'max' (caller handles special logic)
    """
    if time_slices == 'all':
        return list(range(n_timesteps))
    if time_slices == 'last':
        return [n_timesteps - 1]
    if time_slices == 'max':
        return 'max'
    # Assume iterable of integers
    return list(time_slices)


def _select_2d(arr: np.ndarray, indices: list[int]) -> np.ndarray:
    """Select rows *indices* from a 2-D array (nt, nv)."""
    return arr[np.array(indices, dtype=np.intp), :]


def _max_abs(arr: np.ndarray) -> np.ndarray:
    """For each column, return the value with the largest absolute magnitude.

    Parameters
    ----------
    arr : (nt, nv)
    Returns
    -------
    (1, nv) — kept as 2-D for broadcasting consistency.
    """
    idx = np.abs(arr).argmax(axis=0)
    result = np.empty_like(arr[0], dtype=arr.dtype)
    for j in range(result.shape[0]):
        result[j] = arr[idx[j], j]
    return result.reshape(1, -1)


def _centroid_avg_vertex_2d(
    data: np.ndarray,  # (nt, nv)
    vols: np.ndarray,  # (n_tri, 3)
) -> np.ndarray:  # (nt, n_tri)
    """Average a 2-D vertex array to triangle centroids."""
    return (data[:, vols[:, 0]] + data[:, vols[:, 1]] + data[:, vols[:, 2]]) / 3.0


def _centroid_avg_vertex_1d(
    data: np.ndarray,  # (nv,)
    vols: np.ndarray,  # (n_tri, 3)
) -> np.ndarray:  # (n_tri,)
    """Average a 1-D vertex array to triangle centroids."""
    return (data[vols[:, 0]] + data[vols[:, 1]] + data[vols[:, 2]]) / 3.0


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

class QuantityCentroids:
    """Extract triangle-centroid quantities from an SWW file.

    Reads vertex-stored variables from a hydro SWW (NetCDF) file and
    computes centroid values by averaging over the three vertices of
    each triangle.

    Parameters
    ----------
    filename : str | Path
        Path to the ``.sww`` file.
    velocity_extrapolation : bool, default ``False``
        Control how centroid velocities / momenta are derived
        (see module docstring).
    time_slices : list[int] | str, default ``'all'``
        Which timesteps to load.  Accepted strings: ``'all'``, ``'last'``,
        ``'max'``.
    minimum_allowed_height : float, default ``1e-3``
        Water depth below which velocities are zeroed.

    Attributes (set on construction)
    --------------------------------
    time : ndarray (nt,)
        Time values of the selected timesteps.
    x, y : ndarray (n_tri,)
        Centroid x / y coordinates.
    stage : ndarray (nt, n_tri)
        Water stage (bed + depth).
    height : ndarray (nt, n_tri)
        Water depth (= stage - elevation).
    elev : ndarray (n_tri,)
        Bed elevation at centroids (first timestep only).
    xmom, ymom : ndarray (nt, n_tri)
        Momentum components at centroids.
    xvel, yvel : ndarray (nt, n_tri)
        Velocity components at centroids.
    vel : ndarray (nt, n_tri)
        Speed (= sqrt(xvel² + yvel²)) at centroids.
    friction : ndarray (n_tri,)
        Manning's n at centroids (first timestep only).
    xllcorner, yllcorner : float
        Coordinate offsets (global → local).
    timeSlices : list[int] | str
        Normalised ``time_slices`` argument.
    """

    def __init__(
        self,
        filename,
        velocity_extrapolation: bool = False,
        time_slices='all',
        minimum_allowed_height: float = 1e-3,
    ) -> None:
        self.filename = Path(filename)
        self.timeSlices = time_slices
        self.minimum_allowed_height = minimum_allowed_height

        # Resolve which timesteps to read
        with netCDF4.Dataset(str(self.filename), 'r') as fid:
            n_timesteps = fid.dimensions['number_of_timesteps'].size
            n_points = fid.dimensions['number_of_points'].size
            n_tri = fid.dimensions['number_of_volumes'].size

            time_all = fid.variables['time'][:]
            vols = fid.variables['volumes'][:].astype(np.intp)

            # Global offsets
            self.xllcorner = float(getattr(fid, 'xllcorner', 0.0))
            self.yllcorner = float(getattr(fid, 'yllcorner', 0.0))

            # Coordinates (unique vertices)
            x_vert = fid.variables['x'][:]
            y_vert = fid.variables['y'][:]

            # Centroid coordinates (static)
            self.x = _centroid_avg_vertex_1d(x_vert, vols) + self.xllcorner
            self.y = _centroid_avg_vertex_1d(y_vert, vols) + self.yllcorner

            # Elevation (static, per vertex)
            elev_vert = fid.variables['elevation'][:]

            # Friction (static, per vertex) — may be absent
            if 'friction' in fid.variables:
                friction_cent = _centroid_avg_vertex_1d(
                    fid.variables['friction'][:], vols
                )
            else:
                friction_cent = np.full(n_tri, np.nan, dtype=np.float32)

        # ---- Time-resolved quantities --------------------------------
        indices = _resolve_indices(time_slices, n_timesteps)

        if indices == 'max':
            # Need to read all timesteps to compute max
            with netCDF4.Dataset(str(self.filename), 'r') as fid:
                stage_all = fid.variables['stage'][:]
                xmom_all = fid.variables['xmomentum'][:]
                ymom_all = fid.variables['ymomentum'][:]

            # Centroid averages (all timesteps)
            stage_c = _centroid_avg_vertex_2d(stage_all, vols)
            xmom_c = _centroid_avg_vertex_2d(xmom_all, vols)
            ymom_c = _centroid_avg_vertex_2d(ymom_all, vols)

            # Elevation at first timestep
            elev_cent = _centroid_avg_vertex_1d(elev_vert, vols)

            # Height (use first timestep for bed reference)
            height_c = stage_c - elev_cent[np.newaxis, :]

            # Wet mask
            hWet = height_c > self.minimum_allowed_height

            # Velocities
            hInv = 1.0 / (height_c + 1.0e-12)
            xvel_c = xmom_c * hInv * hWet
            yvel_c = ymom_c * hInv * hWet
            vel_c = np.sqrt(xvel_c**2 + yvel_c**2)

            # Max-over-time reductions
            self.time = np.array([time_all.max()], dtype=np.float64)
            self.stage = stage_c.max(axis=0, keepdims=True)
            self.height = height_c.max(axis=0, keepdims=True)
            self.vel = vel_c.max(axis=0, keepdims=True)

            # Max-abs for signed quantities
            self.xmom = _max_abs(xmom_c)
            self.ymom = _max_abs(ymom_c)
            self.xvel = _max_abs(xvel_c)
            self.yvel = _max_abs(yvel_c)

            self.elev = elev_cent
            self.friction = friction_cent

        else:
            # Normal selection
            inds = indices  # list of ints
            n_sel = len(inds)
            self.time = time_all[inds]

            with netCDF4.Dataset(str(self.filename), 'r') as fid:
                stage_all = fid.variables['stage'][:]
                xmom_all = fid.variables['xmomentum'][:]
                ymom_all = fid.variables['ymomentum'][:]

            # Select timesteps
            stage_sel = _select_2d(stage_all, inds)
            xmom_sel = _select_2d(xmom_all, inds)
            ymom_sel = _select_2d(ymom_all, inds)

            # Centroid averages
            stage_c = _centroid_avg_vertex_2d(stage_sel, vols)
            xmom_c = _centroid_avg_vertex_2d(xmom_sel, vols)
            ymom_c = _centroid_avg_vertex_2d(ymom_sel, vols)

            if velocity_extrapolation:
                # --- Path A: avg vertex velocities → centroid velocities → centroid momenta
                # Vertex heights: elevation is static (1-D), broadcast across timesteps
                height_v = stage_sel - elev_vert[np.newaxis, :]
                hInv_v = 1.0 / (height_v + 1.0e-12)
                hWet_v = height_v > self.minimum_allowed_height

                xvel_v = xmom_sel * hInv_v * hWet_v
                yvel_v = ymom_sel * hInv_v * hWet_v

                # Centroid velocities
                xvel_c = _centroid_avg_vertex_2d(xvel_v, vols)
                yvel_c = _centroid_avg_vertex_2d(yvel_v, vols)

                # Centroid heights (from first timestep elevation)
                elev_cent = _centroid_avg_vertex_1d(elev_vert, vols)
                height_c = stage_c - elev_cent[np.newaxis, :]

                # Centroid momenta from centroid velocities
                hWet_c = height_c > self.minimum_allowed_height
                hInv_c = 1.0 / (height_c + 1.0e-12)
                xmom_c = xvel_c * height_c * hWet_c
                ymom_c = yvel_c * height_c * hWet_c
            else:
                # --- Path B: avg vertex momenta → centroid momenta → centroid velocities
                # Centroid heights (from first timestep elevation)
                elev_cent = _centroid_avg_vertex_1d(elev_vert, vols)
                height_c = stage_c - elev_cent[np.newaxis, :]

                hWet_c = height_c > self.minimum_allowed_height
                hInv_c = 1.0 / (height_c + 1.0e-12)

                xvel_c = xmom_c * hInv_c * hWet_c
                yvel_c = ymom_c * hInv_c * hWet_c

            vel_c = np.sqrt(xvel_c**2 + yvel_c**2)

            # Assign attributes
            self.time = self.time
            self.stage = stage_c
            self.height = height_c
            self.elev = elev_cent
            self.xmom = xmom_c
            self.ymom = ymom_c
            self.xvel = xvel_c
            self.yvel = yvel_c
            self.vel = vel_c
            self.friction = friction_cent

        # Normalise stage/height to 1-D when only 1 timestep
        if self.stage.shape[0] == 1:
            self.stage = self.stage[0]
            self.height = self.height[0]
            self.xmom = self.xmom[0]
            self.ymom = self.ymom[0]
            self.xvel = self.xvel[0]
            self.yvel = self.yvel[0]
            self.vel = self.vel[0]
