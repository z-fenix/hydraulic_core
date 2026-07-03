#ifndef HYDRO_STRUCTURES_H
#define HYDRO_STRUCTURES_H

#include "types.h"
#include "domain.h"

#ifdef __cplusplus
extern "C" {
#endif

/* =========================================================================
 * Hydraulic Structure Discharge Computations
 *
 * These functions compute the discharge Q through a structure given
 * upstream and downstream energy heads. They are called once per timestep
 * to determine flow between two regions of the mesh.
 * ========================================================================= */

/* =========================================================================
 * Boyd Box Culvert (Rectangular)
 *
 * Implements Boyd (1987) method for rectangular box culverts.
 * Computes discharge based on inlet control (weir vs orifice) and
 * outlet control (energy slope), taking the minimum.
 * ========================================================================= */

/* Compute discharge through a rectangular box culvert.
 *
 * Parameters:
 *   g: gravitational acceleration (e.g. 9.8)
 *   width: internal culvert width (m)
 *   height: internal culvert height (m) (or 0 if square = width)
 *   barrels: number of parallel barrels
 *   blockage: fractional blockage [0, 1]
 *   losses: total head-loss coefficient (sum of entry, exit, friction losses)
 *   manning: Manning's n for barrel friction
 *   culvert_length: distance between inlet and outlet (m)
 *   use_velocity_head: if non-zero, include v^2/(2g) in driving energy
 *   inlet_energy: upstream total energy head (m) = stage + v^2/(2g)
 *   outlet_energy: downstream total energy head (m)
 *   inlet_depth: water depth at inlet above invert (m)
 *   outlet_depth: water depth at outlet above invert (m)
 *   Q: output discharge (m^3/s)
 *   velocity: output barrel velocity (m/s)
 *   flow_area: output cross-sectional flow area (m^2)
 *
 * Returns 0 on success, -1 if parameters are invalid.
 */
int hydro_boyd_box_discharge(
    double g,
    double width, double height,
    double barrels, double blockage,
    double losses, double manning,
    double culvert_length,
    int use_velocity_head,
    double inlet_energy, double outlet_energy,
    double inlet_depth, double outlet_depth,
    double* Q, double* velocity, double* flow_area);

/* =========================================================================
 * Boyd Pipe Culvert (Circular)
 *
 * Same algorithm as box culvert but with circular cross-section geometry.
 * ========================================================================= */

/* Compute discharge through a circular pipe culvert.
 *
 * Parameters (same as box except):
 *   diameter: internal pipe diameter (m)
 *   width, height are replaced by diameter
 */
int hydro_boyd_pipe_discharge(
    double g,
    double diameter,
    double barrels, double blockage,
    double losses, double manning,
    double culvert_length,
    int use_velocity_head,
    double inlet_energy, double outlet_energy,
    double inlet_depth, double outlet_depth,
    double* Q, double* velocity, double* flow_area);

/* =========================================================================
 * Weir/Orifice (Trapezoidal Cross-Section)
 *
 * Combined weir and orifice equations for trapezoidal cross-section.
 * Uses Newton-Raphson iteration for critical depth.
 * ========================================================================= */

/* Compute discharge through a trapezoidal weir/orifice.
 *
 * Parameters:
 *   width: bottom width (m)
 *   z1, z2: side slopes (horizontal/vertical, e.g. 2 means 2H:1V)
 *   Other parameters as in boyd_box_discharge.
 */
int hydro_weir_orifice_trapezoid_discharge(
    double g,
    double width,
    double z1, double z2,
    double barrels, double blockage,
    double losses, double manning,
    double culvert_length,
    int use_velocity_head,
    double inlet_energy, double outlet_energy,
    double inlet_depth, double outlet_depth,
    double* Q, double* velocity, double* flow_area);

/* =========================================================================
 * Inlet Operator Helper
 *
 * Distributes a volume of water across a set of triangles to achieve
 * a flat free surface. Used for source/sink boundary conditions.
 * ========================================================================= */

/* Distribute a volume of water evenly across a set of triangles.
 * volume: total volume to distribute (positive = add, negative = remove)
 * indices: triangle indices in the inlet region
 * areas: triangle areas (must match indices length)
 * num_indices: number of triangles in inlet
 * stage_c: stage centroid values (modified in place)
 * elev_c: elevation centroid values
 * xmom_c, ymom_c: momentum centroid values (scaled on negative volume)
 *
 * Returns the actual volume distributed (may be less than requested if
 * draining would go below bed).
 */
double hydro_inlet_distribute_volume(
    double volume,
    const hydro_int* indices,
    const double* areas,
    hydro_int num_indices,
    double* stage_c,
    const double* elev_c,
    double* xmom_c,
    double* ymom_c);

#ifdef __cplusplus
}
#endif

#endif /* HYDRO_STRUCTURES_H */
