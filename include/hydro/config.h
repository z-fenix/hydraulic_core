#ifndef HYDRO_CONFIG_H
#define HYDRO_CONFIG_H

/* ==========================================================================
 * Physical Constants
 * ========================================================================== */

/** Acceleration due to gravity [m/s^2] */
#define HYDRO_G 9.8

/** Default Manning friction coefficient */
#define HYDRO_MANNING_DEFAULT 0.03

/** Wind stress coefficient */
#define HYDRO_ETA_W 3.0e-3

/** Atmospheric density [kg/m^3] */
#define HYDRO_RHO_A 1.2e-3

/** Fluid density (salt water) [kg/m^3] */
#define HYDRO_RHO_W 1023.0

/* ==========================================================================
 * Numerical Constants
 * ========================================================================== */

/** Safe division threshold */
#define HYDRO_EPSILON 1.0e-12

/** Tiny value for comparisons */
#define HYDRO_TINY 1.0e-100

/** Single-precision epsilon */
#define HYDRO_SINGLE_PRECISION 1.0e-6

/** Sentinel for uninitialised float values */
#define HYDRO_MAX_FLOAT 1.0e36

/** pi */
#define HYDRO_PI 3.14159265358979

/* ==========================================================================
 * Shallow Water Model Defaults
 * ========================================================================== */

/** Minimum water depth treated as dry [m] */
#define HYDRO_MINIMUM_ALLOWED_HEIGHT 1.0e-05

/** Minimum water depth stored to file [m] */
#define HYDRO_MINIMUM_STORABLE_HEIGHT 1.0e-03

/** Maximum allowed water particle speed (0 = unlimited) [m/s] */
#define HYDRO_MAXIMUM_ALLOWED_SPEED 0.0

/** CFL condition coefficient */
#define HYDRO_CFL 1.0

/** Default spatial order */
#define HYDRO_DEFAULT_ORDER 2

/** Default beta parameter for limiter */
#define HYDRO_BETA_W 1.0

/** Limiter balance parameter */
#define HYDRO_ALPHA_BALANCE 2.0

/** Maximum Froude number for limiter checks */
#define HYDRO_MAXIMUM_FROUDE_NUMBER 100.0

/* ==========================================================================
 * Timestepping Defaults
 * ========================================================================== */

/** Minimum accepted timestep [s] */
#define HYDRO_MIN_TIMESTEP 1.0e-6

/** Maximum accepted timestep [s] */
#define HYDRO_MAX_TIMESTEP 1.0e+3

/** Maximum number of degenerate small steps before error */
#define HYDRO_MAX_SMALLSTEPS 50

/** Default timestepping method: 1=Euler, 2=RK2, 3=RK3 */
#define HYDRO_DEFAULT_TIMESTEPPING_METHOD 2

/* ==========================================================================
 * Performance Flags
 * ========================================================================== */

/** Enable optimisation: skip dry/still cells in flux computation */
#define HYDRO_OPTIMISE_DRY_CELLS 1

/** Use edge-based limiting (vs vertex-based) */
#define HYDRO_USE_EDGE_LIMITER 0

/** Extrapolate velocity (not momentum) in 2nd order reconstruction */
#define HYDRO_EXTRAPOLATE_VELOCITY_SECOND_ORDER 1

/** Low Froude correction mode: 0=off, 1=type1, 2=type2 */
#define HYDRO_LOW_FROUDE 0

/** Use sloped Manning friction */
#define HYDRO_SLOPED_MANNINGS 0

/* ==========================================================================
 * I/O Defaults
 * ========================================================================== */

/** Default institution name for SWW metadata */
#define HYDRO_INSTITUTION "hydro_core"

/** SWW output precision: 0=float32, 1=float64 */
#define HYDRO_SWW_PRECISION 0

#endif /* HYDRO_CONFIG_H */
