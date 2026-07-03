#ifndef HYDRO_H
#define HYDRO_H

/**
 * Hydro Core — Shallow Water Equation Solver
 *
 * A standalone C library for finite-volume simulation of the shallow water
 * equations on unstructured triangular meshes.
 *
 * Usage:
 *   1. Create a domain:   hydro_domain_create()
 *   2. Set geometry:      hydro_domain_set_geometry()
 *   3. Set quantities:    hydro_domain_set_quantity()
 *   4. Set boundaries:    hydro_domain_set_boundary()
 *   5. Evolve:            hydro_domain_evolve()
 *   6. Destroy:           hydro_domain_destroy()
 */

/* Export all symbols when building the shared library */
#pragma GCC visibility push(default)

#include "config.h"
#include "types.h"
#include "domain.h"
#include "mesh.h"
#include "quantity.h"
#include "fluxes.h"
#include "boundaries.h"
#include "friction.h"
#include "forcing.h"
#include "operators.h"
#include "structures.h"
#include "sww.h"
#include "geometry.h"
#include "fit_interpolate.h"
#include "coordinate_transforms.h"
#include "timestepping.h"

#pragma GCC visibility pop

#endif /* HYDRO_H */
