#ifndef HYDRO_SWW_H
#define HYDRO_SWW_H

#include "types.h"
#include "domain.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Opaque handle for SWW file writer */
typedef struct hydro_sww_t hydro_sww_t;

/**
 * Create an SWW file and write the static header + geometry.
 *
 * @param path        Output file path (".sww" appended if missing)
 * @param domain      The domain (must have geometry set up)
 * @param starttime   Absolute start time (seconds since epoch)
 * @return            SWW file handle, or NULL on error
 */
hydro_sww_t* hydro_sww_create(
    const char*           path,
    const hydro_domain_t* domain,
    double                starttime);

/**
 * Open an existing SWW file for appending timesteps.
 *
 * Reads existing geometry and positions the writer after the last
 * stored timestep.  The domain must match the file's geometry.
 *
 * @param path    Path to existing SWW file
 * @param domain  The domain (must match file geometry)
 * @return        SWW file handle, or NULL on error
 */
hydro_sww_t* hydro_sww_open(
    const char*           path,
    const hydro_domain_t* domain);

/**
 * Write one timestep of dynamic quantity data.
 *
 * Appends: time, stage, xmomentum, ymomentum values at triangle vertices.
 *
 * @param sww    SWW file handle
 * @param domain The domain
 * @param time   Current simulation time (relative to starttime)
 * @return 0 on success
 */
int hydro_sww_store_timestep(
    hydro_sww_t*         sww,
    const hydro_domain_t* domain,
    double               time);

/**
 * Close the SWW file.
 *
 * @param sww  SWW file handle (freed on return)
 * @return 0 on success
 */
int hydro_sww_close(hydro_sww_t* sww);

#ifdef __cplusplus
}
#endif

#endif /* HYDRO_SWW_H */
