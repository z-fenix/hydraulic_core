#ifndef HYDRO_TIMESTEPPING_H
#define HYDRO_TIMESTEPPING_H

#include "types.h"
#include "domain.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Evolve domain from current time to finaltime, writing to optional SWW file. */
int hydro_domain_evolve(
    hydro_domain_t* domain,
    double          finaltime,
    double          yieldstep,
    const char*     output_sww_path);

/** Single Euler step. */
void hydro_evolve_one_euler_step(
    hydro_domain_t* domain,
    double          yieldstep,
    double          finaltime);

/** Single RK2 step. */
void hydro_evolve_one_rk2_step(
    hydro_domain_t* domain,
    double          yieldstep,
    double          finaltime);

/** Compute timestep based on CFL condition. */
void hydro_update_timestep(
    hydro_domain_t* domain,
    double          yieldstep,
    double          finaltime);

#ifdef __cplusplus
}
#endif

#endif /* HYDRO_TIMESTEPPING_H */
