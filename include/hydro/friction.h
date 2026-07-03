#ifndef HYDRO_FRICTION_H
#define HYDRO_FRICTION_H

#include "types.h"
#include "domain.h"

#ifdef __cplusplus
extern "C" {
#endif

void hydro_manning_friction_flat(hydro_domain_t* domain);
void hydro_manning_friction_sloped(hydro_domain_t* domain);
void hydro_manning_friction_flat_semi_implicit(hydro_domain_t* domain);
void hydro_manning_friction_sloped_semi_implicit(hydro_domain_t* domain);
void hydro_manning_friction_sloped_semi_implicit_edge_based(hydro_domain_t* domain);

#ifdef __cplusplus
}
#endif

#endif /* HYDRO_FRICTION_H */
