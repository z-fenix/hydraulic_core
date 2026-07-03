/**
 * friction.c — Manning friction (semi-implicit)
 * Ported from anuga/shallow_water/sw_domain_openmp.c
 */
#include "hydro/friction.h"
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif

void hydro_manning_friction_flat_semi_implicit(hydro_domain_t* domain) {
    hydro_int N = domain->number_of_elements;
    double g = domain->g, eps = domain->minimum_allowed_height;
    const double st = 7.0/3.0;
    if (!domain->xmom_centroid_values || !domain->friction_centroid_values) return;
    #ifdef _OPENMP
    #pragma omp parallel for simd schedule(static)
    #endif
    for (hydro_int k = 0; k < N; k++) {
        double S = 0.0, uh = domain->xmom_centroid_values[k];
        double vh = domain->ymom_centroid_values[k];
        double eta = domain->friction_centroid_values[k];
        if (eta > 1e-15) {
            double h = domain->stage_centroid_values[k] - domain->bed_centroid_values[k];
            if (h >= eps) S = -g*eta*eta*sqrt(uh*uh+vh*vh)/pow(h, st);
        }
        domain->xmom_semi_implicit_update[k] += S*uh;
        domain->ymom_semi_implicit_update[k] += S*vh;
    }
}

void hydro_manning_friction_sloped_semi_implicit(hydro_domain_t* domain) {
    hydro_int N = domain->number_of_elements;
    double g = domain->g, eps = domain->minimum_allowed_height;
    const double ot = 1.0/3.0, st = 7.0/3.0;
    if (!domain->xmom_centroid_values || !domain->bed_vertex_values) return;
    #ifdef _OPENMP
    #pragma omp parallel for simd schedule(static)
    #endif
    for (hydro_int k = 0; k < N; k++) {
        double S = 0.0, zx, zy;
        hydro_int k3 = 3*k, k6 = 6*k;
        double uh = domain->xmom_centroid_values[k];
        double vh = domain->ymom_centroid_values[k];
        double eta = domain->friction_centroid_values[k];
        double z0 = domain->bed_vertex_values[k3];
        double z1 = domain->bed_vertex_values[k3+1];
        double z2 = domain->bed_vertex_values[k3+2];
        double x0 = domain->vertex_coordinates[k6];
        double y0 = domain->vertex_coordinates[k6+1];
        double x1 = domain->vertex_coordinates[k6+2];
        double y1 = domain->vertex_coordinates[k6+3];
        double x2 = domain->vertex_coordinates[k6+4];
        double y2 = domain->vertex_coordinates[k6+5];
        if (eta > 1e-16) {
            double det = (y2-y0)*(x1-x0) - (y1-y0)*(x2-x0);
            zx = ((y2-y0)*(z1-z0) - (y1-y0)*(z2-z0))/det;
            zy = ((x1-x0)*(z2-z0) - (x2-x0)*(z1-z0))/det;
            double zs = sqrt(1.0 + zx*zx + zy*zy);
            double z = (z0+z1+z2)*ot, h = domain->stage_centroid_values[k] - z;
            if (h >= eps) S = -g*eta*eta*zs*sqrt(uh*uh+vh*vh)/pow(h, st);
        }
        domain->xmom_semi_implicit_update[k] += S*uh;
        domain->ymom_semi_implicit_update[k] += S*vh;
    }
}

void hydro_manning_friction_flat(hydro_domain_t* d) { (void)d; }
void hydro_manning_friction_sloped(hydro_domain_t* d) { (void)d; }
void hydro_manning_friction_sloped_semi_implicit_edge_based(hydro_domain_t* d) { (void)d; }
