/**
 * hydro_cuda.cu — CUDA kernels for hydro_core shallow water solver
 *
 * Migrated from anuga_core's archive/cupy_cuda/cuda_anuga.cu
 * Adapted to hydro_core's hydro_domain_t SoA layout.
 *
 * Kernels are compiled via CMake and exported as C functions callable
 * from pybind11 and/or Python (via cupy).
 */

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

/* ==========================================================================
 * Device utility functions
 * ========================================================================== */

__device__ int64_t __find_qmin_and_qmax(double dq0, double dq1, double dq2,
                                         double *qmin, double *qmax)
{
    *qmax = fmax(fmax(dq0, fmax(dq0 + dq1, dq0 + dq2)), 0.0);
    *qmin = fmin(fmin(dq0, fmin(dq0 + dq1, dq0 + dq2)), 0.0);
    return 0;
}

__device__ void __limit_gradient(double *dqv, double qmin, double qmax, double beta_w)
{
    int64_t i;
    double r = 1000.0, r0 = 1.0, phi = 1.0;
    static double TINY = 1.0e-100;

    for (i = 0; i < 3; i++) {
        if (dqv[i] < -TINY)
            r0 = qmin / dqv[i];
        if (dqv[i] > TINY)
            r0 = qmax / dqv[i];
        r = fmin(r0, r);
    }

    phi = fmin(r * beta_w, 1.0);
    dqv[0] = dqv[0] * phi;
    dqv[1] = dqv[1] * phi;
    dqv[2] = dqv[2] * phi;
}

__device__ void __rotate(double *q, double n1, double n2)
{
    double q1 = q[1];
    double q2 = q[2];
    q[1] = n1 * q1 + n2 * q2;
    q[2] = -n2 * q1 + n1 * q2;
}

__device__ void __flux_function_central(double *q_left, double *q_right,
                            double h_left, double h_right,
                            double hle, double hre,
                            double n1, double n2,
                            double epsilon,
                            double ze,
                            double limiting_threshold,
                            double g,
                            double *edgeflux, double *max_speed,
                            double *pressure_flux, double hc,
                            double hc_n,
                            int64_t low_froude)
{
    int64_t i;
    double uh_left, vh_left, u_left;
    double uh_right, vh_right, u_right;
    double s_min, s_max, soundspeed_left, soundspeed_right;
    double denom, inverse_denominator;
    double tmp, local_fr, v_right, v_left;
    double q_left_rotated[3], q_right_rotated[3], flux_right[3], flux_left[3];

    if (h_left == 0. && h_right == 0.) {
        edgeflux[0] = 0.0;
        edgeflux[1] = 0.0;
        edgeflux[2] = 0.0;
        *max_speed = 0.0;
        *pressure_flux = 0.;
        return;
    }

    q_left_rotated[0] = q_left[0];
    q_right_rotated[0] = q_right[0];
    q_left_rotated[1] = q_left[1];
    q_right_rotated[1] = q_right[1];
    q_left_rotated[2] = q_left[2];
    q_right_rotated[2] = q_right[2];

    __rotate(q_left_rotated, n1, n2);
    __rotate(q_right_rotated, n1, n2);

    uh_left = q_left_rotated[1];
    vh_left = q_left_rotated[2];
    if (hle > 0.0) {
        tmp = 1.0 / hle;
        u_left = uh_left * tmp;
        uh_left = h_left * u_left;
        v_left = vh_left * tmp;
        vh_left = h_left * tmp * vh_left;
    } else {
        u_left = 0.; uh_left = 0.; vh_left = 0.; v_left = 0.;
    }

    uh_right = q_right_rotated[1];
    vh_right = q_right_rotated[2];
    if (hre > 0.0) {
        tmp = 1.0 / hre;
        u_right = uh_right * tmp;
        uh_right = h_right * u_right;
        v_right = vh_right * tmp;
        vh_right = h_right * tmp * vh_right;
    } else {
        u_right = 0.; uh_right = 0.; vh_right = 0.; v_right = 0.;
    }

    soundspeed_left = sqrt(g * h_left);
    soundspeed_right = sqrt(g * h_right);

    if (low_froude == 1) {
        local_fr = sqrt(
            fmax(0.001, fmin(1.0,
                (u_right * u_right + u_left * u_left + v_right * v_right + v_left * v_left) /
                    (soundspeed_left * soundspeed_left + soundspeed_right * soundspeed_right + 1.0e-10))));
    } else if (low_froude == 2) {
        local_fr = sqrt((u_right * u_right + u_left * u_left + v_right * v_right + v_left * v_left) /
                        (soundspeed_left * soundspeed_left + soundspeed_right * soundspeed_right + 1.0e-10));
        local_fr = sqrt(fmin(1.0, 0.01 + fmax(local_fr - 0.01, 0.0)));
    } else {
        local_fr = 1.0;
    }

    s_max = fmax(u_left + soundspeed_left, u_right + soundspeed_right);
    if (s_max < 0.0) s_max = 0.0;

    s_min = fmin(u_left - soundspeed_left, u_right - soundspeed_right);
    if (s_min > 0.0) s_min = 0.0;

    flux_left[0] = u_left * h_left;
    flux_left[1] = u_left * uh_left;
    flux_left[2] = u_left * vh_left;

    flux_right[0] = u_right * h_right;
    flux_right[1] = u_right * uh_right;
    flux_right[2] = u_right * vh_right;

    denom = s_max - s_min;
    if (denom < epsilon) {
        edgeflux[0] = 0.0;
        edgeflux[1] = 0.0;
        edgeflux[2] = 0.0;
        *max_speed = 0.0;
        *pressure_flux = 0.5 * g * 0.5 * (h_left * h_left + h_right * h_right);
    } else {
        *max_speed = fmax(s_max, -s_min);
        inverse_denominator = 1.0 / fmax(denom, 1.0e-100);
        for (i = 0; i < 3; i++) {
            edgeflux[i] = s_max * flux_left[i] - s_min * flux_right[i];
            if (i == 0)
                edgeflux[i] += (s_max * s_min) * (fmax(q_right_rotated[i], ze) - fmax(q_left_rotated[i], ze));
            if (i == 1)
                edgeflux[i] += local_fr * (s_max * s_min) * (uh_right - uh_left);
            if (i == 2)
                edgeflux[i] += local_fr * (s_max * s_min) * (vh_right - vh_left);
            edgeflux[i] *= inverse_denominator;
        }
        *pressure_flux = 0.5 * g * (s_max * h_left * h_left - s_min * h_right * h_right) * inverse_denominator;
        __rotate(edgeflux, n1, -n2);
    }
}

__device__ double __adjust_edgeflux_with_weir(double *edgeflux,
                                   double h_left, double h_right,
                                   double g, double weir_height,
                                   double Qfactor,
                                   double s1, double s2,
                                   double h1, double h2,
                                   double *max_speed_local)
{
    double rw, rw2;
    double rwRat, hdRat, hdWrRat, scaleFlux, minhd, maxhd;
    double w1, w2, newFlux;
    double two = 2.0;
    double three = 3.0;
    double twothirds = (two / three);

    if ((h_left <= 0.0) && (h_right <= 0.0))
        return 0;

    minhd = fmin(h_left, h_right);
    maxhd = fmax(h_left, h_right);
    rw = Qfactor * twothirds * maxhd * sqrt(twothirds * g * maxhd);
    rw2 = Qfactor * twothirds * minhd * sqrt(twothirds * g * minhd);
    rwRat = rw2 / fmax(rw, 1.0e-100);
    hdRat = minhd / fmax(maxhd, 1.0e-100);
    hdWrRat = minhd / fmax(weir_height, 1.0e-100);

    rw = rw * pow(1.0 - rwRat, 0.385);

    if (h_right > h_left)
        rw *= -1.0;

    if ((hdRat < s2) & (hdWrRat < h2)) {
        w1 = fmin(fmax(hdRat - s1, 0.) / (s2 - s1), 1.0);
        w2 = fmin(fmax(hdWrRat - h1, 0.) / (h2 - h1), 1.0);
        newFlux = (rw * (1.0 - w1) + w1 * edgeflux[0]) * (1.0 - w2) + w2 * edgeflux[0];

        if (fabs(edgeflux[0]) > 1.0e-100)
            scaleFlux = newFlux / edgeflux[0];
        else
            scaleFlux = 0.;

        scaleFlux = fmax(scaleFlux, 0.);
        edgeflux[0] = newFlux;
        edgeflux[1] *= fmin(scaleFlux, 10.);
        edgeflux[2] *= fmin(scaleFlux, 10.);
    }

    if (fabs(edgeflux[0]) > 0.)
        *max_speed_local = sqrt(g * (maxhd + weir_height)) + fabs(edgeflux[0] / (maxhd + 1.0e-12));

    return 0;
}

__device__ double atomicMin_double(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*) address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(fmin(val, __longlong_as_double(assumed))));
    } while (assumed != old);
    return __longlong_as_double(old);
}

__device__ int64_t _gradient(double x0, double y0,
              double x1, double y1,
              double x2, double y2,
              double q0, double q1, double q2,
              double *a, double *b) {
    double det;
    det = (y2-y0)*(x1-x0) - (y1-y0)*(x2-x0);
    *a = (y2-y0)*(q1-q0) - (y1-y0)*(q2-q0);
    *a /= det;
    *b = (x1-x0)*(q2-q0) - (x2-x0)*(q1-q0);
    *b /= det;
    return 0;
}

/* ==========================================================================
 * Global kernels
 * ========================================================================== */

/* --- Flux computation --- */
__global__ void _cuda_compute_fluxes_loop(
                                    double* timestep_k_array,
                                    double* boundary_flux_sum_k_array,

                                    double* max_speed,
                                    double* stage_explicit_update,
                                    double* xmom_explicit_update,
                                    double* ymom_explicit_update,

                                    double* stage_centroid_values,
                                    double* height_centroid_values,
                                    double* bed_centroid_values,

                                    double* stage_edge_values,
                                    double* xmom_edge_values,
                                    double* ymom_edge_values,
                                    double* bed_edge_values,
                                    double* height_edge_values,

                                    double* stage_boundary_values,
                                    double* xmom_boundary_values,
                                    double* ymom_boundary_values,

                                    double* areas,
                                    double* normals,
                                    double* edgelengths,
                                    double* radii,
                                    int64_t*   tri_full_flag,
                                    int64_t*   neighbours,
                                    int64_t*   neighbour_edges,
                                    int64_t*   edge_flux_type,
                                    int64_t*   edge_river_wall_counter,
                                    double* riverwall_elevation,
                                    int64_t*   riverwall_rowIndex,
                                    double* riverwall_hydraulic_properties,

                                    int64_t   number_of_elements,
                                    int64_t   substep_count,
                                    int64_t   ncol_riverwall_hydraulic_properties,
                                    double epsilon,
                                    double g,
                                    int64_t   low_froude,
                                    double limiting_threshold)
{
    int64_t k, i, ki, ki2, n, m, nm;
    int64_t RiverWall_count;
    double max_speed_local, length, inv_area, zl, zr;
    double h_left, h_right;
    double z_half, ql[3], pressuregrad_work;
    double qr[3], edgeflux[3], edge_timestep, normal_x, normal_y;
    double hle, hre, zc, zc_n, Qfactor, s1, s2, h1, h2, pressure_flux, hc, hc_n;
    double h_left_tmp, h_right_tmp, weir_height;

    double local_stage_explicit_update;
    double local_xmom_explicit_update;
    double local_ymom_explicit_update;
    double local_timestep;
    double local_boundary_flux_sum;
    double speed_max_last;

    k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < number_of_elements) {
        local_stage_explicit_update = 0.0;
        local_xmom_explicit_update  = 0.0;
        local_ymom_explicit_update  = 0.0;
        local_timestep = 1.0e+100;
        local_boundary_flux_sum = 0.0;
        speed_max_last = 0.0;

        for (i = 0; i < 3; i++) {
            ki = 3*k+i;
            ki2 = 2*ki;

            ql[0] = stage_edge_values[ki];
            ql[1] = xmom_edge_values[ki];
            ql[2] = ymom_edge_values[ki];
            zl    = bed_edge_values[ki];
            hle   = height_edge_values[ki];

            hc = height_centroid_values[k];
            zc = bed_centroid_values[k];

            n = neighbours[ki];
            hc_n = hc;
            zc_n = bed_centroid_values[k];
            if (n < 0) {
                m = -n - 1;
                qr[0] = stage_boundary_values[m];
                qr[1] = xmom_boundary_values[m];
                qr[2] = ymom_boundary_values[m];
                zr = zl;
                hre = fmax(qr[0] - zr, 0.0);
            } else {
                hc_n = height_centroid_values[n];
                zc_n = bed_centroid_values[n];
                m = neighbour_edges[ki];
                nm = n * 3 + m;
                qr[0] = stage_edge_values[nm];
                qr[1] = xmom_edge_values[nm];
                qr[2] = ymom_edge_values[nm];
                zr    = bed_edge_values[nm];
                hre   = height_edge_values[nm];
            }

            z_half = fmax(zl, zr);

            if (edge_flux_type[ki] == 1) {
                RiverWall_count = edge_river_wall_counter[ki];
                z_half = fmax(riverwall_elevation[RiverWall_count - 1], z_half);
            }

            h_left = fmax(hle + zl - z_half, 0.);
            h_right = fmax(hre + zr - z_half, 0.);

            normal_x = normals[ki2];
            normal_y = normals[ki2 + 1];

            __flux_function_central(ql, qr,
                                    h_left, h_right,
                                    hle, hre,
                                    normal_x, normal_y,
                                    epsilon, z_half, limiting_threshold, g,
                                    edgeflux, &max_speed_local, &pressure_flux,
                                    hc, hc_n, low_froude);

            if (edge_flux_type[ki] == 1) {
                RiverWall_count = edge_river_wall_counter[ki];
                int64_t ii = riverwall_rowIndex[RiverWall_count - 1] * ncol_riverwall_hydraulic_properties;
                Qfactor = riverwall_hydraulic_properties[ii];
                s1 = riverwall_hydraulic_properties[ii + 1];
                s2 = riverwall_hydraulic_properties[ii + 2];
                h1 = riverwall_hydraulic_properties[ii + 3];
                h2 = riverwall_hydraulic_properties[ii + 4];
                weir_height = fmax(riverwall_elevation[RiverWall_count - 1] - fmin(zl, zr), 0.);
                h_left_tmp = fmax(stage_centroid_values[k] - z_half, 0.);
                if (n >= 0) {
                    h_right_tmp = fmax(stage_centroid_values[n] - z_half, 0.);
                } else {
                    h_right_tmp = fmax(hc_n + zr - z_half, 0.);
                }
                if (riverwall_elevation[RiverWall_count - 1] > fmax(zc, zc_n)) {
                    __adjust_edgeflux_with_weir(edgeflux, h_left_tmp, h_right_tmp, g,
                                                weir_height, Qfactor,
                                                s1, s2, h1, h2, &max_speed_local);
                }
            }

            length = edgelengths[ki];
            edgeflux[0] = -edgeflux[0] * length;
            edgeflux[1] = -edgeflux[1] * length;
            edgeflux[2] = -edgeflux[2] * length;

            pressuregrad_work = length * (-g * 0.5 * (h_left * h_left - hle * hle - (hle + hc) * (zl - zc)) + pressure_flux);

            if (substep_count == 0) {
                edge_timestep = radii[k] * 1.0 / fmax(max_speed_local, epsilon);
                if (tri_full_flag[k] == 1) {
                    if (max_speed_local > epsilon) {
                        local_timestep = fmin(local_timestep, edge_timestep);
                        speed_max_last = fmax(speed_max_last, max_speed_local);
                    }
                }
            }

            local_stage_explicit_update += edgeflux[0];
            local_xmom_explicit_update  += edgeflux[1];
            local_ymom_explicit_update  += edgeflux[2];

            if (((n < 0) & (tri_full_flag[k] == 1)) | ((n >= 0) && ((tri_full_flag[k] == 1) & (tri_full_flag[n] == 0)))) {
                local_boundary_flux_sum += edgeflux[0];
            }

            local_xmom_explicit_update -= normals[ki2] * pressuregrad_work;
            local_ymom_explicit_update -= normals[ki2 + 1] * pressuregrad_work;
        }

        if (substep_count == 0)
            max_speed[k] = speed_max_last;

        inv_area = 1.0 / areas[k];
        stage_explicit_update[k] = local_stage_explicit_update * inv_area;
        xmom_explicit_update[k]  = local_xmom_explicit_update * inv_area;
        ymom_explicit_update[k]  = local_ymom_explicit_update * inv_area;

        boundary_flux_sum_k_array[k] = local_boundary_flux_sum;
        timestep_k_array[k] = local_timestep;
    }
}

/* --- Extrapolation loop 1: depth + velocity conversion --- */
__global__ void _cuda_extrapolate_second_order_edge_sw_loop1(
                                                      double* stage_centroid_values,
                                                      double* xmom_centroid_values,
                                                      double* ymom_centroid_values,
                                                      double* height_centroid_values,
                                                      double* bed_centroid_values,
                                                      double* x_centroid_work,
                                                      double* y_centroid_work,
                                                      double minimum_allowed_height,
                                                      int64_t number_of_elements,
                                                      int64_t extrapolate_velocity_second_order)
{
    int64_t k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < number_of_elements) {
        double dk = fmax(stage_centroid_values[k] - bed_centroid_values[k], 0.0);
        height_centroid_values[k] = dk;
        x_centroid_work[k] = 0.0;
        y_centroid_work[k] = 0.0;

        if (dk <= minimum_allowed_height) {
            x_centroid_work[k] = 0.0;
            xmom_centroid_values[k] = 0.0;
            y_centroid_work[k] = 0.0;
            ymom_centroid_values[k] = 0.0;
        }

        if (extrapolate_velocity_second_order == 1) {
            if (dk > minimum_allowed_height) {
                double dk_inv = 1.0 / dk;
                x_centroid_work[k] = xmom_centroid_values[k];
                xmom_centroid_values[k] = xmom_centroid_values[k] * dk_inv;
                y_centroid_work[k] = ymom_centroid_values[k];
                ymom_centroid_values[k] = ymom_centroid_values[k] * dk_inv;
            }
        }
    }
}

/* --- Extrapolation loop 2: 2nd order + TVD limiting --- */
__device__ void __calc_edge_values(double beta_tmp, double cv_k, double cv_k0, double cv_k1, double cv_k2,
                        double dxv0, double dxv1, double dxv2, double dyv0, double dyv1, double dyv2,
                        double dx1, double dx2, double dy1, double dy2, double inv_area2,
                        double *edge_values)
{
    double dqv[3];
    double dq0, dq1, dq2;
    double a, b;
    double qmin, qmax;

    if (beta_tmp > 0.) {
        dq0 = cv_k0 - cv_k;
        dq1 = cv_k1 - cv_k0;
        dq2 = cv_k2 - cv_k0;

        a = dy2 * dq1 - dy1 * dq2;
        a *= inv_area2;
        b = dx1 * dq2 - dx2 * dq1;
        b *= inv_area2;

        dqv[0] = a * dxv0 + b * dyv0;
        dqv[1] = a * dxv1 + b * dyv1;
        dqv[2] = a * dxv2 + b * dyv2;

        __find_qmin_and_qmax(dq0, dq1, dq2, &qmin, &qmax);
        __limit_gradient(dqv, qmin, qmax, beta_tmp);

        edge_values[0] = cv_k + dqv[0];
        edge_values[1] = cv_k + dqv[1];
        edge_values[2] = cv_k + dqv[2];
    } else {
        edge_values[0] = cv_k;
        edge_values[1] = cv_k;
        edge_values[2] = cv_k;
    }
}

__global__ void _cuda_extrapolate_second_order_edge_sw_loop2(
                                                             double* stage_edge_values,
                                                             double* xmom_edge_values,
                                                             double* ymom_edge_values,
                                                             double* height_edge_values,
                                                             double* bed_edge_values,

                                                             double* stage_centroid_values,
                                                             double* xmom_centroid_values,
                                                             double* ymom_centroid_values,
                                                             double* height_centroid_values,
                                                             double* bed_centroid_values,

                                                             double* x_centroid_work,
                                                             double* y_centroid_work,

                                                             int64_t*   number_of_boundaries,
                                                             double* centroid_coordinates,
                                                             double* edge_coordinates,
                                                             int64_t*   surrogate_neighbours,

                                                             double beta_w_dry,
                                                             double beta_w,
                                                             double beta_uh_dry,
                                                             double beta_uh,
                                                             double beta_vh_dry,
                                                             double beta_vh,

                                                             double minimum_allowed_height,
                                                             int64_t   number_of_elements,
                                                             int64_t   extrapolate_velocity_second_order)
{
    int64_t k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < number_of_elements) {
        double a, b;
        int64_t k0, k1, k2, k3, k6, coord_index, i;
        double x, y, x0, y0, x1, y1, x2, y2, xv0, yv0, xv1, yv1, xv2, yv2;
        double dx1, dx2, dy1, dy2, dxv0, dxv1, dxv2, dyv0, dyv1, dyv2, dq1, area2, inv_area2;
        double dqv[3], qmin, qmax, hmin, hmax;
        double hc, h0, h1, h2, beta_tmp, hfactor;
        double dk, dk_inv, a_tmp, b_tmp, c_tmp, d_tmp;
        double edge_values[3];
        double cv_k, cv_k0, cv_k1, cv_k2;

        a_tmp = 0.3;
        b_tmp = 0.1;
        c_tmp = 1.0 / (a_tmp - b_tmp);
        d_tmp = 1.0 - (c_tmp * a_tmp);

        k2 = k * 2;
        k3 = k * 3;
        k6 = k * 6;

        xv0 = edge_coordinates[k6 + 0];
        yv0 = edge_coordinates[k6 + 1];
        xv1 = edge_coordinates[k6 + 2];
        yv1 = edge_coordinates[k6 + 3];
        xv2 = edge_coordinates[k6 + 4];
        yv2 = edge_coordinates[k6 + 5];

        x = centroid_coordinates[k2 + 0];
        y = centroid_coordinates[k2 + 1];

        dxv0 = xv0 - x;
        dxv1 = xv1 - x;
        dxv2 = xv2 - x;
        dyv0 = yv0 - y;
        dyv1 = yv1 - y;
        dyv2 = yv2 - y;

        k0 = surrogate_neighbours[k3 + 0];
        k1 = surrogate_neighbours[k3 + 1];
        k2 = surrogate_neighbours[k3 + 2];

        coord_index = 2 * k0;
        x0 = centroid_coordinates[coord_index + 0];
        y0 = centroid_coordinates[coord_index + 1];
        coord_index = 2 * k1;
        x1 = centroid_coordinates[coord_index + 0];
        y1 = centroid_coordinates[coord_index + 1];
        coord_index = 2 * k2;
        x2 = centroid_coordinates[coord_index + 0];
        y2 = centroid_coordinates[coord_index + 1];

        dx1 = x1 - x0;
        dx2 = x2 - x0;
        dy1 = y1 - y0;
        dy2 = y2 - y0;
        area2 = dy2 * dx1 - dy1 * dx2;

        if (((height_centroid_values[k0] < minimum_allowed_height) | (k0 == k)) &
            ((height_centroid_values[k1] < minimum_allowed_height) | (k1 == k)) &
            ((height_centroid_values[k2] < minimum_allowed_height) | (k2 == k))) {
            x_centroid_work[k] = 0.;
            xmom_centroid_values[k] = 0.;
            y_centroid_work[k] = 0.;
            ymom_centroid_values[k] = 0.;
        }

        if (number_of_boundaries[k] == 3) {
            stage_edge_values[k3 + 0] = stage_centroid_values[k];
            stage_edge_values[k3 + 1] = stage_centroid_values[k];
            stage_edge_values[k3 + 2] = stage_centroid_values[k];
            xmom_edge_values[k3 + 0] = xmom_centroid_values[k];
            xmom_edge_values[k3 + 1] = xmom_centroid_values[k];
            xmom_edge_values[k3 + 2] = xmom_centroid_values[k];
            ymom_edge_values[k3 + 0] = ymom_centroid_values[k];
            ymom_edge_values[k3 + 1] = ymom_centroid_values[k];
            ymom_edge_values[k3 + 2] = ymom_centroid_values[k];
            dk = height_centroid_values[k];
            height_edge_values[k3 + 0] = dk;
            height_edge_values[k3 + 1] = dk;
            height_edge_values[k3 + 2] = dk;
        } else if (number_of_boundaries[k] <= 1) {
            hc = height_centroid_values[k];
            h0 = height_centroid_values[k0];
            h1 = height_centroid_values[k1];
            h2 = height_centroid_values[k2];
            hmin = fmin(fmin(h0, fmin(h1, h2)), hc);
            hmax = fmax(fmax(h0, fmax(h1, h2)), hc);
            hfactor = fmax(0., fmin(c_tmp * fmax(hmin, 0.0) / fmax(hc, 1.0e-06) + d_tmp,
                                    fmin(c_tmp * fmax(hc, 0.) / fmax(hmax, 1.0e-06) + d_tmp, 1.0)));
            hfactor = fmin(1.2 * fmax(hmin - minimum_allowed_height, 0.) / (fmax(hmin, 0.) + 1. * minimum_allowed_height), hfactor);
            inv_area2 = 1.0 / area2;

            // stage
            beta_tmp = beta_w_dry + (beta_w - beta_w_dry) * hfactor;
            cv_k  = stage_centroid_values[k];
            cv_k0 = stage_centroid_values[k0];
            cv_k1 = stage_centroid_values[k1];
            cv_k2 = stage_centroid_values[k2];
            __calc_edge_values(beta_tmp, cv_k, cv_k0, cv_k1, cv_k2,
                               dxv0, dxv1, dxv2, dyv0, dyv1, dyv2,
                               dx1, dx2, dy1, dy2, inv_area2, edge_values);
            stage_edge_values[k3 + 0] = edge_values[0];
            stage_edge_values[k3 + 1] = edge_values[1];
            stage_edge_values[k3 + 2] = edge_values[2];

            // height
            cv_k  = height_centroid_values[k];
            cv_k0 = height_centroid_values[k0];
            cv_k1 = height_centroid_values[k1];
            cv_k2 = height_centroid_values[k2];
            __calc_edge_values(beta_tmp, cv_k, cv_k0, cv_k1, cv_k2,
                               dxv0, dxv1, dxv2, dyv0, dyv1, dyv2,
                               dx1, dx2, dy1, dy2, inv_area2, edge_values);
            height_edge_values[k3 + 0] = edge_values[0];
            height_edge_values[k3 + 1] = edge_values[1];
            height_edge_values[k3 + 2] = edge_values[2];

            // xmom
            beta_tmp = beta_uh_dry + (beta_uh - beta_uh_dry) * hfactor;
            cv_k  = xmom_centroid_values[k];
            cv_k0 = xmom_centroid_values[k0];
            cv_k1 = xmom_centroid_values[k1];
            cv_k2 = xmom_centroid_values[k2];
            __calc_edge_values(beta_tmp, cv_k, cv_k0, cv_k1, cv_k2,
                               dxv0, dxv1, dxv2, dyv0, dyv1, dyv2,
                               dx1, dx2, dy1, dy2, inv_area2, edge_values);
            xmom_edge_values[k3 + 0] = edge_values[0];
            xmom_edge_values[k3 + 1] = edge_values[1];
            xmom_edge_values[k3 + 2] = edge_values[2];

            // ymom
            beta_tmp = beta_vh_dry + (beta_vh - beta_vh_dry) * hfactor;
            cv_k  = ymom_centroid_values[k];
            cv_k0 = ymom_centroid_values[k0];
            cv_k1 = ymom_centroid_values[k1];
            cv_k2 = ymom_centroid_values[k2];
            __calc_edge_values(beta_tmp, cv_k, cv_k0, cv_k1, cv_k2,
                               dxv0, dxv1, dxv2, dyv0, dyv1, dyv2,
                               dx1, dx2, dy1, dy2, inv_area2, edge_values);
            ymom_edge_values[k3 + 0] = edge_values[0];
            ymom_edge_values[k3 + 1] = edge_values[1];
            ymom_edge_values[k3 + 2] = edge_values[2];
        } else {
            // 2 boundaries: single neighbour gradient
            for (k2 = k3; k2 < k3 + 3; k2++) {
                if (surrogate_neighbours[k2] != k) break;
            }
            k1 = surrogate_neighbours[k2];
            coord_index = 2 * k1;
            x1 = centroid_coordinates[coord_index + 0];
            y1 = centroid_coordinates[coord_index + 1];
            dx1 = x1 - x;
            dy1 = y1 - y;
            area2 = dx1 * dx1 + dy1 * dy1;
            dx2 = 1.0 / area2;
            dy2 = dx2 * dy1;
            dx2 *= dx1;

            // stage
            dq1 = stage_centroid_values[k1] - stage_centroid_values[k];
            a = dq1 * dx2;
            b = dq1 * dy2;
            dqv[0] = a * dxv0 + b * dyv0;
            dqv[1] = a * dxv1 + b * dyv1;
            dqv[2] = a * dxv2 + b * dyv2;
            if (dq1 >= 0.0) { qmin = 0.0; qmax = dq1; }
            else { qmin = dq1; qmax = 0.0; }
            __limit_gradient(dqv, qmin, qmax, beta_w);
            stage_edge_values[k3 + 0] = stage_centroid_values[k] + dqv[0];
            stage_edge_values[k3 + 1] = stage_centroid_values[k] + dqv[1];
            stage_edge_values[k3 + 2] = stage_centroid_values[k] + dqv[2];

            // height
            dq1 = height_centroid_values[k1] - height_centroid_values[k];
            a = dq1 * dx2;
            b = dq1 * dy2;
            dqv[0] = a * dxv0 + b * dyv0;
            dqv[1] = a * dxv1 + b * dyv1;
            dqv[2] = a * dxv2 + b * dyv2;
            if (dq1 >= 0.0) { qmin = 0.0; qmax = dq1; }
            else { qmin = dq1; qmax = 0.0; }
            __limit_gradient(dqv, qmin, qmax, beta_w);
            height_edge_values[k3 + 0] = height_centroid_values[k] + dqv[0];
            height_edge_values[k3 + 1] = height_centroid_values[k] + dqv[1];
            height_edge_values[k3 + 2] = height_centroid_values[k] + dqv[2];

            // xmom
            dq1 = xmom_centroid_values[k1] - xmom_centroid_values[k];
            a = dq1 * dx2;
            b = dq1 * dy2;
            dqv[0] = a * dxv0 + b * dyv0;
            dqv[1] = a * dxv1 + b * dyv1;
            dqv[2] = a * dxv2 + b * dyv2;
            if (dq1 >= 0.0) { qmin = 0.0; qmax = dq1; }
            else { qmin = dq1; qmax = 0.0; }
            __limit_gradient(dqv, qmin, qmax, beta_w);
            xmom_edge_values[k3 + 0] = xmom_centroid_values[k] + dqv[0];
            xmom_edge_values[k3 + 1] = xmom_centroid_values[k] + dqv[1];
            xmom_edge_values[k3 + 2] = xmom_centroid_values[k] + dqv[2];

            // ymom
            dq1 = ymom_centroid_values[k1] - ymom_centroid_values[k];
            a = dq1 * dx2;
            b = dq1 * dy2;
            dqv[0] = a * dxv0 + b * dyv0;
            dqv[1] = a * dxv1 + b * dyv1;
            dqv[2] = a * dxv2 + b * dyv2;
            if (dq1 >= 0.0) { qmin = 0.0; qmax = dq1; }
            else { qmin = dq1; qmax = 0.0; }
            __limit_gradient(dqv, qmin, qmax, beta_w);
            ymom_edge_values[k3 + 0] = ymom_centroid_values[k] + dqv[0];
            ymom_edge_values[k3 + 1] = ymom_centroid_values[k] + dqv[1];
            ymom_edge_values[k3 + 2] = ymom_centroid_values[k] + dqv[2];
        }

        // Convert velocity back to momenta if needed
        if (extrapolate_velocity_second_order == 1) {
            for (i = 0; i < 3; i++) {
                dk = height_edge_values[k3 + i];
                xmom_edge_values[k3 + i] *= dk;
                ymom_edge_values[k3 + i] *= dk;
            }
        }

        // Bed elevation at edges
        bed_edge_values[k3 + 0] = stage_edge_values[k3 + 0] - height_edge_values[k3 + 0];
        bed_edge_values[k3 + 1] = stage_edge_values[k3 + 1] - height_edge_values[k3 + 1];
        bed_edge_values[k3 + 2] = stage_edge_values[k3 + 2] - height_edge_values[k3 + 2];
    }
}

/* --- Extrapolation loop 3: velocity to momentum --- */
__global__ void _cuda_extrapolate_second_order_edge_sw_loop3(
                                                      double* xmom_centroid_values,
                                                      double* ymom_centroid_values,
                                                      double* x_centroid_work,
                                                      double* y_centroid_work,
                                                      int64_t extrapolate_velocity_second_order,
                                                      int64_t number_of_elements)
{
    int64_t k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < number_of_elements) {
        if (extrapolate_velocity_second_order == 1) {
            xmom_centroid_values[k] = x_centroid_work[k];
            ymom_centroid_values[k] = y_centroid_work[k];
        }
    }
}

/* --- Extrapolation loop 4: edge to vertex --- */
__global__ void _cuda_extrapolate_second_order_edge_sw_loop4(
                                              double* stage_edge_values,
                                              double* xmom_edge_values,
                                              double* ymom_edge_values,
                                              double* height_edge_values,
                                              double* bed_edge_values,
                                              double* stage_vertex_values,
                                              double* height_vertex_values,
                                              double* xmom_vertex_values,
                                              double* ymom_vertex_values,
                                              double* bed_vertex_values,
                                              int64_t   number_of_elements)
{
    int64_t k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < number_of_elements) {
        int64_t k3 = k * 3;
        stage_vertex_values[k3 + 0] = stage_edge_values[k3 + 1] + stage_edge_values[k3 + 2] - stage_edge_values[k3 + 0];
        stage_vertex_values[k3 + 1] = stage_edge_values[k3 + 0] + stage_edge_values[k3 + 2] - stage_edge_values[k3 + 1];
        stage_vertex_values[k3 + 2] = stage_edge_values[k3 + 0] + stage_edge_values[k3 + 1] - stage_edge_values[k3 + 2];

        height_vertex_values[k3 + 0] = height_edge_values[k3 + 1] + height_edge_values[k3 + 2] - height_edge_values[k3 + 0];
        height_vertex_values[k3 + 1] = height_edge_values[k3 + 0] + height_edge_values[k3 + 2] - height_edge_values[k3 + 1];
        height_vertex_values[k3 + 2] = height_edge_values[k3 + 0] + height_edge_values[k3 + 1] - height_edge_values[k3 + 2];

        xmom_vertex_values[k3 + 0] = xmom_edge_values[k3 + 1] + xmom_edge_values[k3 + 2] - xmom_edge_values[k3 + 0];
        xmom_vertex_values[k3 + 1] = xmom_edge_values[k3 + 0] + xmom_edge_values[k3 + 2] - xmom_edge_values[k3 + 1];
        xmom_vertex_values[k3 + 2] = xmom_edge_values[k3 + 0] + xmom_edge_values[k3 + 1] - xmom_edge_values[k3 + 2];

        ymom_vertex_values[k3 + 0] = ymom_edge_values[k3 + 1] + ymom_edge_values[k3 + 2] - ymom_edge_values[k3 + 0];
        ymom_vertex_values[k3 + 1] = ymom_edge_values[k3 + 0] + ymom_edge_values[k3 + 2] - ymom_edge_values[k3 + 1];
        ymom_vertex_values[k3 + 2] = ymom_edge_values[k3 + 0] + ymom_edge_values[k3 + 1] - ymom_edge_values[k3 + 2];

        bed_vertex_values[k3 + 0] = bed_edge_values[k3 + 1] + bed_edge_values[k3 + 2] - bed_edge_values[k3 + 0];
        bed_vertex_values[k3 + 1] = bed_edge_values[k3 + 0] + bed_edge_values[k3 + 2] - bed_edge_values[k3 + 1];
        bed_vertex_values[k3 + 2] = bed_edge_values[k3 + 0] + bed_edge_values[k3 + 1] - bed_edge_values[k3 + 2];
    }
}

/* --- Update conserved quantities --- */
__global__ void _cuda_update_sw(int64_t number_of_elements,
                                double timestep,
                                double *centroid_values,
                                double *explicit_update,
                                double *semi_implicit_update)
{
    int64_t k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < number_of_elements) {
        double x = centroid_values[k];
        if (x == 0.0) {
            semi_implicit_update[k] = 0.0;
        } else {
            semi_implicit_update[k] /= x;
        }
        centroid_values[k] += timestep * explicit_update[k];
        double denominator = 1.0 - timestep * semi_implicit_update[k];
        if (denominator > 0.0) {
            centroid_values[k] /= denominator;
        }
        semi_implicit_update[k] = 0.0;
    }
}

/* --- Fix negative cells --- */
__global__ void _cuda_fix_negative_cells_sw(
                                              int64_t number_of_elements,
                                              int64_t *tri_full_flag,
                                              double *stage_centroid_values,
                                              double *bed_centroid_values,
                                              double *xmom_centroid_values,
                                              double *ymom_centroid_values,
                                              int64_t *num_negative_cells)
{
    int64_t k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < number_of_elements) {
        int64_t tff = tri_full_flag[k];
        if ((stage_centroid_values[k] - bed_centroid_values[k] < 0.0) && (tff > 0)) {
            atomicAdd(num_negative_cells, 1);
            stage_centroid_values[k] = bed_centroid_values[k];
            xmom_centroid_values[k] = 0.0;
            ymom_centroid_values[k] = 0.0;
        }
    }
}

/* --- Protect against infinitesimal and negative heights --- */
__global__ void _cuda_protect_against_infinitesimal_and_negative_heights(
     double domain_minimum_allowed_height,
     int64_t number_of_elements,
     double* stage_centroid_values,
     double* bed_centroid_values,
     double* xmom_centroid_values,
     double* areas,
     double* stage_vertex_values)
{
    double mass_error = 0.;
    int64_t k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < number_of_elements) {
        int64_t k3 = 3 * k;
        double hc = stage_centroid_values[k] - bed_centroid_values[k];
        if (hc < domain_minimum_allowed_height * 1.0) {
            xmom_centroid_values[k] = 0.;
            if (hc <= 0.0) {
                double bmin = bed_centroid_values[k];
                if (stage_centroid_values[k] < bmin) {
                    mass_error += (bmin - stage_centroid_values[k]) * areas[k];
                    stage_centroid_values[k] = bmin;
                    stage_vertex_values[k3] = bmin;
                    stage_vertex_values[k3 + 1] = bmin;
                    stage_vertex_values[k3 + 2] = bmin;
                }
            }
        }
    }
}

/* --- Manning friction (flat bed) --- */
__global__ void cft_manning_friction_flat(double g, double eps, int64_t N,
        double* w, double* zv,
        double* uh, double* vh,
        double* eta, double* xmom, double* ymom) {

    const double one_third = 1.0/3.0;
    const double seven_thirds = 7.0/3.0;

    int64_t k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < N) {
        if (eta[k] > eps) {
            int64_t k3 = 3 * k;
            double z0 = zv[k3 + 0];
            double z1 = zv[k3 + 1];
            double z2 = zv[k3 + 2];
            double z = (z0 + z1 + z2) * one_third;
            double h = w[k] - z;
            if (h >= eps) {
                double S = -g * eta[k] * eta[k] * sqrt((uh[k] * uh[k] + vh[k] * vh[k]));
                S /= pow(h, seven_thirds);
                xmom[k] += S * uh[k];
                ymom[k] += S * vh[k];
            }
        }
    }
}

/* --- Manning friction (sloped bed) --- */
__global__ void cft_manning_friction_sloped(double g, double eps, int64_t N,
        double* x, double* w, double* zv,
        double* uh, double* vh,
        double* eta, double* xmom_update, double* ymom_update) {

    const double one_third = 1.0/3.0;
    const double seven_thirds = 7.0/3.0;

    int64_t k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < N) {
        if (eta[k] > eps) {
            int64_t k3 = 3 * k;
            double z0 = zv[k3 + 0];
            double z1 = zv[k3 + 1];
            double z2 = zv[k3 + 2];
            int64_t k6 = 6 * k;
            double x0 = x[k6 + 0], y0 = x[k6 + 1];
            double x1 = x[k6 + 2], y1 = x[k6 + 3];
            double x2 = x[k6 + 4], y2 = x[k6 + 5];
            double zx, zy;
            _gradient(x0, y0, x1, y1, x2, y2, z0, z1, z2, &zx, &zy);
            double zs = sqrt(1.0 + zx * zx + zy * zy);
            double z = (z0 + z1 + z2) * one_third;
            double h = w[k] - z;
            if (h >= eps) {
                double S = -g * eta[k] * eta[k] * zs * sqrt((uh[k] * uh[k] + vh[k] * vh[k]));
                S /= pow(h, seven_thirds);
                xmom_update[k] += S * uh[k];
                ymom_update[k] += S * vh[k];
            }
        }
    }
}

/* ==========================================================================
 * C wrapper functions — called from pybind11 / C code
 * ========================================================================== */

#include "hydro_cuda.h"

static const int THREADS_PER_BLOCK = 128;

extern "C" {

hydro_int cuda_compute_fluxes(
    double* timestep_k_array,
    double* boundary_flux_sum_k_array,
    double* max_speed,
    double* stage_explicit_update,
    double* xmom_explicit_update,
    double* ymom_explicit_update,
    double* stage_centroid_values,
    double* height_centroid_values,
    double* bed_centroid_values,
    double* stage_edge_values,
    double* xmom_edge_values,
    double* ymom_edge_values,
    double* bed_edge_values,
    double* height_edge_values,
    double* stage_boundary_values,
    double* xmom_boundary_values,
    double* ymom_boundary_values,
    double* areas,
    double* normals,
    double* edgelengths,
    double* radii,
    int64_t* tri_full_flag,
    int64_t* neighbours,
    int64_t* neighbour_edges,
    int64_t* edge_flux_type,
    int64_t* edge_river_wall_counter,
    double* riverwall_elevation,
    int64_t* riverwall_rowIndex,
    double* riverwall_hydraulic_properties,
    int64_t number_of_elements,
    int64_t substep_count,
    int64_t ncol_riverwall,
    double epsilon,
    double g,
    int64_t low_froude,
    double limiting_threshold,
    double* timestep_out)
{
    dim3 blocks((number_of_elements + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
    dim3 threads(THREADS_PER_BLOCK);

    _cuda_compute_fluxes_loop<<<blocks, threads>>>(
        timestep_k_array, boundary_flux_sum_k_array,
        max_speed, stage_explicit_update, xmom_explicit_update, ymom_explicit_update,
        stage_centroid_values, height_centroid_values, bed_centroid_values,
        stage_edge_values, xmom_edge_values, ymom_edge_values, bed_edge_values, height_edge_values,
        stage_boundary_values, xmom_boundary_values, ymom_boundary_values,
        areas, normals, edgelengths, radii,
        tri_full_flag, neighbours, neighbour_edges, edge_flux_type, edge_river_wall_counter,
        riverwall_elevation, riverwall_rowIndex, riverwall_hydraulic_properties,
        number_of_elements, substep_count, ncol_riverwall,
        epsilon, g, low_froude, limiting_threshold);

    cudaDeviceSynchronize();
    return (hydro_int)cudaGetLastError();
}

void cuda_extrapolate_loop1(
    double* stage_centroid_values,
    double* xmom_centroid_values,
    double* ymom_centroid_values,
    double* height_centroid_values,
    double* bed_centroid_values,
    double* x_centroid_work,
    double* y_centroid_work,
    double minimum_allowed_height,
    int64_t number_of_elements,
    int64_t extrapolate_velocity_second_order)
{
    dim3 blocks((number_of_elements + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
    dim3 threads(THREADS_PER_BLOCK);
    _cuda_extrapolate_second_order_edge_sw_loop1<<<blocks, threads>>>(
        stage_centroid_values, xmom_centroid_values, ymom_centroid_values,
        height_centroid_values, bed_centroid_values,
        x_centroid_work, y_centroid_work,
        minimum_allowed_height, number_of_elements, extrapolate_velocity_second_order);
    cudaDeviceSynchronize();
}

void cuda_extrapolate_loop2(
    double* stage_edge_values,
    double* xmom_edge_values,
    double* ymom_edge_values,
    double* height_edge_values,
    double* bed_edge_values,
    double* stage_centroid_values,
    double* xmom_centroid_values,
    double* ymom_centroid_values,
    double* height_centroid_values,
    double* bed_centroid_values,
    double* x_centroid_work,
    double* y_centroid_work,
    int64_t* number_of_boundaries,
    double* centroid_coordinates,
    double* edge_coordinates,
    int64_t* surrogate_neighbours,
    double beta_w_dry, double beta_w,
    double beta_uh_dry, double beta_uh,
    double beta_vh_dry, double beta_vh,
    double minimum_allowed_height,
    int64_t number_of_elements,
    int64_t extrapolate_velocity_second_order)
{
    dim3 blocks((number_of_elements + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
    dim3 threads(THREADS_PER_BLOCK);
    _cuda_extrapolate_second_order_edge_sw_loop2<<<blocks, threads>>>(
        stage_edge_values, xmom_edge_values, ymom_edge_values,
        height_edge_values, bed_edge_values,
        stage_centroid_values, xmom_centroid_values, ymom_centroid_values,
        height_centroid_values, bed_centroid_values,
        x_centroid_work, y_centroid_work,
        number_of_boundaries, centroid_coordinates, edge_coordinates, surrogate_neighbours,
        beta_w_dry, beta_w, beta_uh_dry, beta_uh, beta_vh_dry, beta_vh,
        minimum_allowed_height, number_of_elements, extrapolate_velocity_second_order);
    cudaDeviceSynchronize();
}

void cuda_extrapolate_loop3(
    double* xmom_centroid_values,
    double* ymom_centroid_values,
    double* x_centroid_work,
    double* y_centroid_work,
    int64_t extrapolate_velocity_second_order,
    int64_t number_of_elements)
{
    dim3 blocks((number_of_elements + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
    dim3 threads(THREADS_PER_BLOCK);
    _cuda_extrapolate_second_order_edge_sw_loop3<<<blocks, threads>>>(
        xmom_centroid_values, ymom_centroid_values,
        x_centroid_work, y_centroid_work,
        extrapolate_velocity_second_order, number_of_elements);
    cudaDeviceSynchronize();
}

void cuda_extrapolate_loop4(
    double* stage_edge_values,
    double* xmom_edge_values,
    double* ymom_edge_values,
    double* height_edge_values,
    double* bed_edge_values,
    double* stage_vertex_values,
    double* height_vertex_values,
    double* xmom_vertex_values,
    double* ymom_vertex_values,
    double* bed_vertex_values,
    int64_t number_of_elements)
{
    dim3 blocks((number_of_elements + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
    dim3 threads(THREADS_PER_BLOCK);
    _cuda_extrapolate_second_order_edge_sw_loop4<<<blocks, threads>>>(
        stage_edge_values, xmom_edge_values, ymom_edge_values,
        height_edge_values, bed_edge_values,
        stage_vertex_values, height_vertex_values,
        xmom_vertex_values, ymom_vertex_values, bed_vertex_values,
        number_of_elements);
    cudaDeviceSynchronize();
}

void cuda_update_sw(
    int64_t number_of_elements,
    double timestep,
    double* centroid_values,
    double* explicit_update,
    double* semi_implicit_update)
{
    dim3 blocks((number_of_elements + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
    dim3 threads(THREADS_PER_BLOCK);
    _cuda_update_sw<<<blocks, threads>>>(
        number_of_elements, timestep, centroid_values, explicit_update, semi_implicit_update);
    cudaDeviceSynchronize();
}

void cuda_fix_negative_cells(
    int64_t number_of_elements,
    int64_t* tri_full_flag,
    double* stage_centroid_values,
    double* bed_centroid_values,
    double* xmom_centroid_values,
    double* ymom_centroid_values,
    int64_t* num_negative_cells)
{
    dim3 blocks((number_of_elements + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
    dim3 threads(THREADS_PER_BLOCK);
    _cuda_fix_negative_cells_sw<<<blocks, threads>>>(
        number_of_elements, tri_full_flag,
        stage_centroid_values, bed_centroid_values,
        xmom_centroid_values, ymom_centroid_values,
        num_negative_cells);
    cudaDeviceSynchronize();
}

void cuda_protect(
    double domain_minimum_allowed_height,
    int64_t number_of_elements,
    double* stage_centroid_values,
    double* bed_centroid_values,
    double* xmom_centroid_values,
    double* areas,
    double* stage_vertex_values)
{
    dim3 blocks((number_of_elements + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
    dim3 threads(THREADS_PER_BLOCK);
    _cuda_protect_against_infinitesimal_and_negative_heights<<<blocks, threads>>>(
        domain_minimum_allowed_height, number_of_elements,
        stage_centroid_values, bed_centroid_values,
        xmom_centroid_values, areas, stage_vertex_values);
    cudaDeviceSynchronize();
}

void cuda_manning_friction_flat(
    double g, double eps, int64_t N,
    double* w, double* zv,
    double* uh, double* vh,
    double* eta, double* xmom, double* ymom)
{
    dim3 blocks((N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
    dim3 threads(THREADS_PER_BLOCK);
    cft_manning_friction_flat<<<blocks, threads>>>(g, eps, N, w, zv, uh, vh, eta, xmom, ymom);
    cudaDeviceSynchronize();
}

void cuda_manning_friction_sloped(
    double g, double eps, int64_t N,
    double* x, double* w, double* zv,
    double* uh, double* vh,
    double* eta, double* xmom_update, double* ymom_update)
{
    dim3 blocks((N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
    dim3 threads(THREADS_PER_BLOCK);
    cft_manning_friction_sloped<<<blocks, threads>>>(g, eps, N, x, w, zv, uh, vh, eta, xmom_update, ymom_update);
    cudaDeviceSynchronize();
}

} // extern "C"
