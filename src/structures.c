/**
 * structures.c — Hydraulic structure discharge computations
 *
 * Ported from ANUGA:
 *   anuga/structures/boyd_box_operator.py
 *   anuga/structures/boyd_pipe_operator.py
 *   anuga/structures/weir_orifice_trapezoid_operator.py
 *   anuga/structures/inlet_operator.py
 *
 * Implements inlet control, outlet control, and critical depth
 * for culverts and weirs using Boyd (1987) methodology.
 */

#include "hydro/structures.h"
#include <math.h>
#include <string.h>

/* =========================================================================
 * Internal helpers
 * ========================================================================= */

/* Effective blockage factor for box/pipes */
static double blockage_factor(double blockage) {
    if (blockage >= 1.0) return 0.0;
    if (blockage <= 0.0) return 1.0;
    return 1.0 - blockage;
}

/* Effective blockage factor for pipes (nonlinear at high blockage) */
static double pipe_blockage_factor(double blockage) {
    if (blockage >= 1.0) return 0.0;
    if (blockage <= 0.0) return 1.0;
    if (blockage > 0.9) {
        return 3.333 - 3.333 * blockage;
    }
    return 1.0 - 0.40123*blockage - 0.37684*blockage*blockage;
}

/* =========================================================================
 * Boyd Box Culvert Discharge
 * ========================================================================= */

int hydro_boyd_box_discharge(
    double g,
    double width, double height,
    double barrels, double blockage,
    double losses, double manning,
    double culvert_length,
    int use_velocity_head,
    double inlet_energy, double outlet_energy,
    double inlet_depth, double outlet_depth,
    double* Q, double* velocity, double* flow_area) {

    if (width <= 0 || barrels <= 0) return -1;
    if (height <= 0) height = width;

    double bf = blockage_factor(blockage);
    if (bf < 1e-15) {
        *Q = 0.0;
        *velocity = 0.0;
        *flow_area = 0.0;
        return 0;
    }

    /* Driving energy = upstream - downstream head */
    double driving_energy = inlet_energy - outlet_energy;
    if (driving_energy <= 0.0) {
        *Q = 0.0;
        *velocity = 0.0;
        *flow_area = 0.0;
        return 0;
    }

    double sqrt_g = sqrt(g);
    double eff_width = bf * width;
    double depth = inlet_depth;  /* water depth above inlet invert */

    /* Inlet control: unsubmerged (weir) and submerged (orifice) */
    double Q_weir, Q_orifice;

    if (use_velocity_head) {
        Q_weir = 0.544 * sqrt_g * eff_width * barrels *
                 pow(driving_energy, 1.50);
        Q_orifice = 0.702 * sqrt_g * eff_width * barrels *
                    pow(depth, 0.89) * pow(driving_energy, 0.61);
    } else {
        /* Without velocity head, use outlet head as driving head */
        double head_diff = inlet_energy - outlet_energy;
        Q_weir = 0.544 * sqrt_g * eff_width * barrels *
                 pow(head_diff, 1.50);
        Q_orifice = 0.702 * sqrt_g * eff_width * barrels *
                    pow(depth, 0.89) * pow(head_diff, 0.61);
    }

    /* Inlet control Q = min(weir, orifice) */
    double Q_inlet = (Q_weir < Q_orifice) ? Q_weir : Q_orifice;

    /* Critical depth */
    double dcrit = pow(Q_inlet*Q_inlet / (g * eff_width*eff_width * barrels*barrels), 1.0/3.0);

    /* Flow area and wetted perimeter based on critical depth */
    double A, P;
    if (dcrit >= height) {
        /* Full-flowing */
        A = eff_width * height * barrels;
        P = 2.0 * (eff_width + height) * barrels;
    } else {
        /* Partial flow */
        A = eff_width * dcrit * barrels;
        P = (eff_width + 2.0 * dcrit) * barrels;
    }

    /* Hydraulic radius */
    double R = (P > 0) ? A / P : 0.0;

    /* Outlet control */
    double delta_total_energy = inlet_energy - outlet_energy;
    if (delta_total_energy < driving_energy) {
        double sum_loss = losses;
        if (manning > 0 && R > 0 && culvert_length > 0) {
            sum_loss += 2.0 * g * manning * manning * culvert_length /
                        pow(R, 4.0/3.0);
        }

        double Q_outlet_tailwater = 0.0;
        if (sum_loss > 1e-15) {
            double culvert_velocity = sqrt(delta_total_energy /
                (sum_loss / (2.0 * g)));
            Q_outlet_tailwater = A * culvert_velocity;
        }

        *Q = (Q_inlet < Q_outlet_tailwater) ? Q_inlet : Q_outlet_tailwater;
    } else {
        *Q = Q_inlet;
    }

    /* Barrel velocity with protection against division by zero */
    double vp = 1.0e-6;  /* velocity_protection */
    *flow_area = A;
    *velocity = *Q / (A + vp / A);

    return 0;
}

/* =========================================================================
 * Boyd Pipe Culvert Discharge
 * ========================================================================= */

int hydro_boyd_pipe_discharge(
    double g,
    double diameter,
    double barrels, double blockage,
    double losses, double manning,
    double culvert_length,
    int use_velocity_head,
    double inlet_energy, double outlet_energy,
    double inlet_depth, double outlet_depth,
    double* Q, double* velocity, double* flow_area) {

    if (diameter <= 0 || barrels <= 0) return -1;

    double bf = pipe_blockage_factor(blockage);
    if (bf < 1e-15) {
        *Q = 0.0;
        *velocity = 0.0;
        *flow_area = 0.0;
        return 0;
    }

    double driving_energy = inlet_energy - outlet_energy;
    if (driving_energy <= 0.0) {
        *Q = 0.0;
        *velocity = 0.0;
        *flow_area = 0.0;
        return 0;
    }

    double sqrt_g = sqrt(g);
    double eff_D = bf * diameter;
    double depth = inlet_depth;

    /* Inlet control: unsubmerged and submerged for circular pipe */
    double Q_weir = barrels * 0.421 * sqrt_g * pow(eff_D, 0.87) *
                    pow(driving_energy, 1.63);
    double Q_orifice = barrels * 0.530 * sqrt_g * pow(eff_D, 1.87) *
                       pow(driving_energy, 0.63);

    double Q_inlet = (Q_weir < Q_orifice) ? Q_weir : Q_orifice;

    /* Critical depth (two formulations from Boyd) */
    double dcrit1 = (eff_D / 1.26) *
        pow(Q_inlet / (sqrt_g * pow(eff_D, 2.5)), 1.0/3.75);
    double dcrit2 = (eff_D / 0.95) *
        pow(Q_inlet / (sqrt_g * pow(eff_D, 2.5)), 1.0/1.95);

    double dcrit = (dcrit1 / eff_D > 0.85) ? dcrit2 : dcrit1;

    /* Circular geometry: angle from depth */
    double alpha = 2.0 * acos(1.0 - 2.0 * dcrit / eff_D);

    double A = barrels * eff_D*eff_D / 8.0 * (alpha - sin(alpha));
    double P = barrels * alpha * eff_D / 2.0;
    double R = (P > 0) ? A / P : 0.0;

    /* Outlet control */
    double delta_total_energy = inlet_energy - outlet_energy;
    if (delta_total_energy < driving_energy) {
        double sum_loss = losses;
        if (manning > 0 && R > 0 && culvert_length > 0) {
            sum_loss += 2.0 * g * manning * manning * culvert_length /
                        pow(R, 4.0/3.0);
        }

        double Q_outlet = 0.0;
        if (sum_loss > 1e-15) {
            double cv = sqrt(delta_total_energy /
                (sum_loss / (2.0 * g)));
            Q_outlet = A * cv;
        }

        *Q = (Q_inlet < Q_outlet) ? Q_inlet : Q_outlet;
    } else {
        *Q = Q_inlet;
    }

    /* Barrel velocity */
    double vp = 1.0e-6;
    *flow_area = A;
    *velocity = *Q / (A + vp / A);

    return 0;
}

/* =========================================================================
 * Weir/Orifice Trapezoid Discharge
 * ========================================================================= */

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
    double* Q, double* velocity, double* flow_area) {

    if (width <= 0 || barrels <= 0) return -1;

    double bf = blockage_factor(blockage);
    if (bf < 1e-15) {
        *Q = 0.0;
        *velocity = 0.0;
        *flow_area = 0.0;
        return 0;
    }

    double driving_energy = inlet_energy - outlet_energy;
    if (driving_energy <= 0.0) {
        *Q = 0.0;
        *velocity = 0.0;
        *flow_area = 0.0;
        return 0;
    }

    double sqrt_g = sqrt(g);
    double depth = inlet_depth;

    /* Inlet control for trapezoidal section */
    double zsum = z1 + z2;
    double top_width = 2.0*width + depth*zsum;

    /* Unsubmerged (weir) — broad-crested weir formula */
    double Q_weir = 1.7 * bf * barrels *
        (top_width / 2.0) * pow(driving_energy, 1.50);

    /* Submerged (orifice) */
    double A_orifice = 0.5 * depth * (2.0*width + depth*zsum);
    double Q_orifice = 0.8 * bf * barrels * sqrt_g *
        A_orifice * pow(driving_energy, 0.50);

    double Q_inlet = (Q_weir < Q_orifice) ? Q_weir : Q_orifice;

    /* Critical depth: Newton-Raphson iteration solving
     *   Ac^1.5 / sqrt(Tc) - Q/sqrt(g) = 0
     * where Ac = bf*d*(width + 0.5*zsum*d)
     *       Tc = bf*(width + zsum*d)
     */
    double dcrit = depth * 0.5;  /* initial guess */
    for (int iter = 0; iter < 20; iter++) {
        double A_crit = barrels * bf * dcrit * (width + 0.5*zsum*dcrit);
        double T_crit = barrels * bf * (width + zsum*dcrit);
        double f_val = pow(A_crit, 1.5) / sqrt(T_crit + 1e-15) - Q_inlet/sqrt_g;
        /* Numerical derivative */
        double eps_h = dcrit * 1e-6 + 1e-10;
        double A_crit2 = barrels * bf * (dcrit+eps_h) * (width + 0.5*zsum*(dcrit+eps_h));
        double T_crit2 = barrels * bf * (width + zsum*(dcrit+eps_h));
        double f_val2 = pow(A_crit2, 1.5) / sqrt(T_crit2 + 1e-15) - Q_inlet/sqrt_g;
        double df = (f_val2 - f_val) / eps_h;

        if (fabs(df) < 1e-15) break;
        dcrit -= f_val / df;
        if (dcrit < 0) dcrit = eps_h;
        if (fabs(f_val) < 1e-12) break;
    }

    /* Flow area */
    double A = barrels * bf * width * dcrit + 0.5*zsum*dcrit*dcrit;
    double P = barrels * (width + dcrit * sqrt(1.0 + z1*z1) + dcrit * sqrt(1.0 + z2*z2));
    double R = (P > 0) ? A / P : 0.0;

    /* Outlet control */
    double delta_total_energy = inlet_energy - outlet_energy;
    if (delta_total_energy < driving_energy) {
        double sum_loss = losses;
        if (manning > 0 && R > 0 && culvert_length > 0) {
            sum_loss += 2.0 * g * manning * manning * culvert_length /
                        pow(R, 4.0/3.0);
        }

        double Q_outlet = 0.0;
        if (sum_loss > 1e-15) {
            double cv = sqrt(delta_total_energy /
                (sum_loss / (2.0 * g)));
            Q_outlet = A * cv;
        }

        *Q = (Q_inlet < Q_outlet) ? Q_inlet : Q_outlet;
    } else {
        *Q = Q_inlet;
    }

    double vp = 1.0e-6;
    *flow_area = A;
    *velocity = *Q / (A + vp / A);

    return 0;
}

/* =========================================================================
 * Inlet Volume Distribution
 * ========================================================================= */

double hydro_inlet_distribute_volume(
    double volume,
    const hydro_int* indices,
    const double* areas,
    hydro_int num_indices,
    double* stage_c,
    const double* elev_c,
    double* xmom_c,
    double* ymom_c) {

    if (num_indices < 1 || !indices || !areas) return 0.0;

    /* Compute total area and current water volume */
    double total_area = 0.0, current_volume = 0.0;
    for (hydro_int j = 0; j < num_indices; j++) {
        hydro_int k = indices[j];
        double a = areas[j];
        total_area += a;
        double h = stage_c[k] - elev_c[k];
        if (h > 0) current_volume += h * a;
    }

    if (total_area <= 0) return 0.0;

    double new_volume = current_volume + volume;

    if (new_volume < 0.0) {
        /* Draining — cannot drain below bed */
        volume = -current_volume;
        new_volume = 0.0;
    }

    /* Target flat free-surface height above bed */
    double target_height = new_volume / total_area;

    for (hydro_int j = 0; j < num_indices; j++) {
        hydro_int k = indices[j];
        double old_h = stage_c[k] - elev_c[k];
        if (old_h < 0) old_h = 0;

        stage_c[k] = elev_c[k] + target_height;

        /* Scale momentum proportionally */
        if (old_h > 1e-10 && target_height > 0) {
            double scale = target_height / old_h;
            xmom_c[k] *= scale;
            ymom_c[k] *= scale;
        } else if (target_height <= 0) {
            xmom_c[k] = 0.0;
            ymom_c[k] = 0.0;
        }
    }

    return volume;
}
