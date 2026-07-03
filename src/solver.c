/**
 * solver.c — Conjugate Gradient solver for sparse SPD systems
 *
 * Solves A*x = b and (I - dt*L)*x = b (parabolic solve).
 */

#include "hydro/solver.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* =========================================================================
 * Vector helpers (avoiding external BLAS dependency)
 * ========================================================================= */

static double dot(const double* a, const double* b, hydro_int n) {
    double s = 0.0;
    for (hydro_int i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}

static double norm2(const double* a, hydro_int n) {
    return sqrt(dot(a, a, n));
}

static void axpy(double alpha, const double* x, double* y, hydro_int n) {
    for (hydro_int i = 0; i < n; i++) y[i] += alpha * x[i];
}

static void scal(double alpha, double* x, hydro_int n) {
    for (hydro_int i = 0; i < n; i++) x[i] *= alpha;
}

static void copy_vec(const double* src, double* dst, hydro_int n) {
    memcpy(dst, src, (size_t)n * sizeof(double));
}

/* =========================================================================
 * Standard CG: solve A * x = b
 * ========================================================================= */

int hydro_conjugate_gradient_solve(
    const hydro_sparse_csr_t* A,
    const double* b,
    double* x,
    hydro_int N,
    hydro_int max_iter,
    double tol,
    hydro_cg_stats_t* stats)
{
    if (N <= 0 || !A || !b || !x) return -1;

    double* r = (double*)calloc((size_t)N, sizeof(double));
    double* p = (double*)calloc((size_t)N, sizeof(double));
    double* Ap = (double*)calloc((size_t)N, sizeof(double));
    if (!r || !p || !Ap) { free(r); free(p); free(Ap); return -1; }

    /* r = b - A*x */
    hydro_sparse_csr_mv(A, x, r);
    for (hydro_int i = 0; i < N; i++) r[i] = b[i] - r[i];

    double bnorm = norm2(b, N);
    double tol_abs = tol * bnorm;
    if (bnorm < 1e-30) tol_abs = tol;

    double rnorm = norm2(r, N);

    /* Initial check */
    if (rnorm <= tol_abs) {
        if (stats) {
            stats->iterations = 0;
            stats->residual = rnorm;
            stats->converged = 1;
        }
        free(r); free(p); free(Ap);
        return 0;
    }

    /* p = r */
    copy_vec(r, p, N);

    hydro_int iter;
    for (iter = 0; iter < max_iter; iter++) {
        /* Ap = A * p */
        hydro_sparse_csr_mv(A, p, Ap);

        double pAp = dot(p, Ap, N);
        if (fabs(pAp) < 1e-30) break;

        double alpha = dot(r, r, N) / pAp;

        /* x = x + alpha * p */
        axpy(alpha, p, x, N);

        /* r_new = r - alpha * Ap */
        double rsq_old = dot(r, r, N);
        axpy(-alpha, Ap, r, N);
        double rsq_new = dot(r, r, N);

        rnorm = sqrt(rsq_new);
        if (rnorm <= tol_abs) {
            iter++;
            break;
        }

        /* p = r + beta * p, beta = rsq_new / rsq_old */
        double beta = rsq_new / rsq_old;
        scal(beta, p, N);
        axpy(1.0, r, p, N);
    }

    if (stats) {
        stats->iterations = iter;
        stats->residual = rnorm;
        stats->converged = (rnorm <= tol_abs) ? 1 : 0;
    }

    free(r); free(p); free(Ap);
    return 0;
}

/* =========================================================================
 * Parabolic CG: solve (I - dt * L) * x = b
 *
 * The matrix-vector product is y = (I - dt*L) * x = x - dt * L*x.
 * ========================================================================= */

int hydro_parabolic_cg_solve(
    const hydro_sparse_csr_t* L,
    double dt,
    const double* b,
    double* x,
    hydro_int N,
    hydro_int max_iter,
    double tol,
    hydro_cg_stats_t* stats)
{
    if (N <= 0 || !L || !b || !x) return -1;

    double* r = (double*)calloc((size_t)N, sizeof(double));
    double* p = (double*)calloc((size_t)N, sizeof(double));
    double* Ap = (double*)calloc((size_t)N, sizeof(double));
    if (!r || !p || !Ap) { free(r); free(p); free(Ap); return -1; }

    /* r = b - (I - dt*L)*x = b - x + dt*L*x */
    /* Lx = L * x */
    hydro_sparse_csr_mv(L, x, Ap);
    for (hydro_int i = 0; i < N; i++) {
        r[i] = b[i] - x[i] + dt * Ap[i];
    }

    double bnorm = norm2(b, N);
    double tol_abs = tol * bnorm;
    if (bnorm < 1e-30) tol_abs = tol;

    double rnorm = norm2(r, N);
    if (rnorm <= tol_abs) {
        if (stats) {
            stats->iterations = 0;
            stats->residual = rnorm;
            stats->converged = 1;
        }
        free(r); free(p); free(Ap);
        return 0;
    }

    copy_vec(r, p, N);
    double rsq_old = dot(r, r, N);

    hydro_int iter;
    for (iter = 0; iter < max_iter; iter++) {
        /* Ap = (I - dt*L) * p = p - dt * L*p */
        hydro_sparse_csr_mv(L, p, Ap);
        for (hydro_int i = 0; i < N; i++) {
            Ap[i] = p[i] - dt * Ap[i];
        }

        double pAp = dot(p, Ap, N);
        if (fabs(pAp) < 1e-30) break;

        double alpha = rsq_old / pAp;

        axpy(alpha, p, x, N);
        axpy(-alpha, Ap, r, N);

        double rsq_new = dot(r, r, N);
        rnorm = sqrt(rsq_new);
        if (rnorm <= tol_abs) {
            iter++;
            break;
        }

        double beta = rsq_new / rsq_old;
        scal(beta, p, N);
        axpy(1.0, r, p, N);
        rsq_old = rsq_new;
    }

    if (stats) {
        stats->iterations = iter;
        stats->residual = rnorm;
        stats->converged = (rnorm <= tol_abs) ? 1 : 0;
    }

    free(r); free(p); free(Ap);
    return 0;
}
