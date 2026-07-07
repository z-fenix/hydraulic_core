#ifndef HYDRO_SOLVER_H
#define HYDRO_SOLVER_H

#include "types.h"
#include "sparse.h"

#ifdef __cplusplus
extern "C" {

#endif

/* =========================================================================
 * Conjugate Gradient Solver
 *
 * Solves A * x = b for symmetric positive-definite A.
 * ========================================================================= */

/* Result statistics from CG solve */
typedef struct
{
    hydro_int iterations; /* number of iterations performed */
    double residual; /* final residual norm */
    int converged; /* 1 if converged within tolerance */
} hydro_cg_stats_t;

/**
 * Solve A * x = b using conjugate gradient.
 *
 * @param A        CSR sparse matrix (must be SPD)
 * @param b        Right-hand side vector [N]
 * @param x        Initial guess on input, solution on output [N]
 * @param N        Number of unknowns
 * @param max_iter Maximum iterations
 * @param tol      Convergence tolerance (norm of residual / norm of b)
 * @param stats    Output statistics (may be NULL)
 * @return 0 on success, -1 on failure
 */
int hydro_conjugate_gradient_solve(
    const hydro_sparse_csr_t* A,
    const double* b,
    double* x,
    hydro_int N,
    hydro_int max_iter,
    double tol,
    hydro_cg_stats_t* stats);

/**
 * Solve (I - dt * L) * x = b using conjugate gradient.
 *
 * The linear operator is A = I - dt * L, where L is the input matrix.
 * This avoids explicitly forming the dense identity matrix.
 *
 * @param L        CSR matrix representing the operator
 * @param dt       Timestep (multiplies L)
 * @param b        Right-hand side vector [N]
 * @param x        Initial guess on input, solution on output [N]
 * @param N        Number of unknowns
 * @param max_iter Maximum iterations
 * @param tol      Convergence tolerance
 * @param stats    Output statistics (may be NULL)
 * @return 0 on success
 */
int hydro_parabolic_cg_solve(
    const hydro_sparse_csr_t* L,
    double dt,
    const double* b,
    double* x,
    hydro_int N,
    hydro_int max_iter,
    double tol,
    hydro_cg_stats_t* stats);

#ifdef __cplusplus
}
#endif

#endif /* HYDRO_SOLVER_H */
