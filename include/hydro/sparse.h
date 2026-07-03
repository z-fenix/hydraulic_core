#ifndef HYDRO_SPARSE_H
#define HYDRO_SPARSE_H

#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* =========================================================================
 * Sparse matrix in CSR (Compressed Sparse Row) format
 *
 * Stores a sparse N x M matrix with `nnz` non-zero entries:
 *   data[nnz]   — values
 *   colind[nnz] — column indices
 *   rowptr[N+1] — row start pointers (row i: data[rowptr[i]]..data[rowptr[i+1]-1])
 * ========================================================================= */

typedef struct {
    hydro_int N;        /* number of rows */
    hydro_int M;        /* number of columns */
    hydro_int nnz;      /* number of non-zero entries (allocated size) */
    double*   data;     /* [nnz] non-zero values */
    hydro_int* colind;  /* [nnz] column indices */
    hydro_int* rowptr;  /* [N+1] row pointers */
} hydro_sparse_csr_t;

/**
 * Create a CSR matrix with known row pointers.
 * N = number of rows, M = number of columns.
 * nnz_per_row = number of non-zeros per row (must be uniform for now).
 */
hydro_sparse_csr_t* hydro_sparse_csr_create(
    hydro_int N, hydro_int M, hydro_int nnz_per_row);

/**
 * Create a CSR matrix with explicit nnz total and pre-allocated arrays.
 * Caller must set rowptr, colind, data after creation.
 */
hydro_sparse_csr_t* hydro_sparse_csr_create_raw(
    hydro_int N, hydro_int M, hydro_int nnz);

/** Free a CSR matrix. */
void hydro_sparse_csr_destroy(hydro_sparse_csr_t* A);

/**
 * Compute y = A * x.  x and y must be pre-allocated (length M and N
 * respectively).
 */
void hydro_sparse_csr_mv(
    const hydro_sparse_csr_t* A,
    const double* x,
    double* y);

/**
 * Compute y = (I + dt * A) * x.
 * Equivalent to y = x + dt * A*x, but computed in a single pass.
 */
void hydro_sparse_csr_identity_plus_dt_mv(
    const hydro_sparse_csr_t* A,
    double dt,
    const double* x,
    double* y);

#ifdef __cplusplus
}
#endif

#endif /* HYDRO_SPARSE_H */
