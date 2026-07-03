/**
 * sparse.c — CSR sparse matrix operations
 */

#include "hydro/sparse.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifdef _OPENMP
#include <omp.h>
#endif

hydro_sparse_csr_t* hydro_sparse_csr_create(
    hydro_int N, hydro_int M, hydro_int nnz_per_row)
{
    hydro_sparse_csr_t* A = (hydro_sparse_csr_t*)calloc(1, sizeof(hydro_sparse_csr_t));
    if (!A) return NULL;

    hydro_int nnz = N * nnz_per_row;
    A->N = N;
    A->M = M;
    A->nnz = nnz;
    A->data   = (double*)calloc((size_t)nnz, sizeof(double));
    A->colind = (hydro_int*)calloc((size_t)nnz, sizeof(hydro_int));
    A->rowptr = (hydro_int*)calloc((size_t)(N + 1), sizeof(hydro_int));

    /* Uniform row pointers: row i has entries [i*nnz_per_row, (i+1)*nnz_per_row) */
    for (hydro_int i = 0; i <= N; i++) {
        A->rowptr[i] = i * nnz_per_row;
    }

    return A;
}

hydro_sparse_csr_t* hydro_sparse_csr_create_raw(
    hydro_int N, hydro_int M, hydro_int nnz)
{
    hydro_sparse_csr_t* A = (hydro_sparse_csr_t*)calloc(1, sizeof(hydro_sparse_csr_t));
    if (!A) return NULL;

    A->N = N;
    A->M = M;
    A->nnz = nnz;
    A->data   = (double*)calloc((size_t)nnz, sizeof(double));
    A->colind = (hydro_int*)calloc((size_t)nnz, sizeof(hydro_int));
    A->rowptr = (hydro_int*)calloc((size_t)(N + 1), sizeof(hydro_int));

    return A;
}

void hydro_sparse_csr_destroy(hydro_sparse_csr_t* A) {
    if (A) {
        free(A->data);
        free(A->colind);
        free(A->rowptr);
        free(A);
    }
}

void hydro_sparse_csr_mv(
    const hydro_sparse_csr_t* A,
    const double* x,
    double* y)
{
    hydro_int N = A->N;

    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (hydro_int i = 0; i < N; i++) {
        double sum = 0.0;
        hydro_int start = A->rowptr[i];
        hydro_int end   = A->rowptr[i + 1];
        for (hydro_int p = start; p < end; p++) {
            sum += A->data[p] * x[A->colind[p]];
        }
        y[i] = sum;
    }
}

void hydro_sparse_csr_identity_plus_dt_mv(
    const hydro_sparse_csr_t* A,
    double dt,
    const double* x,
    double* y)
{
    hydro_int N = A->N;

    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (hydro_int i = 0; i < N; i++) {
        double sum = x[i];  /* identity: y = x + dt * A * x */
        hydro_int start = A->rowptr[i];
        hydro_int end   = A->rowptr[i + 1];
        for (hydro_int p = start; p < end; p++) {
            sum += dt * A->data[p] * x[A->colind[p]];
        }
        y[i] = sum;
    }
}
