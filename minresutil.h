/*
 * minresutil.h
 *
 * The program provides necessary linear algebra functions for minres.c
 * This file contains functions similar to Level 1 BLAS, written in C89.
 * Implemented FORTRAN functions in minresblas.f for C89 implementation.
 *
 * Written by Heekun Roh.  06 Feb 2023
 *
 */

#ifndef MINRESUTIL_H
#define MINRESUTIL_H

#include <math.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void daxpy_c(uint64_t n, double a, double *x, double *y);
void dcopy_c(uint64_t n, double *x, double *y);
double ddot_c(uint64_t n, double *x, double *y);
double dnrm2_c(uint64_t n, double *x);
void daxpy2_c(uint64_t n, double a, double *x, double *y, double *z);
void dload2_c(uint64_t n, double a, double *x);
void dscal2_c(uint64_t n, double a, double *x, double *y);

#ifdef __cplusplus
}
#endif

#endif
