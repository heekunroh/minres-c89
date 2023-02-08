/*
 * minresutil.c
 *
 * The program provides necessary linear algebra functions for minres.c
 * This file contains functions similar to Level 1 BLAS, written in C89.
 * Implemented FORTRAN functions in minresblas.f for C89 implementation.
 *
 * Written by Heekun Roh.  06 Feb 2023
 *
 */

#include "minresutil.h"

/* daxpy_c returns double y[n] with y = ax + y */
void daxpy_c(uint64_t n, double a, double *x, double *y) {
  uint64_t i;

  if ((n == 0) || (!x) || (!y)) return;

  for (i = 0; i < n; i++) {
    y[i] = a * x[i] + y[i];
  }
  return;
}

/* dcopy_c returns double y[n] with y = x */
void dcopy_c(uint64_t n, double *x, double *y) {
  uint64_t i;

  if ((n == 0) || (!x) || (!y)) return;

  for (i = 0; i < n; i++) {
    y[i] = x[i];
  }
  return;
}

/* dcopy_c returns the dotproduct of two vectors. dot = x dot y */
double ddot_c(uint64_t n, double *x, double *y) {
  uint64_t i;
  double dot;

  if ((n == 0) || (!x) || (!y)) return 0.0f;

  dot = 0.0f;
  for (i = 0; i < n; i++) {
    dot += x[i] * y[i];
  }
  return dot;
}

/* dnrm2_c returns the Euclidean norm of x, so that dnrm(n, x) = sqrt(x' * x) */
double dnrm2_c(uint64_t n, double *x) {
  uint64_t i;
  double nrm, absxi, ssq, scale, sqt;
  double flmax = 1.0e50;

  if ((n == 0) || (!x)) return 0.0f;

  nrm = 0.0f;
  if (n == 1) {
    return fabs(x[0]);
  } else {
    scale = 0.0f;
    ssq = 1.0f;

    for (i = 0; i < n; i++) {
      if (x[i] != 0.0f) {\
        absxi = fabs(x[i]);

        if (scale < absxi) {
          ssq = 1.0f + ssq * (scale / absxi) * (scale / absxi);
          scale = absxi;
        } else {
          ssq = ssq + (absxi / scale) * (absxi / scale);
        }
      }
    }
    sqt = sqrt(ssq);
    if (scale < flmax / sqt)
      nrm = scale * sqt;
    else
      nrm = flmax;
  }
  return nrm;
}

/* daxpy2_c sets z = a * x + y */
void daxpy2_c(uint64_t n, double a, double *x, double *y, double *z) {
  uint64_t i;

  if ((n == 0) || (!x) || (!y) || (!z)) return;

  for (i = 0; i < n; i++) {
    z[i] = a * x[i] + y[i];
  }
  return;
}

/* dload2_c sets all elements of x with a, i.e. x = {a, a, a,...,a} */
void dload2_c(uint64_t n, double a, double *x) {
  uint64_t i;

  if ((n == 0) || (!x)) return;

  for (i = 0; i < n; i++) {
    x[i] = a;
  }
  return;
}

/* dscal2_c sets y = a * x */
void dscal2_c(uint64_t n, double a, double *x, double *y) {
  uint64_t i;

  if ((n == 0) || (!x) || (!y)) return;

  for (i = 0; i < n; i++) {
    y[i] = a * x[i];
  }
  return;
}
