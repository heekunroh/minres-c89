#include "minres.h"

static double shiftm = 0.0f;
static double pertm = 0.0f;

/*
 *+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 *     File minrestest.c
 *
 *     These routines are for testing MINRES.
 *
 *     15 Jul 2003: Derived from symmlqtest.f.
 *     10 Feb 2023: Ported from F77 to C89 by Heekun Roh
 *                  (heekunroh@kaist.ac.kr)
 *+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 */

void Aprod(uint64_t n, double* x, double* y) {
  /*
   *     ------------------------------------------------------------------
   *     Aprod  computes  y = A*x  for some matrix  A.
   *     This is a simple example for testing MINRES.
   *     ------------------------------------------------------------------
   */
  uint64_t i;
  double d;

  for (i = 0; i < n; i++) {
    d = i + 1; /* *1.1 */
    d = d / n;
    y[i] = d * x[i];
  }
}

/*
 *+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 */

void Msolve(uint64_t n, double* x, double* y) {
  /*
   *     ------------------------------------------------------------------
   *     Msolve  solves  M*y = x  for some symmetric positive-definite
   *     matrix  M.
   *     This is a simple example for testing MINRES.
   *     shiftm will be the same as shift in MINRES.
   *
   *     If pertbn = 0, the preconditioner will be exact, so
   *     MINRES should require either one or two iterations,
   *     depending on whether (A - shift*I) is positive definite or not.
   *
   *     If pertbn is nonzero, somewhat more iterations will be required.
   *     ------------------------------------------------------------------
   */
  uint64_t i;
  double d;

  for (i = 0; i < n; i++) {
    d = i + 1; /* *1.1 */
    d = d / n;
    d = fabs(d - shiftm);
    if ((i + 1) % 10 == 0) d = d + pertm;
    y[i] = x[i] / d;
  }
}

void test(uint64_t n, uint8_t precon, double shift, double pertbn) {
  /*
   *     ------------------------------------------------------------------
   *     test   solves sets up and solves a system (A - shift * I)x = b,
   *     using Aprod to define A and Msolve to define a preconditioner.
   *     ------------------------------------------------------------------
   */

  uint8_t checkA;
  double b[100], r1[100], r2[100], v[100], w[100], w1[100], w2[100], x[100],
      y[100], xtrue[100];

  double one, two;
  one = 1.0f;
  two = 2.0f;

  uint64_t itnlim, itn, nout;
  int8_t istop;
  double rtol, Anorm, Acond, rnorm, ynorm, r1norm, enorm, etol;

  uint64_t i;

  shiftm = shift;
  pertm = fabs(pertbn);
  nout = 6;
  printf(
      " ------------------------------------------------------\n"
      " Test of MINRES.\n"
      " ------------------------------------------------------\n"
      " shift = %12.4f      pertbn =%12.4f\n\n\n",
      shiftm, pertbn);

  /*
   *     Set the true solution and the rhs
   *     so that  (A - shift*I) * xtrue = b.
   */
  for (i = 0; i < n; i++) {
    xtrue[i] = n - i;
  }

  Aprod(n, xtrue, b);
  daxpy_c(n, (-shift), xtrue, b);

  /*
   *     Set other parameters and solve.
   */
  checkA = MINRES_TRUE;
  itnlim = n * 2;
  rtol = 1.0e-12f;

  minres(n, b, r1, r2, v, w, w1, w2, x, y, Aprod, Msolve, checkA, precon, shift,
         nout, itnlim, rtol, &istop, &itn, &Anorm, &Acond, &rnorm, &ynorm);
  /*
   *     Compute the final residual,  r1 = b - (A - shift*I)*x.
   */
  Aprod(n, x, y);
  daxpy_c(n, (-shift), x, y);

  for (i = 0; i < n; i++) {
    r1[i] = b[i] - y[i];
  }

  r1norm = dnrm2_c(n, r1);

  printf(" Final residual = %8.1e\n", r1norm);

  /*
   *     Print the solution and some clue about whether it is OK.
   */

  printf(" Solution x\n");
  for (i = 0; i < n; i++) {
    printf("    %lu\t%14.6e\t", i + 1, x[i]);
    if ((i + 1) % 4 == 0) printf("\n");
  }
  printf("\n");

  for (i = 0; i < n; i++) {
    w[i] = x[i] - xtrue[i];
  }

  enorm = dnrm2_c(n, w) / dnrm2_c(n, xtrue);
  etol = 1.0e-5f;

  if (enorm <= etol) {
    printf(
        " MINRES  appears to be successful.     Relative error in x =%8.1e\n",
        enorm);
  } else {
    printf(" MINRES  appears to have failed.    Relative error in x =%8.1e\n",
           enorm);
  }
}

void main(void) {
  uint8_t normal, precon;
  uint64_t n;
  double zero, one, shift, pertbn;
  zero = 0.0f;
  one = 1.0f;

  normal = MINRES_FALSE;
  precon = MINRES_TRUE;
  shift = 0.25f;
  pertbn = 0.1f;

  /*
   *        Test the unlikely tiny cases that often trip us up.
   */
  n = 1;
  test(n, normal, zero, zero);
  test(n, normal, shift, zero);

  n = 2;
  test(n, normal, zero, zero);
  test(n, normal, shift, zero);

  /*
   *        Test small positive-definite and indefinite systems
   *        without preconditioners.  MINRES should take n iterations.
   */
  n = 50;
  test(n, normal, zero, zero);
  test(n, normal, shift, zero);

  /* if (n <= 5) return;    /* WHILE TESTING */

  n = 1;
  test(n, precon, zero, zero);
  test(n, precon, shift, zero);

  n = 2;
  test(n, precon, zero, zero);
  test(n, precon, shift, zero);

  n = 50;
  test(n, precon, zero, zero);
  test(n, precon, shift, zero);

  /*
   *        pertbn makes the preconditioners incorrect in n/10 entries.
   *        MINRES should take about n/10 iterations.
   */
  test(n, precon, zero, pertbn);
  test(n, precon, shift, pertbn);
}