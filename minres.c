#include "minres.h"

#define DISPLAY_FINAL_RESULTS_AND_EXIT(exit, istop, itn, Anorm, Acond, rnorm, \
                                       ynorm, msg)                            \
  {                                                                           \
    printf(                                                                   \
        "%s      istop  =%3i             itn    =%8ld\n"                      \
        "%s      Anorm  =%12.4e    Acond  =%12.4e\n"                          \
        "%s      rnorm  =%12.4e    ycond  =%12.4e\n",                         \
        exit, *istop, *itn, exit, *Anorm, *Acond, exit, *rnorm, *ynorm);      \
    printf("%s      %s\n\n", exit, msg[*istop + 1]);                          \
    return;                                                                   \
  }

void minres(uint64_t n, double *b, double *r1, double *r2, double *v, double *w,
            double *w1, double *w2, double *x, double *y,
            void (*Aprod)(uint64_t, double *, double *),
            void (*Msolve)(uint64_t, double *, double *), uint8_t checkA,
            uint8_t precon, double shift, uint64_t nout, uint64_t itnlim,
            double rtol, int8_t *istop, uint64_t *itn, double *Anorm,
            double *Acond, double *rnorm, double *ynorm) {
  /*     ------------------------------------------------------------------
   *
   *     MINRES  is designed to solve the system of linear equations
   *
   *                Ax = b
   *
   *     or the least-squares problem
   *
   *         min || Ax - b ||_2,
   *
   *     where A is an n by n symmetric matrix and b is a given vector.
   *     The matrix A may be indefinite and/or singular.
   *
   *     1. If A is known to be positive definite, the Conjugate Gradient
   *        Method might be preferred, since it requires the same number
   *        of iterations as MINRES but less work per iteration.
   *
   *     2. If A is indefinite but Ax = b is known to have a solution
   *        (e.g. if A is nonsingular), SYMMLQ might be preferred,
   *        since it requires the same number of iterations as MINRES
   *        but slightly less work per iteration.
   *
   *     The matrix A is intended to be large and sparse.  It is accessed
   *     by means of a subroutine call of the form
   *     SYMMLQ development:
   *
   *                call Aprod ( n, x, y )
   *
   *     which must return the product y = Ax for any given vector x.
   *
   *
   *     More generally, MINRES is designed to solve the system
   *
   *                (A - shift*I) x = b
   *     or
   *         min || (A - shift*I) x - b ||_2,
   *
   *     where  shift  is a specified scalar value.  Again, the matrix
   *     (A - shift*I) may be indefinite and/or singular.
   *     The work per iteration is very slightly less if  shift = 0.
   *
   *     Note: If  shift  is an approximate eigenvalue of  A
   *     and  b  is an approximate eigenvector,  x  might prove to be
   *     a better approximate eigenvector, as in the methods of
   *     inverse iteration and/or Rayleigh-quotient iteration.
   *     However, we're not yet sure on that -- it may be better
   *     to use SYMMLQ.
   *
   *     A further option is that of preconditioning, which may reduce
   *     the number of iterations required.  If M = C C' is a positive
   *     definite matrix that is known to approximate  (A - shift*I)
   *     in some sense, and if systems of the form  My = x  can be
   *     solved efficiently, the parameters precon and Msolve may be
   *     used (see below).  When  precon = .true., MINRES will
   *     implicitly solve the system of equations
   *
   *             P (A - shift*I) P' xbar  =  P b,
   *
   *     i.e.                  Abar xbar  =  bbar
   *     where                         P  =  C**(-1),
   *                                Abar  =  P (A - shift*I) P',
   *                                bbar  =  P b,
   *
   *     and return the solution       x  =  P' xbar.
   *     The associated residual is rbar  =  bbar - Abar xbar
   *                                      =  P (b - (A - shift*I)x)
   *                                      =  P r.
   *
   *     In the discussion below, eps refers to the machine precision.
   *     eps is computed by MINRES.  A typical value is eps = 2.22d-16
   *     for IEEE double-precision arithmetic.
   *
   *     Parameters
   *     ----------
   *
   *     n       input      The dimension of the matrix A.
   *
   *     b(n)    input      The rhs vector b.
   *
   *     r1(n)   workspace
   *     r2(n)   workspace
   *     v(n)    workspace
   *     w(n)    workspace
   *     w1(n)   workspace
   *     w2(n)   workspace
   *
   *     x(n)    output     Returns the computed solution  x.
   *
   *     y(n)    workspace
   *
   *     Aprod   external   A subroutine defining the matrix A.
   *                        For a given vector x, the statement
   *
   *                              call Aprod ( n, x, y )
   *
   *                        must return the product y = Ax
   *                        without altering the vector x.
   *
   *     Msolve  external   An optional subroutine defining a
   *                        preconditioning matrix M, which should
   *                        approximate (A - shift*I) in some sense.
   *                        M must be positive definite.
   *                        For a given vector x, the statement
   *
   *                              call Msolve( n, x, y )
   *
   *                        must solve the linear system My = x
   *                        without altering the vector x.
   *
   *                        In general, M should be chosen so that Abar
   * has clustered eigenvalues.  For example, if A is positive definite,
   * Abar would ideally be close to a multiple of I. If A or A - shift*I
   * is indefinite, Abar might be close to a multiple of diag( I  -I ).
   *
   *                        NOTE.  The program calling MINRES must declare
   *                        Aprod and Msolve to be external.
   *
   *     checkA  input      If checkA = .true., an extra call of Aprod
   * will be used to check if A is symmetric.  Also, if precon = .true.,
   * an extra call of Msolve will be used to check if M is symmetric.
   *
   *     precon  input      If precon = .true., preconditioning will
   *                        be invoked.  Otherwise, subroutine Msolve
   *                        will not be referenced; in this case the
   *                        actual parameter corresponding to Msolve may
   *                        be the same as that corresponding to Aprod.
   *
   *     shift   input      Should be zero if the system Ax = b is to be
   *                        solved.  Otherwise, it could be an
   *                        approximation to an eigenvalue of A, such as
   *                        the Rayleigh quotient b'Ab / (b'b)
   *                        corresponding to the vector b.
   *                        If b is sufficiently like an eigenvector
   *                        corresponding to an eigenvalue near shift,
   *                        then the computed x may have very large
   *                        components.  When normalized, x may be
   *                        closer to an eigenvector than b.
   *
   *     nout    input      A file number.
   *                        If nout .gt. 0, a summary of the iterations
   *                        will be printed on unit nout.
   *
   *     itnlim  input      An upper limit on the number of iterations.
   *
   *     rtol    input      A user-specified tolerance.  MINRES terminates
   *                        if it appears that norm(rbar) is smaller than
   *                              rtol * norm(Abar) * norm(xbar),
   *                        where rbar is the transformed residual vector,
   *                              rbar = bbar - Abar xbar.
   *
   *                        If shift = 0 and precon = .false., MINRES
   *                        terminates if norm(b - A*x) is smaller than
   *                              rtol * norm(A) * norm(x).
   *
   *     istop   output     An integer giving the reason for
   * termination...
   *
   *              -1        beta2 = 0 in the Lanczos iteration; i.e. the
   *                        second Lanczos vector is zero.  This means the
   *                        rhs is very special.
   *                        If there is no preconditioner, b is an
   *                        eigenvector of A.
   *                        Otherwise (if precon is true), let My = b.
   *                        If shift is zero, y is a solution of the
   *                        generalized eigenvalue problem Ay = lambda My,
   *                        with lambda = alpha1 from the Lanczos vectors.
   *
   *                        In general, (A - shift*I)x = b
   *                        has the solution         x = (1/alpha1) y
   *                        where My = b.
   *
   *               0        b = 0, so the exact solution is x = 0.
   *                        No iterations were performed.
   *
   *               1        Norm(rbar) appears to be less than
   *                        the value  rtol * norm(Abar) * norm(xbar).
   *                        The solution in  x  should be acceptable.
   *
   *               2        Norm(rbar) appears to be less than
   *                        the value  eps * norm(Abar) * norm(xbar).
   *                        This means that the residual is as small as
   *                        seems reasonable on this machine.
   *
   *               3        Norm(Abar) * norm(xbar) exceeds norm(b)/eps,
   *                        which should indicate that x has essentially
   *                        converged to an eigenvector of A
   *                        corresponding to the eigenvalue shift.
   *
   *               4        Acond (see below) has exceeded 0.1/eps, so
   *                        the matrix Abar must be very ill-conditioned.
   *                        x may not contain an acceptable solution.
   *
   *               5        The iteration limit was reached before any of
   *                        the previous criteria were satisfied.
   *
   *               6        The matrix defined by Aprod does not appear
   *                        to be symmetric.
   *                        For certain vectors y = Av and r = Ay, the
   *                        products y'y and r'v differ significantly.
   *
   *               7        The matrix defined by Msolve does not appear
   *                        to be symmetric.
   *                        For vectors satisfying My = v and Mr = y, the
   *                        products y'y and r'v differ significantly.
   *
   *               8        An inner product of the form  x' M**(-1) x
   *                        was not positive, so the preconditioning
   * matrix M does not appear to be positive definite.
   *
   *                        If istop .ge. 5, the final x may not be an
   *                        acceptable solution.
   *
   *     itn     output     The number of iterations performed.
   *
   *     Anorm   output     An estimate of the norm of the matrix operator
   *                        Abar = P (A - shift*I) P',   where P =
   * C**(-1).
   *
   *     Acond   output     An estimate of the condition of Abar above.
   *                        This will usually be a substantial
   *                        under-estimate of the true condition.
   *
   *     rnorm   output     An estimate of the norm of the final
   *                        transformed residual vector,
   *                           P (b  -  (A - shift*I) x).
   *
   *     ynorm   output     An estimate of the norm of xbar.
   *                        This is sqrt( x'Mx ).  If precon is false,
   *                        ynorm is an estimate of norm(x).
   *     ------------------------------------------------------------------
   *
   *
   *     MINRES is an implementation of the algorithm described in
   *     the following reference:
   *
   *     C. C. Paige and M. A. Saunders (1975),
   *     Solution of sparse indefinite systems of linear equations,
   *     SIAM J. Numer. Anal. 12(4), pp. 617-629.
   *     ------------------------------------------------------------------
   *
   *
   *     MINRES development:
   *            1972: First version, similar to original SYMMLQ.
   *                  Later lost @#%*!
   *        Oct 1995: Tried to reconstruct MINRES from
   *                  1995 version of SYMMLQ.
   *     30 May 1999: Need to make it more like LSQR.
   *                  In middle of major overhaul.
   *     19 Jul 2003: Next attempt to reconstruct MINRES.
   *                  Seems to need two vectors more than SYMMLQ.(w1, w2)
   *                  Lanczos is now at the top of the loop,
   *                  so the operator Aprod is called in just one place
   *                  (not counting the initial check for symmetry).
   *     22 Jul 2003: Success at last.  Preconditioning also works.
   *                  minres.f added to http://www.stanford.edu/group/SOL/.
   *     10 Feb 2023: Ported from F77 to C89 by Heekun Roh
   *                  (heekunroh@kaist.ac.kr)
   *
   *     FUTURE WORK: A stopping rule is needed for singular systems.
   *                  We need to estimate ||Ar|| as in LSQR.  This will be
   *                  joint work with Sou Cheng Choi, SCCM, Stanford.
   *                  Note that ||Ar|| small => r is a null vector for A.
   *
   *
   *     Michael A. Saunders           na.msaunders@na-net.ornl.gov
   *     Department of MS&E            saunders@stanford.edu
   *     Stanford University
   *     Stanford, CA 94305-4026       (650) 723-1875
   *     ------------------------------------------------------------------
   *
   *
   *     Subroutines and functions
   *
   *     USER       Aprod, Msolve
   *     Utilities  daxpy_c, dcopy_c, ddot_c, dnrm2_c} These are all in
   *     Utilities  daxpy2_c,dload2_c,dscal2_c       } the file minresutil.c
   */

  /* local variables */
  double alfa, beta, beta1, cs, dbar, delta, denom, diag, eps, epsa, epsln,
      epsr, epsx, gamma, gbar, gmax, gmin, oldb, oldeps, qrnorm, phi, phibar,
      rhs1, rhs2, s, sn, t, tnorm2, ynorm2, z;
  uint64_t i;
  uint8_t debug, prnt;

  double zero, one, two, ten;
  zero = 0.0f;
  one = 1.0f;
  two = 1.0f;
  ten = 10.0f;

  char enter[16], exit[16], msg[10][52];
  strncpy(enter, " Enter MINRES. ", 16);
  strncpy(exit, " Exit  MINRES. ", 16);
  strncpy(msg[0], "beta2 = 0.  If M = I, b and x are eigenvectors of A", 52);
  strncpy(msg[1], "beta1 = 0.  The exact solution is  x = 0", 52);
  strncpy(msg[2], "Requested accuracy achieved, as determined by rtol", 52);
  strncpy(msg[3], "Reasonable accuracy achieved, given eps", 52);
  strncpy(msg[4], "x has converged to an eigenvector", 52);
  strncpy(msg[5], "Acond has exceeded 0.1/eps", 52);
  strncpy(msg[6], "The iteration limit was reached", 52);
  strncpy(msg[7], "Aprod  does not define a symmetric matrix", 52);
  strncpy(msg[8], "Msolve does not define a symmetric matrix", 52);
  strncpy(msg[9], "Msolve does not define a pos-def preconditioner", 52);

  debug = MINRES_FALSE;

  /*
   *     ------------------------------------------------------------------
   *     Compute eps, the machine precision.  The call to daxpy is
   *     intended to fool compilers that use extra-length registers.
   *     31 May 1999: Hardwire eps so the debugger can step thru easily.
   *     ------------------------------------------------------------------
   */
  eps = 2.22e-16f; /* Set eps = zero here if you want it computed. */

  if (eps <= 0.0f) {
    eps = pow(two, -12);
    do {
      eps = eps / two;
      x[0] = eps;
      y[0] = one;
      daxpy_c(1, one, x, y);
    } while (y[0] > one);
    eps = eps * two;
  }

  /*
   *   ------------------------------------------------------------------
   *   Print heading and initialize.
   *   ------------------------------------------------------------------
   */
  if (nout > 0) {
    printf(
        "%s      Solution of symmetric   Ax = b\n"
        " n      =%7lu     checkA =%4u            precon =%4u\n"
        " itnlim =%7lu     rtol   =%11.2e     shift  =%23.14e\n",
        enter, n, checkA, precon, itnlim, rtol, shift);
  }
  *istop = 0;
  *itn = 0;
  *Anorm = zero;
  *Acond = zero;
  *rnorm = zero;
  *ynorm = zero;
  dload2_c(n, zero, x);

  /*
   *     ------------------------------------------------------------------
   *     Set up y and v for the first Lanczos vector v1.
   *     y  =  beta1 P' v1,  where  P = C**(-1).
   *     v is really P' v1.
   *     ------------------------------------------------------------------
   */
  dcopy_c(n, b, y);  /* y = b */
  dcopy_c(n, b, r1); /* r1 = b */
  if (precon) Msolve(n, b, y);
  beta1 = ddot_c(n, b, y);

  if (beta1 < zero) { /* M must be indefinite. */
    *istop = 8;
    DISPLAY_FINAL_RESULTS_AND_EXIT(exit, istop, itn, Anorm, Acond, rnorm, ynorm,
                                   msg)
  }

  if (beta1 == zero) { /* b = 0 exactly.  Stop with x = 0. */
    *istop = 0;
    DISPLAY_FINAL_RESULTS_AND_EXIT(exit, istop, itn, Anorm, Acond, rnorm, ynorm,
                                   msg)
  }

  beta1 = sqrt(beta1); /* Normalize y to get v1 later. */

  /*
   *     ------------------------------------------------------------------
   *     See if Msolve is symmetric.
   *     ------------------------------------------------------------------
   */
  if (checkA && precon) {
    Msolve(n, y, r2);
    s = ddot_c(n, y, y);
    t = ddot_c(n, r1, r2);
    z = fabs(s - t);
    epsa = (s + eps) * pow(eps, 0.3333f);
    if (z > epsa) {
      *istop = 7;
      DISPLAY_FINAL_RESULTS_AND_EXIT(exit, istop, itn, Anorm, Acond, rnorm,
                                     ynorm, msg)
    }
  }

  /*
   *     ------------------------------------------------------------------
   *     See if Aprod  is symmetric.
   *     ------------------------------------------------------------------
   */
  if (checkA) {
    Aprod(n, y, w);
    Aprod(n, w, r2);
    s = ddot_c(n, w, w);
    t = ddot_c(n, y, r2);
    z = fabs(s - t);
    epsa = (s + eps) * pow(eps, 0.3333f);
    if (z > epsa) {
      *istop = 6;
      DISPLAY_FINAL_RESULTS_AND_EXIT(exit, istop, itn, Anorm, Acond, rnorm,
                                     ynorm, msg)
    }
  }

  /*
   *     ------------------------------------------------------------------
   *     Initialize other quantities.
   *     ------------------------------------------------------------------
   */
  oldb = zero;
  beta = beta1;
  dbar = zero;
  epsln = zero;
  qrnorm = beta1;
  phibar = beta1;
  rhs1 = beta1;
  rhs2 = zero;
  tnorm2 = zero;
  ynorm2 = zero;
  cs = -one;
  sn = zero;
  dload2_c(n, zero, w);
  dload2_c(n, zero, w2);
  dcopy_c(n, r1, r2);

  if (debug) {
    printf("  \n");
    printf("b\t");
    for (i = 0; i < n; i++) printf("%f\t", b[i]);
    printf("\n");
    printf("beta\t%f\n", beta);
    printf("  \n");
  }

  /*
   *     ------------------------------------------------------------------
   *     Main iteration loop.
   *     ------------------------------------------------------------------
   */
  while (MINRES_TRUE) {
    *itn = *itn + 1; /* k = itn = 1 first time through */
    if (*istop != 0)
      DISPLAY_FINAL_RESULTS_AND_EXIT(exit, istop, itn, Anorm, Acond, rnorm,
                                     ynorm, msg)
    /*
     *-----------------------------------------------------------------
     * Obtain quantities for the next Lanczos vector vk+1, k = 1, 2,...
     * The general iteration is similar to the case k = 1 with v0 = 0:
     *
     *   p1      = Operator * v1  -  beta1 * v0,
     *   alpha1  = v1'p1,
     *   q2      = p2  -  alpha1 * v1,
     *   beta2^2 = q2'q2,
     *   v2      = (1/beta2) q2.
     *
     * Again, y = betak P vk,  where  P = C**(-1).
     * .... more description needed.
     *-----------------------------------------------------------------
     */
    s = one / beta;       /* Normalize previous vector (in y). */
    dscal2_c(n, s, y, v); /* v = vk if P = I */

    Aprod(n, v, y);
    daxpy_c(n, (-shift), v, y);
    if (*itn >= 2) {
      daxpy_c(n, (-beta / oldb), r1, y);
    }

    alfa = ddot_c(n, v, y); /* alphak */

    daxpy_c(n, (-alfa / beta), r2, y);
    dcopy_c(n, r2, r1);
    dcopy_c(n, y, r2);
    if (precon) Msolve(n, r2, y);

    oldb = beta;             /* oldb = betak */
    beta = ddot_c(n, r2, y); /* beta = betak+1^2 */
    if (beta < zero) {
      *istop = 6;
      DISPLAY_FINAL_RESULTS_AND_EXIT(exit, istop, itn, Anorm, Acond, rnorm,
                                     ynorm, msg)
    }

    beta = sqrt(beta); /* beta = betak+1 */
    tnorm2 = tnorm2 + alfa * alfa + oldb * oldb + beta * beta;
    if (*itn == 1) {                   /* Initialize a few things. */
      if (beta / beta1 <= ten * eps) { /* beta2 = 0 or ~ 0. */
        *istop = -1;                   /* Terminate later. */
      }
      /* tnorm2 = alfa * alfa; */
      gmax = fabs(alfa); /* alpha1 */
      gmin = gmax;       /* alpha1 */
    }

    /*
     * Apply previous rotation Qk-1 to get
     *   [deltak epslnk+1] = [cs  sn][dbark    0   ]
     *   [gbar k dbar k+1]   [sn -cs][alfak betak+1].
     */
    oldeps = epsln;
    delta = cs * dbar + sn * alfa; /* delta1 = 0         deltak */
    gbar = sn * dbar - cs * alfa;  /* gbar 1 = alfa1     gbar k */
    epsln = sn * beta;             /* epsln2 = 0         epslnk+1 */
    dbar = -cs * beta;             /* dbar 2 = beta2     dbar k+1 */

    /*
     *Compute the next plane rotation Qk
     */
    gamma = sqrt(gbar * gbar + beta * beta); /* gammak */
    cs = gbar / gamma;                       /* ck */
    sn = beta / gamma;                       /* sk */
    phi = cs * phibar;                       /* phik */
    phibar = sn * phibar;                    /* phibark+1 */

    if (debug) {
      printf("  \n");
      printf("v\t");
      for (i = 0; i < n; i++) printf("%f\t", v[i]);
      printf("\n");
      printf("alfa \t%f\n", alfa);
      printf("beta \t%f\n", beta);
      printf("gamma\t%f\n", gamma);
      printf("delta\t%f\n", delta);
      printf("gbar \t%f\n", gbar);
      printf("epsln\t%f\n", epsln);
      printf("dbar \t%f\n", dbar);
      printf("phi  \t%f\n", phi);
      printf("phiba\t%f\n", phibar);
      printf("  \n");
    }

    /* Update  x. */

    denom = one / gamma;

    for (i = 0; i < n; i++) {
      w1[i] = w2[i];
      w2[i] = w[i];
      w[i] = (v[i] - oldeps * w1[i] - delta * w2[i]) * denom;
      x[i] = x[i] + phi * w[i];
    }

    /* Go round again. */

    gmax = MINRES_MAX(gmax, gamma);
    gmin = MINRES_MIN(gmin, gamma);
    z = rhs1 / gamma;
    ynorm2 = z * z + ynorm2;
    rhs1 = rhs2 - delta * z;
    rhs2 = -epsln * z;

    /* Estimate various norms and test for convergence. */

    *Anorm = sqrt(tnorm2);
    *ynorm = sqrt(ynorm2);
    epsa = *Anorm * eps;
    epsx = *Anorm * *ynorm * eps;
    epsr = *Anorm * *ynorm * rtol;
    diag = gbar;
    if (diag == zero) diag = epsa;

    qrnorm = phibar;
    *rnorm = qrnorm;

    /*
     * Estimate  cond(A).
     * In this version we look at the diagonals of  R  in the
     * factorization of the lower Hessenberg matrix,  Q * H = R,
     * where H is the tridiagonal matrix from Lanczos with one
     * extra row, beta(k+1) e_k^T.
     */
    *Acond = gmax / gmin;

    /*
     * See if any of the stopping criteria are satisfied.
     * In rare cases, istop is already -1 from above (Abar = const*I).
     */
    if (*istop == 0) {
      if (*itn >= itnlim) *istop = 5;
      if (*Acond >= 0.1f / eps) *istop = 4;
      if (epsx >= beta1) *istop = 3;
      if (qrnorm <= epsx) *istop = 2;
      if (qrnorm <= epsr) *istop = 1;
    }

    /* See if it is time to print something. */
    if (nout >= 0) {
      prnt = MINRES_FALSE;
      if (n <= 40) prnt = MINRES_TRUE;
      if (*itn <= 10) prnt = MINRES_TRUE;
      if (*itn >= itnlim - 10) prnt = MINRES_TRUE;
      if (*itn % 10 == 0) prnt = MINRES_TRUE;
      if (qrnorm <= ten * epsx) prnt = MINRES_TRUE;
      if (qrnorm <= ten * epsr) prnt = MINRES_TRUE;
      if (*Acond >= 1.0e-2f / eps) prnt = MINRES_TRUE;
      if (*istop == 0) prnt = MINRES_TRUE;
    }

    if (prnt) {
      if (*itn == 1) {
        printf("     itn        x[0]          norm(r)   norm(A)   cond(A)\n");
      }
      printf("%8lu%19.10f%10.2f%10.2f%10.2f\n", *itn, x[0], qrnorm, *Anorm,
             *Acond);
      if ((*itn + 1) % 10 == 0) {
        printf(" \n");
      }
    }
  }
  /*
   *     ------------------------------------------------------------------
   *     End of main iteration loop.
   *     ------------------------------------------------------------------
   */
}