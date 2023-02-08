#ifndef MINRES_H
#define MINRES_H

#include <math.h>
#include <stdio.h>
#include <string.h>

#include "minresutil.h"

#define MINRES_FALSE (0)
#define MINRES_TRUE (1)
#define MINRES_MAX(a, b) ((a) > (b) ? (a) : (b))
#define MINRES_MIN(a, b) ((a) < (b) ? (a) : (b))

#ifdef __cplusplus
extern "C" {
#endif

void minres(uint64_t n, double *b, double *r1, double *r2, double *v, double *w,
            double *w1, double *w2, double *x, double *y,
            void (*Aprod)(uint64_t, double *, double *),
            void (*Msolve)(uint64_t, double *, double *), uint8_t checkA,
            uint8_t precon, double shift, uint64_t nout, uint64_t itnlim,
            double rtol, int8_t *istop, uint64_t *itn, double *Anorm,
            double *Acond, double *rnorm, double *ynorm);

#ifdef __cplusplus
}
#endif

#endif
