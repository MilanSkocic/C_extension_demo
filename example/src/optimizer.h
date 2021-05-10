#ifndef OPTIMIZER_H
#define OPTIMIZER_H
#include <stdio.h>

/**
 *@file optimizer.h
 */

void optimizer(void (*fcn)(size_t, size_t, double *, double *), size_t m, size_t n, double *x, double *fvec);

#endif
