#include "optimizer.h"
/**
 *@file optimizer.c
 */


/**
 * @brief Perform a dummy optimization
 * @param fcn Pointer to the cost function with the signature (size_t, size_t, double *, double *)
 * @param m Number of elements in array x to be optimized
 * @param n Number of elements in array fvec
 * @param x array to be optimized
 * @param fvec array of cost function values
 */void optimizer(void (*fcn)(size_t, size_t, double *, double *), size_t m, size_t n, double *x, double *fvec){

    size_t i;

    // dummy optimizer loop
    for (i=0; i<3; i++)
    {
        /* simulate changes in vector to optimize */
        *x = i;
        *(x+1) = i+1;
        /* call callback C wrapper */
        (*fcn)(m, n, x, fvec);

    }
}