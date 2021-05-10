#include "chi2.h"

/**
 * @file chi2.c
 */

/**
 * @brief Compute chi2 from x, y, yerr arrays
 * @param m slope
 * @param b intercept
 * @param x Pointer to an array
 * @param y Pointer to an array
 * @param yerr Pointer to an array
 * @param N Size of the arrays
 * @return chi2 Chi2 value
 */
double chi2(double m, double b, double *x, double *y, double *yerr, int N) {
    int n;
    double result = 0.0, diff;

    for (n = 0; n < N; n++) {
        diff = (y[n] - (m * x[n] + b)) / yerr[n];
        result += diff * diff;
    }

    return result;
}
