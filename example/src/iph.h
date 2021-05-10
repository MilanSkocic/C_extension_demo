#ifndef IPH_H
#define IPH_H
/**
 *@file iph.h
 */

#include <math.h>
#include <complex.h>


#if _MSC_VER >= 1910
    #pragma message "WINDOWS C Compiler"
    #define COMPLEX_TYPE _Dcomplex
    #define buildz(real, imag) _Cbuild(real, imag)
#else
    #pragma message "GCC COMPILER"
    #define COMPLEX_TYPE double complex
    #define buildz(real, imag) (real+I*imag)
#endif

#define PI 3.1415926535897932384626433832795L

void iphC(int N, int SC,
        COMPLEX_TYPE (*iph_complex),
        double (*hv),
        double (*K_i),
        double (*theta_i),
        double (*Eg_i),
        double (*phiN));

#endif