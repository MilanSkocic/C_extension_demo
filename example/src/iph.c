#include "iph.h"
/**
 *@file iph.c
 */


/**
 * @brief Compute complex photocurrent.
 * @param N Size of arrays
 * @param SC Number of semiconductive phases
 * @param iph_complex Pointer to the output array
 * @param hv Pointer to an array
 * @param K_i Pointer to an array
 * @param theta_i Pointer to an array
 * @param Eg_i Pointer to an array
 * @param phiN Pointer to an array
 * @param N Size of the arrays
 */
void iphC(int N, int SC,
        COMPLEX_TYPE (*iph_complex),
        double (*hv),
        double (*K_i),
        double (*theta_i),
        double (*Eg_i),
        double (*phiN))
{
    double phase=0.0;
    double dhv=0.0;
    double mod_i=0.0;
    double zr_i=0.0;
    double zi_i=0.0;
    int i = 0;

    for (int j=0; j<SC; j++)
    {
	    phase = *(theta_i+j)*PI/180.0;

        for(i=0; i<N; i++)
        {
            dhv = (*(hv+i) - *(Eg_i+j));
            if (dhv>0.0)
            {
                mod_i = pow(*(K_i+j)*dhv, 2)/(*(hv+i))*(*(phiN+i));
                zr_i = mod_i*cos(phase) + creal(*(iph_complex+i));
                zi_i = mod_i*sin(phase) + cimag(*(iph_complex+i));;
                *(iph_complex+i) = buildz(zr_i, zi_i);
            }
        }

    }
}

