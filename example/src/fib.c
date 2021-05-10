#include "fib.h"
/**
 *@file fib.c
 */


/**
 * @brief Compute the Fibonacci serie.
 * @param x Pointer to the array to perform the computation on
 * @param N array size
 */
void fib(double *x, int N){

    int i;

    for (i=0; i<N; i++){
        if (i==0){
            *(x+i)=0.0;
        }
        else if (i==1){
            *(x+i) = 1.0;
        }
        else{
            *(x+i) = *(x+i-1) + *(x+i-2);
        }
    }

}