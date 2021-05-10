#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "Python.h"
#include "math.h"
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"
#include "numpy/halffloat.h"

/*
 * Usual prototype for declaring methods in CPython module
 * Except that they are empty in the case of ufuncs
 */

static PyMethodDef AbsorbanceMethods[] = {
        {NULL, NULL, 0, NULL}
};

/* The loop definitions must precede the PyMODINIT_FUNC. */


/*
 *******************************************
 * ABSORBANCE FUNCTIONS
 * Prototype: abrsorbance(hv, Eg, n, powhv)
 ********************************************
 */
/* DOCSTRINGS */
PyDoc_STRVAR(absorbance_docstring, "Compute the absorbance coefficient.\n\n"
"Parameters\n"
"------------\n"
"hv: array_like\n"
"\tLight energy in eV.\n"
"Eg: array-like\n"
"\tbandgap in eV.\n"
"n: array-like\n"
"\tExponent for (in)direct transitions.\n"
"powhv: array-like\n"
"\tExponent hv in the denominator.\n");

static void double_absorbance(char **args, npy_intp *dimensions,
                            npy_intp* steps, void* data)
{
    npy_intp i;
    npy_intp n = dimensions[0];
    char *in1 = args[0], *in2 = args[1], *in3 = args[2], *in4 = args[3];
    char *out1 = args[4];
    npy_intp in1_step = steps[0], in2_step = steps[1], in3_step = steps[2], in4_step = steps[3];
    npy_intp out1_step = steps[4];

    npy_double hv;
    npy_double d;
    npy_double Eg;
    npy_double powd;
    npy_double powhv;

    for (i = 0; i < n; i++) {
        /*BEGIN main ufunc computation*/
        hv = *(npy_double *)in1;
        Eg = *(npy_double *)in2;
        powd = *(npy_double *)in3;
        powhv = *(npy_double *)in4;
        d = hv - Eg;
        if (d>0)
        {
            *((npy_double *)out1) = pow(d,powd)/pow(hv, powhv);
        }
        else
        {
            *((npy_double *)out1) = 0.0;
        }


        /*END main ufunc computation*/

        in1 += in1_step;
        in2 += in2_step;
        in3 += in3_step;
        in4 += in4_step;
        out1 += out1_step;
    }
}

/* Pointers to functions */
PyUFuncGenericFunction absorbance_funcs[1] = {&double_absorbance};


/* These are the input and return dtypes.*/
static char absorbance_types[5] = {NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE};
static void *absorbance_data[1] = {NULL};


/* Module definition */
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "ufuncs_example",
    NULL,
    -1,
    AbsorbanceMethods,
    NULL,
    NULL,
    NULL,
    NULL
};

/* Module initialization */
PyMODINIT_FUNC PyInit_ufuncs_example(void)
{
    PyObject *m, *d, *absorbance;
    m = PyModule_Create(&moduledef);
    if (!m) {
        return NULL;
    }

    import_array();
    import_umath();

    absorbance = PyUFunc_FromFuncAndData(absorbance_funcs, absorbance_data, absorbance_types, 1, 4, 1,
                                    PyUFunc_None, "absorbance",
                                    absorbance_docstring, 0);

    d = PyModule_GetDict(m);

    PyDict_SetItemString(d, "absorbance", absorbance);

    Py_DECREF(absorbance);

    return m;
}