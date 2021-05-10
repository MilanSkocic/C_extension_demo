#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "numpy/arrayobject.h"
#include "numpy/ndarraytypes.h"
#include "numpy/npy_math.h"
#include "chi2.h"
#include "iph.h"

PyDoc_STRVAR(module_docstring, "This module provides an example of interfacing with C code.");

PyDoc_STRVAR(chi2_docstring, "Calculate the chi-squared of some data given a model.");

PyDoc_STRVAR(wrap_chi2_docstring, "Wrapper of C code. "
"Calculate the chi-squared of some data given a model.");

PyDoc_STRVAR(iph_docstring, "Calculate the complex Iph through C function.\n\n"
"Parameters\n"
"------------\n"
"hv: array_like\n"
"\tLight energy in eV.\n"
"K: array-like\n"
"\tAmplitude.\n"
"theta: array-like\n"
"\tPhase in degrees.\n"
"Eg: array-like\n"
"\tbandgap in eV.\n"
"phi_N: array-like\n"
"\tPhoton flux.\n");

PyDoc_STRVAR(wrap_iph_docstring, "Wrapper of C Code. "
"Calculate the complex Iph through C function wrapped and using strides from Numpy API.");


static PyObject *capi_wrap_chi2(PyObject *self, PyObject *args)
{
    double m, b;
    PyObject *x_obj, *y_obj, *yerr_obj;

    /* Parse the input tuple */
    if (!PyArg_ParseTuple(args, "ddOOO", &m, &b, &x_obj, &y_obj,
                                        &yerr_obj))
        return NULL;

    /* Interpret the input objects as numpy arrays. */
    PyArrayObject *x_array = PyArray_FROM_OTF(x_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *y_array = PyArray_FROM_OTF(y_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *yerr_array = PyArray_FROM_OTF(yerr_obj, NPY_DOUBLE,
                                            NPY_ARRAY_IN_ARRAY);

    /* If that didn't work, throw an exception. */
    if (x_array == NULL || y_array == NULL || yerr_array == NULL) {
        Py_XDECREF(x_array);
        Py_XDECREF(y_array);
        Py_XDECREF(yerr_array);
        return NULL;
    }

    /* How many data points are there? */
    int N = (int)PyArray_DIM(x_array, 0);

    /* Get pointers to the data as C-types. */
    double *x    = (double*)PyArray_DATA(x_array);
    double *y    = (double*)PyArray_DATA(y_array);
    double *yerr = (double*)PyArray_DATA(yerr_array);

    /* Call the external C function to compute the chi-squared. */
    double value = chi2(m, b, x, y, yerr, N);

    /* Clean up. */
    Py_DECREF(x_array);
    Py_DECREF(y_array);
    Py_DECREF(yerr_array);

    if (value < 0.0) {
        PyErr_SetString(PyExc_RuntimeError,
                    "Chi-squared returned an impossible value.");
        return NULL;
    }

    /* Build the output tuple */
    PyObject *ret = Py_BuildValue("d", value);
    return ret;
}

static PyObject *capi_chi2(PyObject *self, PyObject *args)
{
    double m, b, value, diff;
    int i;
    PyObject *x_obj, *y_obj, *yerr_obj;

    /* Parse the input tuple */
    if (!PyArg_ParseTuple(args, "ddOOO", &m, &b, &x_obj, &y_obj,
                                        &yerr_obj))
        return NULL;
    /* Interpret the input objects as contiguous numpy arrays. */
    PyArrayObject *x_array = PyArray_FROM_OTF(x_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *y_array = PyArray_FROM_OTF(y_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *yerr_array = PyArray_FROM_OTF(yerr_obj, NPY_DOUBLE,
                                            NPY_ARRAY_IN_ARRAY);
    /* If that didn't work, throw an exception. */
    if (x_array == NULL || y_array == NULL || yerr_array == NULL) {
        Py_XDECREF(x_array);
        Py_XDECREF(y_array);
        Py_XDECREF(yerr_array);
        return NULL;
    }


    /* Clean up. */
    Py_DECREF(x_array);
    Py_DECREF(y_array);
    Py_DECREF(yerr_array);

    /* How many data points are there? */
    int N = (int)PyArray_DIM(x_array, 0);

    double *x    = (double*)PyArray_DATA(x_array);
    double *y    = (double*)PyArray_DATA(y_array);
    double *yerr = (double*)PyArray_DATA(yerr_array);    /* Build the output tuple */

    value = 0.0;
    for (i = 0; i < N; i++) {
        diff = (y[i] - (m * x[i] + b)) / yerr[i];
        value += diff * diff;
    }

    PyObject *ret = Py_BuildValue("d", value);
    return ret;

}

static PyObject *capi_wrap_iph(PyObject *self, PyObject *args)
{
    PyObject *hv;
    PyObject *K;
    PyObject *theta;
    PyObject *Eg;
    PyObject *phi_N;

    PyArrayObject *hv_array;
    PyArrayObject *K_array;
    PyArrayObject *theta_array;
    PyArrayObject *Eg_array;
    PyArrayObject *phi_N_array;

    PyArrayObject *new_iph_complex_array = NULL;
    PyArrayObject *new_iph_complex_array_real = NULL;
    PyArrayObject *new_iph_complex_array_imag = NULL;


    /* Parse the input tuple */
    if (!PyArg_ParseTuple(args, "OOOOO", &hv, &K, &theta, &Eg, &phi_N))
        return NULL;

    /* Interpret the input objects as numpy arrays. */
    hv_array = (PyArrayObject *) PyArray_FROM_OTF(hv, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    K_array = (PyArrayObject *) PyArray_FROM_OTF(K, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    theta_array = (PyArrayObject *)PyArray_FROM_OTF(theta, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    Eg_array = (PyArrayObject *)PyArray_FROM_OTF(Eg, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    phi_N_array = (PyArrayObject *)PyArray_FROM_OTF(phi_N, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

    /* If that didn't work, throw an exception. */
    if (hv_array == NULL||K_array==NULL||theta_array==NULL||Eg_array==NULL||phi_N_array==NULL) {
        Py_XDECREF(hv_array);
        Py_XDECREF(K_array);
        Py_XDECREF(theta_array);
        Py_XDECREF(Eg_array);
        Py_XDECREF(phi_N_array);
        return NULL;
    }

    /*Create a new ZEROS ARRAY for output*/
    new_iph_complex_array = (PyArrayObject *)PyArray_ZEROS(PyArray_NDIM(hv_array), PyArray_DIMS(hv_array), NPY_COMPLEX128, 0);

    /* Get the number of points and parameters*/
    int N;
    int SC;
    SC = PyArray_DIM(K_array, 0);
    N = PyArray_DIM(hv_array, 0);

    /* pointers to the data arrays*/
    double *hvptr = (double *)PyArray_DATA(hv_array);
    double *Kptr = (double*)PyArray_DATA(K_array);
    double *thetaptr = (double*)PyArray_DATA(theta_array);
    double *Egptr = (double*)PyArray_DATA(Eg_array);
    double *phi_N_ptr = (double *)PyArray_DATA(phi_N_array);

    npy_cdouble *outptr = (npy_cdouble *)PyArray_DATA(new_iph_complex_array);

    /* Call C function */
    iphC(N, SC, outptr, hvptr, Kptr, thetaptr, Egptr, phi_N_ptr);

    /* Clean up. */
    Py_DECREF(hv_array);
    Py_DECREF(K_array);
    Py_DECREF(theta_array);
    Py_DECREF(Eg_array);
    Py_DECREF(phi_N_array);

    return PyArray_Return(new_iph_complex_array);

}

static PyObject *capi_iph(PyObject *self, PyObject *args)
{
    PyObject *hv;
    PyObject *prm;
    PyObject *phi_N;

    PyArrayObject *hv_array;
    PyArrayObject *prm_array;
    PyArrayObject *phi_N_array;

    PyArrayObject *new_iph_complex_array = NULL;

    /* Parse the input tuple */
    if (!PyArg_ParseTuple(args, "OOO", &hv, &prm, &phi_N))
        return NULL;

    /* Interpret the input objects as numpy arrays. */
    hv_array = (PyArrayObject *) PyArray_FROM_OTF(hv, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    prm_array = (PyArrayObject *) PyArray_FROM_OTF(prm, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    phi_N_array = (PyArrayObject *) PyArray_FROM_OTF(phi_N, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

    /*Create a new ZEROS ARRAY for output*/
    new_iph_complex_array = (PyArrayObject *)PyArray_ZEROS(PyArray_NDIM(hv_array), PyArray_DIMS(hv_array), NPY_COMPLEX128, 0);

    /* If that didn't work, throw an exception. */
    if (hv_array == NULL || prm_array == NULL || phi_N_array == NULL){
        Py_XDECREF(hv_array);
        Py_XDECREF(prm_array);
        Py_XDECREF(phi_N_array);
        return NULL;
    }

    unsigned int  N;
    unsigned int SC;
    unsigned int i, j;

    double *Ki;
    double *thetai;
    double *Egi;
    double *hvi;
    double *ni;
    double *powhvi;
    double dhvi;
    double *phi_N_i;
    double phase;
    double mod_i;

    npy_cdouble *out_i;

    N = PyArray_DIM(hv_array, 0);
    SC = PyArray_DIM(prm_array, 0);

    for (i=0; i<SC; i++)
    {
        Ki = (double *)((char *)PyArray_DATA(prm_array) + i*PyArray_STRIDES(prm_array)[0]);
        thetai = (double *)((char *)PyArray_DATA(prm_array) + 2*PyArray_STRIDES(prm_array)[1] + i*PyArray_STRIDES(prm_array)[0]);
        Egi = (double *)((char *)PyArray_DATA(prm_array) + 4*PyArray_STRIDES(prm_array)[1] + i*PyArray_STRIDES(prm_array)[0]);
        ni = (double *)((char *)PyArray_DATA(prm_array) + 6*PyArray_STRIDES(prm_array)[1] + i*PyArray_STRIDES(prm_array)[0]);
        powhvi = (double *)((char *)PyArray_DATA(prm_array) + 7*PyArray_STRIDES(prm_array)[1] + i*PyArray_STRIDES(prm_array)[0]);

        phase = (*thetai)*PI/180.0;


        for (j=0; j<N;j++)
        {
            out_i = (npy_cdouble *)((char *)PyArray_DATA(new_iph_complex_array)+j*PyArray_STRIDES(new_iph_complex_array)[0]);
            hvi = (double *)((char *)PyArray_DATA(hv_array) + j*PyArray_STRIDES(hv_array)[0]);
            phi_N_i = (double *)((char *)PyArray_DATA(phi_N_array) + j*PyArray_STRIDES(phi_N_array)[0]);
            dhvi = (*hvi)-(*Egi);
            if (dhvi>0.0)
            {
                mod_i = (*phi_N_i)*pow((*Ki)*dhvi, *ni)/(*hvi)*(*phi_N_i);
                *out_i = npy_cpack(mod_i*cos(phase) + npy_creal(*out_i), mod_i*sin(phase) + npy_cimag(*out_i));
            }


        }


    }
    /* Clean up. */
    Py_DECREF(hv_array);
    Py_DECREF(prm_array);
    Py_DECREF(phi_N_array);

    return PyArray_Return(new_iph_complex_array);

}


static PyMethodDef myMethods[] = {
    { "capi_wrap_chi2", (PyCFunction) capi_wrap_chi2, METH_VARARGS, wrap_chi2_docstring },
    { "capi_chi2", (PyCFunction) capi_chi2, METH_VARARGS, chi2_docstring },
    { "capi_wrap_iph", (PyCFunction) capi_wrap_iph, METH_VARARGS, wrap_iph_docstring },
    { "capi_iph", (PyCFunction) capi_iph, METH_VARARGS, iph_docstring },
    { NULL, NULL, 0, NULL }
};

// Our Module Definition struct
static struct PyModuleDef capi = {
    PyModuleDef_HEAD_INIT,
    "capi",
    module_docstring,
    -1,
    myMethods
};

// Initializes our module using our above struct
PyMODINIT_FUNC PyInit_capi(void)
{
    import_array();
    return PyModule_Create(&capi);
}


