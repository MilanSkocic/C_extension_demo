#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "numpy/arrayobject.h"
#include "numpy/ndarraytypes.h"
#include "numpy/npy_math.h"
#include "optimizer.h"

PyDoc_STRVAR(module_docstring, "This module provides an example of calling a Python callback function from C Code."
"Usually used for wrapping function optimizers.");

PyDoc_STRVAR(optimizer_docstring, "Dummy optimizer. It does not perform any calculations. "
"optimizer(fcn, args)-> (fvec, xopt)");

/* Python callback function - borrowed reference */
static PyObject *py_fcn=NULL;

/* passed from caller - borrowed references */
static PyObject *fcn_args=NULL;

/* new_args = (x, *args) - stealed references */
static PyObject *fcn_xargs=NULL;

/* initial guess for the optimizer - borrowed reference - will be turned into ndarray if not already */
static PyObject *xopt_obj=NULL;

/* ndarray of the initial guess - owned reference */
static PyArrayObject *xopt_array = NULL;

/* returned values from Python callback - owned reference */
static PyObject *fvec_obj=NULL;

/* returned values from Python callback - owned reference */
static PyArrayObject *fvec_array=NULL;


/* C wrapper of the Python callback function
 * signature is dependent on the definition of cost function signature in the optimizer
 * the principle is to get C variable and update the Python variable passed as arguments to the Python callback */
void fcn(size_t m, size_t n, double *x, double *fvec){

    fvec_obj = PyObject_Call(py_fcn, fcn_xargs, NULL);
    fvec_array = (PyArrayObject *) PyArray_FROM_OTF(fvec_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

}

/* Python wrapper of the C optimizer */
static PyObject *wrap_optimizer(PyObject *self, PyObject *args)
{
    size_t m, n, i, nargs;
    PyObject *item=NULL;

    /* Parse the input tuple */
    if (!PyArg_ParseTuple(args, "OOO",&py_fcn, &xopt_obj, &fcn_args))
        return NULL;
    /* check if py_fcn is a callable */
    if (!PyCallable_Check(py_fcn)) {
        PyErr_SetString(PyExc_TypeError, "func must be callable");
        return NULL;
    }
    /* check if args is an iterable */
    if (!PySequence_Check(xopt_obj)){
        PyErr_SetString(PyExc_TypeError, "x0 must be an iterable");
        return NULL;
    }
    /* check if args is an iterable */
    if (!PySequence_Check(fcn_args)){
        PyErr_SetString(PyExc_TypeError, "args must be an iterable");
        return NULL;
    }
    /* check if fcn_args can turned to an Numpy ndarray */
    xopt_array = (PyArrayObject *) PyArray_FROM_OTF(xopt_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (xopt_array == NULL){
        PyErr_SetString(PyExc_TypeError, "x0 must be an iterable.");
        Py_XDECREF(xopt_array);
        return NULL;
    }
    // args = (p, *args) in Python
    // PyTuple_GetItem borrow a reference
    // PyTuple_SetItem steals a reference
    nargs = PySequence_Length(fcn_args)+1;
    fcn_xargs = PyTuple_New(nargs);
    PyTuple_SetItem(fcn_xargs, 0, (PyObject *)xopt_array);
    Py_INCREF(xopt_array);
    for (i=1; i<nargs; i++) {
        item = PyTuple_GetItem(fcn_args, i-1);
        PyTuple_SetItem(fcn_xargs, i, item);
    }
    /* get number of elements in array to be optimized */
    m = PyArray_DIM(xopt_array, 0);

    /* call callback function */
    fvec_obj = PyObject_Call(py_fcn, fcn_xargs, NULL);
    fvec_array = (PyArrayObject *) PyArray_FROM_OTF(fvec_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    /* check if returned object from callback is an iterable that can be turned into an Numpy array */
    if (fvec_obj == NULL )
    {
        PyErr_SetString(PyExc_TypeError, "func must return an iterable.");
        Py_XDECREF(fvec_obj);
        return NULL;
    }
    if (fvec_array == NULL )
    {
        PyErr_SetString(PyExc_TypeError, "func must return an iterable.");
        Py_XDECREF(fvec_array);
        return NULL;
    }
    /* get number of elements in fvec */
    n = PyArray_DIM(fvec_array, 0);

    /* call optimizer
     * pass xopt_array pointer to data for the optimizer to work on */
    optimizer(&fcn, m, n,
                (double *)(PyArray_DATA(xopt_array)),
                    NULL);

    /* cleanup not returned objects */
    Py_XDECREF(fcn_xargs);
    Py_XDECREF(fvec_obj);

    /* Return objects */
    return Py_BuildValue("(OO)", PyArray_Return(xopt_array), PyArray_Return(fvec_array));
}

// Methods definitons
static PyMethodDef myMethods[] = {
    { "wrap_optimizer", (PyCFunction) wrap_optimizer, METH_VARARGS, optimizer_docstring },
    { NULL, NULL, 0, NULL }
};

// Our Module Definition struct
static struct PyModuleDef capi_callback = {
    PyModuleDef_HEAD_INIT,
    "capi_callback",
    module_docstring,
    -1,
    myMethods
};

// Initializes our module using our above struct
PyMODINIT_FUNC PyInit_capi_callback(void)
{
    import_array();
    return PyModule_Create(&capi_callback);
}


