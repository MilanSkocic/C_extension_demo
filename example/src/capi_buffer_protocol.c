#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "fib.h"

PyDoc_STRVAR(module_docstring, "This module provides an example of interfacing with the buffer protocol at C level.");

PyDoc_STRVAR(fib_array_docstring, "Compute the fibonacci serie with input array from the array module. Avoid compilation with Numpy");

PyDoc_STRVAR(fib_mview_docstring, "Compute the fibonacci serie by creating a new memory view using the buffer protocol. Avoid compilation with Numpy");


static PyObject *capi_fibonacci_array_input(PyObject *self, PyObject *args)
{
    PyObject *array;
    PyObject *mview;
    Py_buffer *buffer;
    Py_ssize_t *shape;

    /* Parse the input tuple */
    if (!PyArg_ParseTuple(args, "O", &array))
        return NULL;

    /* Get memoryview and the associated buffer */
    mview = PyMemoryView_FromObject(array);
    buffer = PyMemoryView_GET_BUFFER(mview);
    shape = buffer->shape;

    /* cast and pass buffer and shape */
    fib((double *) buffer->buf, (int) *shape);

    /* Release buffer */
    PyBuffer_Release(buffer);

    return array;
}

static PyObject *capi_fibonacci_new_array(PyObject *self, PyObject *args)
{
    Py_ssize_t n;
    PyObject *new_mview;
    Py_buffer new_buffer;
    Py_ssize_t new_shape[1]= {0};
    Py_ssize_t strides[1] = {sizeof(double)};

    /* Parse the input tuple */
    if (!PyArg_ParseTuple(args, "n", &n))
        return NULL;

    new_shape[0] = n;

    /* Create a new buffer with the associated properties as struct members */
    /* adapt the creatioon the desired type */
    new_buffer.buf = PyMem_Calloc((size_t) *new_shape, sizeof(double));
    new_buffer.obj = NULL;
    new_buffer.len = *new_shape * sizeof(double);
    new_buffer.readonly = 0;
    new_buffer.itemsize = sizeof(double);
    new_buffer.format = "d";
    new_buffer.ndim = 1;
    new_buffer.shape = new_shape;
    new_buffer.strides = strides;
    new_buffer.suboffsets = NULL;

    /* create a memoryview from buffer.
    An advantage of returning the memory view instead of the
    buffer only is that the former is a Python object and can be used by Numpy
    If necessary to create an extension that creates n-dimensional c-contiguous buffers
    that can be used by Numpy without having to compile the extension with Numpy */
    new_mview = PyMemoryView_FromBuffer(&new_buffer);

    /* cast and pass buffer and shape */
    fib((double *) new_buffer.buf, (int) *new_shape);

    /* Release buffer */
    PyBuffer_Release(&new_buffer);

    return new_mview;
}

static PyMethodDef myMethods[] = {
    { "capi_fibonacci_array_input", (PyCFunction) capi_fibonacci_array_input, METH_VARARGS, fib_array_docstring },
    { "capi_fibonacci_new_array", (PyCFunction) capi_fibonacci_new_array, METH_VARARGS, fib_mview_docstring },
    { NULL, NULL, 0, NULL }
};

// Our Module Definition struct
static struct PyModuleDef capi_buffer_protocol = {
    PyModuleDef_HEAD_INIT,
    "capi_buffer_protocol",
    module_docstring,
    -1,
    myMethods
};

// Initializes our module using our above struct
PyMODINIT_FUNC PyInit_capi_buffer_protocol(void)
{
    return PyModule_Create(&capi_buffer_protocol);
}


