from example import capi, capi_buffer_protocol, capi_callback
from example import cython_chi2
from example import ufuncs_example, ufuncs

import array
import numpy as np
import io

m=1
b=1
x = array.array('d',[1, 2, 3])
y = array.array('d',[1, 1, 1])
yerr = array.array('d', [1, 1, 1])
N = len(x)

print('************************************************************************************')
print('**** Example of computing chi2 - Demonstration of extentions with real numbers. ****')
print('************************************************************************************')

print("Chi2 written in C and wrap with Python/Numpy C API")
print(capi.capi_wrap_chi2(m, b, x, y, yerr))

print("\nChi2 written in C with Python/Numpy C API")
print(capi.capi_chi2(m, b, x, y, yerr))

print("\nChi2 written in C wrapped with Cython")
print(cython_chi2.cython_chi2(m, b, x, y, yerr, len(x)))

print("\nChi2 written with Cython")
print(cython_chi2.cython_wrap_chi2(m, b, x, y, yerr, len(x)))


print('\n')
print('*********************************************************************************************')
print('*****************************Example of custom ufunc*****************************************')
print('*********************************************************************************************')
print('Ufunc example')
print(ufuncs_example.absorbance(np.asarray(x), 1, 1, 1))

print('\n')
print('*********************************************************************************************')
print('Example of computing complex photocurrent - Demonstration of extentions with complex numbers.')
print('*********************************************************************************************')

t = """4.0e-05	1.0e+00	-5.0e+01	1.00e+00	1.70e+00	1.00e+00	2.00e+00	1.0000000e+00
6.0e-05	1.0e+00	1.30e+02	1.00e+00	2.40e+00	1.00e+00	2.00e+00	1.0000000e+00
5.0e-05	1.0e+00	1.40e+02	1.00e+00	2.80e+00	1.00e+00	2.00e+00	1.0000000e+00
9.0e-05	1.0e+00	-5.0e+01	1.00e+00	3.50e+00	1.00e+00	2.00e+00	1.0000000e+00"""

hv = hv = np.linspace(6, 2, 3)
prm = np.genfromtxt(io.StringIO(t))
K = prm[:,0].copy()
theta = prm[:, 2].copy()
Eg = prm[:, 4].copy()
phi_N = np.ones_like(hv)

print('\niph written in C and wrap with Python/Numpy C API')
print(capi.capi_wrap_iph(hv, K, theta, Eg, phi_N))


print("\niph written in C with Python/Numpy C API")
print(capi.capi_iph(hv, prm, phi_N))


print('\niph written with custom ufunc')
print(ufuncs.iph_with_ufunc(hv, prm, phi_N))


print('\n')
print('**********************************************************************************')
print('Example of creating memoryview - Demonstration of extentions with buffer protocol.')
print('***********************************************************************************')
a = array.array('d', (1, 1, 1, 1, 1, 1, 1, 1, 1, 1))

print('\nInitial array from the array module')
print(a, "id=", id(a))

out = capi_buffer_protocol.capi_fibonacci_array_input(a)

print('\nInitial array with the Fibonacci serie computed at C level working directly on the underlying buffer.')
print(out, "id=", id(out))

out = capi_buffer_protocol.capi_fibonacci_new_array(10)

print('\nNew memory view created at C level - can be used by Numpy')
print(out)

print('\tnbytes', out.nbytes)
print('\tndim', out.ndim)
print('\tshape', out.ndim)
print('\titemsize', out.itemsize)
print('\tobj', out.obj)
print('\tformat', out.format)
print('\tc_contiguous', out.c_contiguous)
print('\tstrides', out.strides)
print('\treadonly', out.readonly)

print('\n\tExample of exporting to list')
print(out.tolist())
print('\n\tExample of importing in array module')
print(array.array("d", out))
print('\n\tExample of importing in Numpy')
print(np.asarray(out))

print('\n')
print('**********************************************************************************')
print('Example callback function.')
print('***********************************************************************************')
p = [1.0, 1.0]
x = np.linspace(1, 10, 3)
y = x+1
w = 1/y


def model(p, x):
    return p[0]*x+p[1]


def residuals(p, x, y, w, model):
    res = (y - model(p, x))*w
    return res


args = (x, y, w, model)
res = capi_callback.wrap_optimizer(residuals, p, args)
print(res)