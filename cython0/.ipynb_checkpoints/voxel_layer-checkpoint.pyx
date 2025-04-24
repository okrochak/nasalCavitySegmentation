cimport numpy as np
cimport cython

import numpy as np

def voxel_layer(a_in, b_in, c_in):
    cdef np.ndarray[double, ndim=3] a = a_in
    cdef np.ndarray[double, ndim=3] b = b_in
    cdef np.ndarray[double, ndim=3] c = c_in
    cdef int i,j,k
    cdef int N = a.shape[0]
    cdef int M = a.shape[1]
    cdef int L = a.shape[2]
 
    for i in range(1,N-2):
        for j in range(1,M-2):
            for k in range(1,L-2): 
                if a[i,j,k] == 0 and a[i-1,j-1,k-1] == 1:
                    c[i,j,k] = b[i,j,k]
                    c[i-1,j-1,k-1] = b[i-1,j-1,k-1]
                elif a[i,j,k] == 0 and a[i-1,j,k-1] == 1: 
                    c[i,j,k] = b[i,j,k]
                    c[i-1,j,k-1] = b[i-1,j,k-1]
                elif a[i,j,k] == 0 and a[i-1,j+1,k-1] == 1:
                    c[i,j,k] = b[i,j,k]
                    c[i-1,j+1,k-1] = b[i-1,j+1,k-1]
                elif a[i,j,k] == 0 and a[i,j-1,k-1] == 1:
                    c[i,j,k] = b[i,j,k] 
                    c[i,j-1,k-1] = b[i,j-1,k-1]
                elif a[i,j,k] == 0 and a[i,j,k-1] == 1:
                    c[i,j,k] = b[i,j,k]
                    c[i,j,k-1] = b[i,j,k-1]
                elif a[i,j,k] == 0 and a[i,j+1,k-1] == 1:
                    c[i,j,k] = b[i,j,k]
                    c[i,j+1,k-1] = b[i,j+1,k-1]
                elif a[i,j,k] == 0 and a[i+1,j-1,k-1] == 1:
                    c[i,j,k] = b[i,j,k]
                    c[i+1,j-1,k-1] = b[i+1,j-1,k-1]
                elif a[i,j,k] == 0 and a[i+1,j,k-1] == 1:
                    c[i,j,k] = b[i,j,k]
                    c[i+1,j,k-1] = b[i+1,j,k-1]
                elif a[i,j,k] == 0 and a[i+1,j+1,k-1] == 1:
                    c[i,j,k] = b[i,j,k]
                    c[i+1,j+1,k-1] = b[i+1,j+1,k-1]
                elif a[i,j,k] == 0 and a[i-1,j-1,k] == 1:
                    c[i,j,k] = b[i,j,k]
                    c[i-1,j-1,k] = b[i-1,j-1,k]
                elif a[i,j,k] == 0 and a[i-1,j,k] == 1:
                    c[i,j,k] = b[i,j,k]
                    c[i-1,j,k] = b[i-1,j,k]
                elif a[i,j,k] == 0 and a[i-1,j+1,k] == 1:
                    c[i,j,k] = b[i,j,k]
                    c[i-1,j+1,k] = b[i-1,j+1,k]
                elif a[i,j,k] == 0 and a[i,j-1,k] == 1:
                    c[i,j,k] = b[i,j,k]
                    c[i,j-1,k] = b[i,j-1,k]
                elif a[i,j,k] == 0 and a[i,j+1,k] == 1:
                    c[i,j,k] = b[i,j,k]
                    c[i,j+1,k] = b[i,j+1,k]
                elif a[i,j,k] == 0 and a[i+1,j-1,k] == 1:
                    c[i,j,k] = b[i,j,k]
                    c[i+1,j-1,k] = b[i+1,j-1,k]
                elif a[i,j,k] == 0 and a[i+1,j,k] == 1:
                    c[i,j,k] = b[i,j,k]
                    c[i+1,j,k] = b[i+1,j,k]
                elif a[i,j,k] == 0 and a[i+1,j+1,k] == 1:
                    c[i,j,k] = b[i,j,k]
                    c[i+1,j+1,k] = b[i+1,j+1,k]
                elif a[i,j,k] == 0 and a[i-1,j-1,k+1] == 1:
                    c[i,j,k] = b[i,j,k]
                    c[i-1,j-1,k+1] = b[i-1,j-1,k+1]
                elif a[i,j,k] == 0 and a[i-1,j,k+1] == 1:
                    c[i,j,k] = b[i,j,k]
                    c[i-1,j,k+1] = b[i-1,j,k+1]
                elif a[i,j,k] == 0 and a[i-1,j+1,k+1] == 1:
                    c[i,j,k] = b[i,j,k]
                    c[i-1,j+1,k+1] = b[i-1,j+1,k+1]
                elif a[i,j,k] == 0 and a[i,j-1,k+1] == 1:
                    c[i,j,k] = b[i,j,k]
                    c[i,j-1,k+1] = b[i,j-1,k+1]
                elif a[i,j,k] == 0 and a[i,j,k+1] == 1:
                    c[i,j,k] = b[i,j,k]
                    c[i,j,k+1] = b[i,j,k+1]
                elif a[i,j,k] == 0 and a[i,j+1,k+1] == 1:
                    c[i,j,k] = b[i,j,k]
                    c[i,j+1,k+1] = b[i,j+1,k+1]
                elif a[i,j,k] == 0 and a[i+1,j-1,k+1] == 1:
                    c[i,j,k] = b[i,j,k]
                    c[i+1,j-1,k+1] = b[i+1,j-1,k+1]
                elif a[i,j,k] == 0 and a[i+1,j+1,k+1] == 1:
                    c[i,j,k] = b[i,j,k]
                    c[i+1,j+1,k+1] = b[i+1,j+1,k+1]
    return c