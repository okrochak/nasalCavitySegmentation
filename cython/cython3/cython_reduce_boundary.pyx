cimport numpy as np
cimport cython

import numpy as np

def reduce_boundary(a_in, b_in):
    cdef np.ndarray[double, ndim=3] a = a_in
    cdef np.ndarray[double, ndim=3] b = b_in
    cdef int i,j,k
    cdef int N = a.shape[0]
    cdef int M = a.shape[1]
    cdef int L = a.shape[2]
 
    for i in range(1,N-2):
        for j in range(1,M-2):
            for k in range(1,L-2): 
                if a[i,j,k] == 0 and a[i-1,j-1,k-1] == 1:
                    b[i-1,j-1,k-1] = 0
                elif a[i,j,k] == 0 and a[i-1,j,k-1] == 1: 
                    b[i-1,j,k-1] = 0
                elif a[i,j,k] == 0 and a[i-1,j+1,k-1] == 1:
                    b[i-1,j+1,k-1] = 0
                elif a[i,j,k] == 0 and a[i,j-1,k-1] == 1:
                    b[i,j-1,k-1] = 0
                elif a[i,j,k] == 0 and a[i,j,k-1] == 1:
                    b[i,j,k-1] = 0
                elif a[i,j,k] == 0 and a[i,j+1,k-1] == 1:
                    b[i,j+1,k-1] = 0
                elif a[i,j,k] == 0 and a[i+1,j-1,k-1] == 1:
                    b[i+1,j-1,k-1] = 0
                elif a[i,j,k] == 0 and a[i+1,j,k-1] == 1:
                    b[i+1,j,k-1] = 0
                elif a[i,j,k] == 0 and a[i+1,j+1,k-1] == 1:
                    b[i+1,j+1,k-1] = 0
                elif a[i,j,k] == 0 and a[i-1,j-1,k] == 1:
                    b[i-1,j-1,k] = 0
                elif a[i,j,k] == 0 and a[i-1,j,k] == 1:
                    b[i-1,j,k] = 0
                elif a[i,j,k] == 0 and a[i-1,j+1,k] == 1:
                    b[i-1,j+1,k] = 0
                elif a[i,j,k] == 0 and a[i,j-1,k] == 1:
                    b[i,j-1,k] = 0
                elif a[i,j,k] == 0 and a[i,j+1,k] == 1:
                    b[i,j+1,k] = 0
                elif a[i,j,k] == 0 and a[i+1,j-1,k] == 1:
                    b[i+1,j-1,k] = 0
                elif a[i,j,k] == 0 and a[i+1,j,k] == 1:
                    b[i+1,j,k] = 0
                elif a[i,j,k] == 0 and a[i+1,j+1,k] == 1:
                    b[i+1,j+1,k] = 0
                elif a[i,j,k] == 0 and a[i-1,j-1,k+1] == 1:
                    b[i-1,j-1,k+1] = 0
                elif a[i,j,k] == 0 and a[i-1,j,k+1] == 1:
                    b[i-1,j,k+1] = 0
                elif a[i,j,k] == 0 and a[i-1,j+1,k+1] == 1:
                    b[i-1,j+1,k+1] = 0
                elif a[i,j,k] == 0 and a[i,j-1,k+1] == 1:
                    b[i,j-1,k+1] = 0
                elif a[i,j,k] == 0 and a[i,j,k+1] == 1:
                    b[i,j,k+1] = 0
                elif a[i,j,k] == 0 and a[i,j+1,k+1] == 1:
                    b[i,j+1,k+1] = 0
                elif a[i,j,k] == 0 and a[i+1,j-1,k+1] == 1:
                    b[i+1,j-1,k+1] = 0
                elif a[i,j,k] == 0 and a[i+1,j+1,k+1] == 1:
                    b[i+1,j+1,k+1] = 0
    return b
