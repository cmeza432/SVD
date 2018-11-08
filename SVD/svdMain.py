"""
Carlos Meza
"""

import numpy as np
from svdStudent import *

#  use SVD to find least squares solution of a rank-deficient matrix
#

A = np.array([[64, 31,  9,  4,  3, 39, -15],
              [38, 92, 26, 89, 74, 270, 218],
              [81, 43, 80, 91, 50, 225, -40],
              [53, 18,  3, 80, 48, 178, 183],
              [35, 90, 93, 10, 90, 110, 81],
              [94, 98, 73, 26, 61, 150, 25],
              [88, 44, 49, 34, 62, 112, 101],
              [55, 11, 58, 68, 86, 147, 170],
              [62, 26, 24, 14, 81, 54, 252],
              [59, 41, 46, 72, 58, 185, 94],
              [21, 59, 96, 11, 18, 81, -216],
              [30, 26, 55, 65, 24, 156, -69],
              [47, 60, 52, 49, 89, 158, 200],
              [23, 71, 23, 78,  3, 227, -57],
              [84, 22, 49, 72, 49, 166, 49],
              [19, 12, 62, 90, 17, 192, -118],
              [23, 30, 68, 89, 98, 208, 188],
              [17, 32, 40, 33, 71, 98, 164],
              [23, 42, 37, 70, 50, 182, 89],
              [44, 51, 99, 20, 47, 91, -109]])

b = np.array([-311, -148, -1509, 159, 451, -407, 356, 673, 1871, -335, -1761, -1416, 784, -2108, -509, -2072, 369, 856, -449, -915])
  
#
#  DO NOT MODIFY THIS FILE
#  create the file svdStudent.py
#

x1, norm1, x2, norm2, erank, x3, norm3 = svdStudent(A, b)

print('part a: solution using least squares solver\n')
print(x1)
print('\nnorm of residual = %.3e\n\n' % norm1)


print('part b: solution using n columns of U\n')
print(x2)
print('\nnorm of residual = %.3e\n\n' % norm2)


print('part c: effective rank\n')
print('%d\n\n' % erank)


print('part d: solution using r columns of U\n')
print(x3)
print('\nnorm of residual = %.3e\n\n' % norm3)

