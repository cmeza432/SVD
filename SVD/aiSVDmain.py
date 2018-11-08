"""
@author: Carlos Meza
"""

import numpy as np
from aiSVDstudent import *

#
#  DO NOT MODIFY THIS FILE
#  create the file ai_svd_student.m
#

pClass = np.array([[0, 0, 0],
                   [1, 1, 1],
                   [0, 0, 1],
                   [1, 1, 1],
                   [1, 1, 1],
                   [1, 1, 0],
                   [0, 0, 0],
                   [1, 1, 1],
                   [0, 0, 0]])
xClass = np.array([[1, 1, 1],
                   [0, 0, 0],
                   [1, 1, 1],
                   [0, 0, 0],
                   [1, 1, 1],
                   [0, 0, 0],
                   [1, 0, 1],
                   [0, 0, 0],
                   [1, 1, 1]])

test1 = np.array([0, 1, 0, 1, 1, 1, 1, 1, 0])  # plus
test2 = np.array([1, 0, 1, 0, 1, 1, 1, 0, 1])  # X
k = 2  # number of columns of U to use

Ukp, Ukx, np1, nx1, np2, nx2 = aiSVDstudent(k, pClass, xClass, test1, test2)

print('U from plus class\n')
print(Ukp)

print('\nU from x class\n')
print(Ukx)

print('training plus class, test vector 1\n')
print('%.2f\n\n' % np1)

print('training x class, test vector 1\n')
print('%.2f\n\n' % nx1)

print('training plus class, test vector 2\n')
print('%.2f\n\n' % np2)

print('training x class, test vector 2\n')
print('%.2f\n' % nx2)