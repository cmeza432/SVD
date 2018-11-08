"""
@author: Carlos Meza
"""

import numpy as np

def svdStudent(A, b):
    # i -- Solution 1 ---> Use least square method, find x1 and norm
    x1 = np.linalg.lstsq(A, b)[0]
    norm1 = np.linalg.norm(x1)

    
    # ii -- Solution 2 ---> Use SVD method, find x2 and norm
    numcol = len(A[0])
    u, s, v = np.linalg.svd(A)
    c = np.dot(u.T, b)
    c.resize((numcol, 1))
    w = np.linalg.solve(np.diag(s), c)
    x2 = np.dot(v.T, w)
    norm2 = np.linalg.norm(x2)
    
    
    # iii -- Use SVD, find rank of A
    S = s >= 0.001
    erank = len(s[S])
    
    
    # iv -- Solution 3 ---> Find x3 using rank of A and norm
    rank = len(A[0])
    c = np.dot(u.T, b)
    c.resize((rank, 1))
    w = np.linalg.solve(np.diag(s), c)
    x3 = np.dot(v.T, w)
    norm3 = np.linalg.norm(x2)
    
    # Return all values
    return x1, norm1, x2, norm2, erank, x3, norm3