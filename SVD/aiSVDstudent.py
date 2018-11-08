"""
@author: Carlos Meza
"""

import numpy as np

def aiSVDstudent(k, pClass, xClass, test1, test2):
    # Find SVD of the training matrices
    Ukp, s1, v1 = np.linalg.svd(pClass)
    Ukx, s2, v2 = np.linalg.svd(xClass)
    # Number of columns to use == k
    rowl = len(pClass)
    Ukp.resize((rowl, k))
    Ukx.resize((rowl, k))    
    # Set up Eq 1
    ident = np.identity(len(Ukp))
    U1 = np.dot(Ukp, Ukp.T)
    U2 = np.dot(Ukx, Ukx.T)
    temp1 = ident - U1 
    temp2 = ident - U2
    # Calculate Eq 1 for test 1
    nx1 = np.multiply(temp1, test1)
    nx1 = np.linalg.norm(nx1)
    np1 = np.multiply(temp2, test1)
    np1 = np.linalg.norm(np1)
    # Calculate Eq 1 for test 2
    nx2 = np.multiply(temp1, test2)
    nx2 = np.linalg.norm(nx2)
    np2 = np.multiply(temp2, test2)
    np2 = np.linalg.norm(np2)
    # Return all values
    return Ukp, Ukx, np1, nx1, np2, nx2