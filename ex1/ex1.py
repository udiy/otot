import numpy as np


###############################################################################

def conv(f, g):
    
    """
    Perform convolution for series f and g
    
    Parameters
    ----------
    f, g: array-like
        List of numbers
    """
    
    # cast to numpy arrays to perform vectorized operations
    f = np.array(f)
    g = np.array(g)
    
    n = len(f)
    m = len(g)
    
    # initialize a zero valued array for output
    y = np.zeros(m+n-1)
    
    for i in range(n):
        
        # multiply g by ith element of f
        yi = f[i] * g
        
        # pad array with zeros for unused indicies
        yi = np.pad(yi, (i,n-i-1), "constant")
        
        # add to final result
        y += yi
        
    return y


###############################################################################

f = [-2, 7, 5]
g = [3, -2, 1, 4]

print(conv(f, g))