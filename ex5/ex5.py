import numpy as np

#################################################################################################################

def cyc_conv(f, g):
    
    """
    Perform a cyclic convolution for f and g
    
    Parameters
    ----------
    
    f, g: array-like
    """
    
    # check length of f and g
    m = len(f)
    n = len(g)
    
    
    # if f and g are not of the same length, zero padding is needed
    if m != n:
        N = m + n - 1
        f = np.pad(f, (0, N - m))
        g = np.pad(g, (0, N - n))
    else:
        N = m
        
        
    # initialize an array for results
    h = np.zeros(N)

    for i in range(N):
        h[i] = sum([f[j] * g[(i-j)%N] for j in range(N)])
        
    return h

#################################################################################################################

def dft(x):
    
    """
    Perfom DFT for x
    
    Parameters
    ----------
    x: array-like
    """
    
    N = len(x)
    
    # array for results
    X = np.zeros(N, dtype=complex)
    
    # define exponent factor
    W = np.exp(-1j * (2*np.pi / N))
    
    for k in range(N):
        
        X[k] = sum([x[n]*(W**(k*n)) for n in range(N)])
        
    return X

#################################################################################################################

def idft(X):
    
    """
    Inverse DFT for X
    
    Parameters
    ----------
    X: array-like
    """
    
    N = len(X)
    
    # array for results
    x = np.zeros(N, dtype=complex)
    
    # define exponent factor
    W = np.exp(-1j * (2*np.pi / N))
    
    for n in range(N):
        
        x[n] = sum([X[k]*(W**(-k*n)) for k in range(N)]) / N
        
    return x

#################################################################################################################

# define arrays
x = [1, -1, -1, -1, 1, 0, 1, 2]
y = [5, -4, 3, 2, -1, 1, 0, -1]


# Direct convolution
result1 = cyc_conv(x, y)

print(f"""
    #########################
    Direct convolution result:

    {result1}
    """)


# Using DFT, multiplying in frequency domain, then iDFT
X = dft(x)
Y = dft(y)
result2 = idft(X * Y)

np.set_printoptions(suppress=True)

print(f"""
    #########################
    Results from the process of
        1. Perform DFT for x and y seperately
        2. Multiply X and Y (frequency domain)
        3. Perform inverse DFT for the multiplication of X and Y

    {result2}
    """)