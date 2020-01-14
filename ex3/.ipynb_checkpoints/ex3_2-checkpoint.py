import numpy as np


def sinc_interp(t, xn, dt):
    
    """
    Perfroms a sinc interpolation
    
    Parameters
    ----------
    t: float
        Desired time at which we want to estimate the original signal
    
    xn: array-like
        Interpolation points values
    
    dt: float
        Sampling rate
    """
    
    return sum([xn[n] *  np.sinc(np.pi*(t-n*dt)/dt) for n in range(len(xn))])


# define function
f = lambda t: np.exp(-(t**2))

dt = 0.5    # sampling rate in seconds
t = np.arange(0,5,dt)    # sampling points
xn = f(t)

ti = 2.3    # the point we want to interpolate

analytic = f(ti)    # analytic result
diff5 = np.abs(f(ti) - sinc_interp(ti, xn[4:9], dt))    # 5 points
diff7 = np.abs(f(ti) - sinc_interp(ti, xn[3:10], dt))    # 7 points
diff9 = np.abs(f(ti) - sinc_interp(ti, xn[1:10], dt))    # 9 points


print(f"""
    Estimating exp(-t^2) at t = {ti}.
    
    Analytic result: {analytic}
    
    Difference from sinc interpolation
    ----------------------------------
    
    -\t5 points: {diff5}
    -\t7 points: {diff7}
    -\t9 points: {diff9}
""")