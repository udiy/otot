import numpy as np
import matplotlib.pyplot as plt

###############################################################################

def lpf(w, wc, filter_type="lpf"):
    """
    Filters array to get only specific frequecies as 1 values, rest 0.
    
    Parameters
    ----------
    w: np.array
        Frequency spectrum
    
    wc: int
        Cutoff frequency
    
    filter_type: str
        Filter type to use, either low pass (lpf) or high pass (hpf)
    """
    w = np.abs(w)
    wc = np.abs(wc)
    
    if filter_type == "lpf":
        return np.less_equal(w, wc).astype(np.int)
    elif filter_type == "hpf":
        return np.greater_equal(w, wc).astype(np.int)
    else:
        raise ValueError("filter_type not supproted")

###############################################################################

w = np.linspace(-np.pi, np.pi, 500)    # frequecny vector
wc = np.pi/6    # cutoff freq
freq_response = lpf(w, wc)

# define a function of a series of sinc functions to build the low pass filter 
H = lambda m,w: (np.sin(wc*m) * np.e**(-1j*w*m))/(np.pi*m)
Hm = lambda m,w: wc/np.pi + sum([H(i,w) for i in np.delete(np.arange(-m,m+1),m)])


# plot differend filters
plt.figure(figsize=(16,9))
plt.plot(w, freq_response, label="Ideal LPF")
for m in [2, 5, 15, 30]:
    plt.plot(w, np.real(Hm(m, w)), label=f"M = {m}")
    
plt.legend()
plt.show()