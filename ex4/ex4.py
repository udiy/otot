import inspect
import numpy as np
import matplotlib.pyplot as plt

##################################################################################################


def plot_H(H, fig_name=None):
    
    """
    """
    
    # define a generic complex number z with magnitude of 1
    r = 1
    w = np.linspace(0, 4*np.pi, 1000)
    z = r * np.exp(1j * w)

    real_H = np.real(H(z))    # real part
    img_H = np.imag(H(z))    # imaginary part
    phi = np.arctan( img_H/real_H )    # phase


    # plot
    plt.plot(w, real_H, label="Real part")
    plt.plot(w, img_H, label="Imaginary part")
    plt.plot(w, phi, label="Phase")

    # layout
    plt.grid()
    plt.legend()
    
    # get function defintion as a string and set as title
    str_func = inspect.getsource(H)
    str_func = str_func.replace(" = lambda ", "(").replace(":", ") = ")
    plt.title(str_func)
    
    # save figure as png, if fig_name is provided
    if fig_name:
        plt.savefig(f"{fig_name}.png")
    
    plt.show()


##################################################################################################


H1 = lambda z: 1 - z / ( 1.25 * np.exp(1j * (2/3)*np.pi) )
H2 = lambda z: 1 - z * 1.25 * np.exp( -1j * (2/3)*np.pi)

plot_H(H1)
plot_H(H2)