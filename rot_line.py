#/usr/bin/ipython
from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy.linalg as la
from mpl_toolkits.mplot3d import Axes3D
import GreenF
import Impurity


def line(g,x,E, theta):
    M = np.array([[np.cos(theta), - np.sin(theta)],
                     [np.sin(theta),   np.cos(theta)]])

    roh = np.zeros(x.shape)

    for i in range(x.shape[0]):
        r      = np.array([0.0, 0.0])
        r[0]   = x[i]
        r      = np.dot(M,r)
        roh[i] = g.Roh(r)
    return roh

def plot_line(g,E, theta):
    x = np.linspace(0.2, 3.0, 300)
    plt.plot(x, line(g,x,E,theta), label=r"$\theta$ = %f"%(theta / np.pi * 180))

 
N = 1
V = 0.23 * np.ones(N)
R = np.array([[0.0, 0.0]])
#R = np.array([[0.0,    1.0],
              # [-0.781, 0.623],
              # [-0.974, -0.222],
              # [-0.433, -0.900],
              # [0.433,  -0.900],
              # [0.974,  -0.222],
              # [0.781,  0.623],
              # [-0.5, 0.0]])

B      = np.zeros((N,3))
B[:,0] = V[0]

I      = Impurity.Imp(R,V,B)

m     = 10.0 * np.ones(5)
alpha = np.array([1E-3, 1.0,  1E-3, 2.0,  1E-3])
beta  = np.array([1E-3, 1E-3, 1.0,  1E-3, 1.0])
B0    = np.array([1.0,  0.0,  0.0,  1.0,  2.0])
t = np.linspace(0, 2 * np.pi, 5)

E = 2.5
for i in range(5):
    g = GreenF.GF(m[i], alpha[i], beta[i], B0[i], I,E, True)
    for theta in t:
        plot_line(g, E, theta)
        plt.title("Set = %d, E = %f"%(i+1, E))
    plt.legend()
    plt.show()

