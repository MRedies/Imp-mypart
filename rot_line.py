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

def plot_k(g,ax, E0):
    E  = np.linspace(-10, 10, 100)
    k1 = np.zeros(E.shape, dtype=np.complex_)
    k2 = np.zeros(E.shape, dtype=np.complex_)
    
    for i in range(E.shape[0]):
        k1[i], k2[i] = g.find_ks(E[i] + g.eta)
    ax.plot(E, np.real(k1))
    ax.plot(E, np.real(k2))
    ax.plot([E0,E0],ax.get_ylim(), 'k:')
    ax.set_xlabel("E")
    ax.set_ylabel("k")

def plot_line(ax, g,E, theta):
    x = np.linspace(0.2, 3.0, 300)
    ax.plot(x, line(g,x,E,theta), label=r"$\theta$ = %0.1f"%(theta / np.pi * 180))
    ax.set_xlabel("r")
    ax.set_ylabel(r"$\Delta \rho$")

def latex(i, mag):
    print(r"\begin{figure}")
    print(r"\centering")
    print(r"\includegraphics[width=0.7\textwidth]{img/Set_" \
            + str(i+1) \
            + r"_Bx_mag_" + str(mag) + ".pdf}")
    print(r"\caption{Set %d , magnetic "%(i+1) + str(mag) + " Bx}")
    print(r"\end{figure}")
    print(r"\begin{figure}")
    print(r"\centering")
    print(r"\includegraphics[width=0.7\textwidth]{img/Set_" \
            + str(i+1) \
            + r"_Bz_mag_" + str(mag) + ".pdf}")
    print(r"\caption{Set %d , magnetic "%(i+1) + str(mag) + " Bz}")
    print(r"\end{figure}")

comm   = MPI.COMM_WORLD
rank   = comm.Get_rank()
nprocs = comm.Get_size()
 
N = 1
V = 0.23 * np.ones(N)
R = np.array([[0.0, 0.0]])

B      = np.zeros((N,3))
B[:,2] = V[0]

I      = Impurity.Imp(R,V,B)

m     = 10.0 * np.ones(5)
alpha = np.array([1E-3, 1.0,  1E-3, 2.0,  1E-3])
beta  = np.array([1E-3, 1E-3, 1.0,  1E-3, 1.0])
B0    = np.array([1.0,  0.0,  0.0,  1.0,  2.0])
mag   = False
t     = np.linspace(0, 2 * np.pi, 4)[:-1]
E = 2.5
for i in range(5):
    f, ax1 = plt.subplots(1, 1)
    g = GreenF.GF(m[i], alpha[i], beta[i], B0[i], I,E, mag, nprocs, rank, comm)
    #plot_k(g, ax1, E)
    for theta in t:
        plot_line(ax1,g, E, theta)
        #f.title("Set = %d, E = %f"%(i+1, E))
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/Set_%d_Bz_mag_"%(i+1) + str(mag) +".pdf")
    plt.clf()
    #plt.show()
    latex(i, mag)

