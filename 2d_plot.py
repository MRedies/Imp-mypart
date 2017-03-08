#/usr/bin/ipython
from sys import exit
from mpi4py import MPI
import numpy as np
import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy.linalg as la
from scipy import integrate
import scipy.optimize as optimize
from mpl_toolkits.mplot3d import Axes3D
import GreenF
import Impurity

class Param():
    def __init__(self, m, alpha, beta, B0):
        self.m     = m
        self.alpha = alpha
        self.beta  = beta
        self.B0    = B0

def calc_den_distr(g,E, xpart, ypart, x_num, y_num, rank):
    zpart = calc_den_loc(g, xpart, ypart, E)
    X = recomb_grid(x_num, y_num, rank, xpart)
    Y = recomb_grid(x_num, y_num, rank, ypart)
    Z = recomb_grid(x_num, y_num, rank, zpart)
    if rank == 0:
            return X,Y,Z
    else:
            return None, None, None

def calc_den_loc(g, x_part, y_part, E):
    r      = np.array([0.0, 0.0])
    z_part = np.zeros(x_part.shape)

    for j in range(x_part.shape[0]):
        r[0]      = x_part[j]
        r[1]      = y_part[j]
        z_part[j] = g.Roh(r)
    return z_part

def plot_den(X,Y,Z, Set, E, ax, f):
    lvls = np.linspace(np.min(Zarr[i]), np.max(Zarr[i]), 20)
    cont = ax.contourf(X, Y, Z, cmap=cm.viridis, levels=lvls)
    f.colorbar(cont, ax=ax)
    #ax.set_ylabel("Set: %d, E = %2.2f"%(Set+1, E))

def plot_k(g,ax, f):
    E  = np.linspace(-10, 10, 100)
    k1 = np.zeros(E.shape, dtype=np.complex_)
    k2 = np.zeros(E.shape, dtype=np.complex_)
    
    for i in range(E.shape[0]):
        k1[i], k2[i] = g.find_ks(E[i] + g.eta)
    ax.plot(E, np.real(k1))#, label="k1")
    ax.plot(E, np.real(k2))#, label="k2")
    # ax.set_xlabel("E")
    # ax.set_ylabel("k")
    ax.legend()

def create_den_plot(g,x,y, Set, E, comm):
    rank     = comm.Get_rank()
    nprocs   = comm.Get_size()
    dim_num  = x.shape[0]

    #create k-plot
    if rank == 0:
        f, axarr = plt.subplots(1,2, figsize=(8,4))
        #axarr_flat = [axarr[1,0], axarr[0,1], axarr[1,1]]
        plot_k(g, axarr[0], f)
        y_l, y_u = axarr[0].get_ylim()
        axarr[0].plot([E,E],[y_l,y_u], "k:")#,label="E")
        axarr[0].legend()
    
    #distrib grid
    x_part, y_part = SplitSpread_grid(x,y, comm)
    X, Y, Z = calc_den_distr(g,E, x_part, y_part,
            dim_num, dim_num, rank)
    if rank == 0:
        global R
        axarr[1].plot(R[:,0],R[:,1], '.')
        axarr[1].plot(-R[-1,0], R[-1,1], 'r.')
        lvls = np.linspace(np.min(Z), np.max(Z), 20)
        cont = axarr[1].contourf(X,Y,Z, cmap=cm.coolwarm, levels=lvls)
        f.colorbar(cont, ax=axarr[1])
        #axarr[1].set_title("Set: %d  E = %2.2f"%(Set+1,E))
        
        plt.show()

def SplitSpread_grid(x,y, comm):
    rank   = comm.Get_rank()
    nprocs = comm.Get_size()
    if rank == 0:
        X, Y = np.meshgrid(x,y)
        xf   = X.flatten()
        yf   = Y.flatten()
        xsp  = np.array_split(xf, nprocs)
        ysp  = np.array_split(yf, nprocs)
    else:
        x = y = xsp = ysp = None
    x_part = comm.scatter(xsp, root=0)
    y_part = comm.scatter(ysp, root=0)
    return x_part, y_part


def recomb_grid(x_num, y_num, rank, part):
	grid = comm.gather(part, root=0)
	if rank == 0:
		grid = np.hstack(grid)
		grid = grid.reshape(x_num, y_num)
		return grid
	else:
		return None




def calc_mag_loc(g, x_part, y_part):
    E = g.E
    r       = np.array([0.0, 0.0])
    u       = np.zeros(x_part.shape)
    v       = np.zeros(x_part.shape)
    w       = np.zeros(x_part.shape)
    z_ratio = np.zeros(x_part.shape)

    for j in range(x_part.shape[0]):
        r[0] = x_part[j]
        r[1] = y_part[j]
        tmp  = g.Ms(r)
        
        u[j] = tmp[0]
        v[j] = tmp[1]
        w[j] = tmp[2]
        z_ratio[j] = w[j]/np.sqrt(tmp[0]**2 + tmp[1]**2 + tmp[2]**2)
    return u, v, w, z_ratio


def calc_mag_distr(g,E, xpart, ypart, x_num, y_num, rank):
    u,v,w,z_ratio = calc_mag_loc(g, xpart, ypart)
    X = recomb_grid(x_num, y_num, rank, xpart)
    Y = recomb_grid(x_num, y_num, rank, ypart)
    U = recomb_grid(x_num, y_num, rank, u)
    V = recomb_grid(x_num, y_num, rank, v)
    W = recomb_grid(x_num, y_num, rank, w)

    Z_ratio = recomb_grid(x_num, y_num, rank, z_ratio)
    if rank == 0:
            return X,Y,U,V,W,Z_ratio
    else:
            return None, None, None, None, None, None

def create_mag_plot_stream(g,x,y, Set, comm):
    E = g.E
    rank     = comm.Get_rank()
    nprocs   = comm.Get_size()
    dim_num  = x.shape[0]

    #create k-plot
    if rank == 0:
        f, axarr = plt.subplots(2,1, figsize=(8,8))
        plot_k(g, axarr[0], f)
        a,b = axarr[0].get_ylim()
        axarr[0].plot([E,E],[a,b], "k:")#,label="E")
        axarr[0].legend()
    
    #distrib grid
    x_part, y_part = SplitSpread_grid(x,y, comm)

    
    X, Y, U, V, W, Z_ratio = calc_mag_distr(g,E, x_part, y_part,
            dim_num, dim_num, rank)
    if rank == 0:
        global R
        axarr[1].plot(R[:,0],R[:,1], '.')
        axarr[1].plot(-R[-1,0], R[-1,1], 'r.')
        strm = axarr[1].streamplot(X,Y,U,V, color=Z_ratio,
                        cmap=cm.viridis)
        f.colorbar(strm.lines, ax=axarr[1])
        #axarr[1].set_title("Set: %d  E = %2.2f"%(Set+1,np.real(E)))
        axarr[1].set_xlim(np.min(X), np.max(X))
        axarr[1].set_ylim(np.min(Y), np.max(Y))
        plt.show()

def create_mag_plot_contour(g,x,y, Set, name, comm):
    E = g.E
    rank     = comm.Get_rank()
    nprocs   = comm.Get_size()
    dim_num  = x.shape[0]


    #create k-plot
    if rank == 0:
        f, axarr = plt.subplots(2,2, figsize=(8,8))
        plot_k(g, axarr[0,0], f)
        a,b = axarr[0,0].get_ylim()
        axarr[0,0].plot([E,E],[a,b], "k:")#,label="E")
        axarr[0,0].legend()
    #f.title("Set: %d  E = %2.2f"%(Set+1,E))
    #distrib grid
    x_part, y_part = SplitSpread_grid(x,y, comm)
    X, Y, U, V, W, Z_ratio = calc_mag_distr(g,E, x_part, y_part,
            dim_num, dim_num, rank)
    if rank == 0:
        U = np.real(U)
        V = np.real(V)
        W = np.real(W)

        R = np.sqrt(U*U + V*V)
        phi = np.arctan2(V,U)
       
        lvls = np.linspace(np.min(U), np.max(U), 20)
        cont = axarr[0,1].contourf(X,Y,U, cmap=cm.coolwarm, levels=lvls)
        f.colorbar(cont, ax=axarr[0,1])

        lvls = np.linspace(np.min(V), np.max(V), 20)
        cont = axarr[1,0].contourf(X,Y,V, cmap=cm.coolwarm, levels=lvls)
        f.colorbar(cont, ax=axarr[1,0])
        
        lvls = np.linspace(np.min(W), np.max(W), 20)
        cont = axarr[1,1].contourf(X,Y,W, cmap=cm.coolwarm, levels=lvls)
        f.colorbar(cont, ax=axarr[1,1])
        
        axarr[0,0].set_title("Set: %d  E = %2.2f"%(Set+1,np.real(E)))
        axarr[0,1].set_title("X-component")
        axarr[1,0].set_title("Y-component")
        axarr[1,1].set_title("Z-component")

        plt.tight_layout()
        plt.savefig(name)
        plt.clf()
        #plt.show()


def create_mag_plot_nice(g,x,y, Set, name, comm):
    E = g.E
    rank     = comm.Get_rank()
    nprocs   = comm.Get_size()
    dim_num  = x.shape[0]


    #create k-plot
    if rank == 0:
        f, axarr = plt.subplots(2,1, figsize=(4,6))
        # plot_k(g, axarr[0], f)
        # a,b = axarr[0].get_ylim()
        # axarr[0].plot([E,E],[a,b], "k:")#,label="E")
        # axarr[0].legend()

    #distrib grid
    x_part, y_part = SplitSpread_grid(x,y, comm)
    X, Y, U, V, W, Z_ratio = calc_mag_distr(g,E, x_part, y_part,
            dim_num, dim_num, rank)
    if rank == 0:
        U = np.real(U)
        V = np.real(V)
        W = np.real(W)

        R = np.sqrt(U*U + V*V)
       
        strm = axarr[0].streamplot(X, Y, U, V, linewidth=2)

        lvls = np.linspace(np.min(W), np.max(W), 20)
        cont = axarr[1].contourf(X,Y,W, cmap=cm.viridis, levels=lvls)
        f.colorbar(cont, ax=axarr[1])
        
        axarr[0].set_title("X-Y-stream")
        axarr[1].set_title("Z-component")

        #plt.tight_layout()
        plt.savefig(name)
        plt.clf()
        #plt.show()

def circumf(a,b, end):
    U = lambda t:  np.sqrt( a**2 * ( np.sin(t)**2 + b**2/(a**2) * np.cos(t)**2))
    res, err = integrate.quad(U, 0, end)
    return res

def ellipse(a,b,n):
    global rank
    if rank == 0:
        print("Excentricity: %f"%(np.sqrt(1.0 - b*b/ (a*a) )))
    x = lambda t: a * np.cos(t)
    y = lambda t: b * np.sin(t)
    
    U     = circumf(a,b,2 * np.pi)
    U_arr = np.linspace(0.0, U, n+1)[:-1]
    dU    = U_arr[1] - U_arr[0]

    x_arr = np.zeros(n)
    y_arr = np.zeros(n)

    x_arr[0] = x(0.0)
    y_arr[0] = y(0.0)

    for i in range(1,n):
        f        = lambda t: circumf(a,b,t) - i*dU
        # print(circumf(a,b,0.0) - i*dU)
        # print("i*dU = %f"%(i*dU))
        # print("f(0) = %f   f(2pi) = %f"%(f(0.0), f(2 * np.pi)))
        t        = optimize.bisect(f, 0.0, 2.0 * np.pi)
        x_arr[i] = x(t)
        y_arr[i] = y(t)

    return x_arr, y_arr

def foc(a,b):
    return np.sqrt(a**2 - b**2)

def calc_diff_den(E,a,b,N, para, prefix):
    global comm, rank, nprocs
    R = np.zeros((N, 2))
    R[:-1,0],R[:-1,1] = ellipse(a,b,N-1)
    R[-1,:] = np.array([-foc(a,b),0.0])


    # init values
    V = 0.23 * np.ones(N)
    B      = np.zeros((N,3))
    B[:,0] = 1e-6

    dim_num = 300
    x = np.linspace(-1.20 * a, 1.2 * a, dim_num)
    #y = np.linspace(-1.20 * b, 1.2 * b, dim_num) 
    y = np.array([0.0])

    x_dim = dim_num
    y_dim = 1 #dim_num

    I = Impurity.Imp(R,V,B)
    g = GreenF.GF(para.m, para.alpha, para.beta, para.B0, I, E, True,
                  nprocs, rank, comm)
    
    #distrib grid
    x_part, y_part = SplitSpread_grid(x,y, comm)
    X, Y, den_full = calc_den_distr(g,E, x_part, y_part,
            x_dim, y_dim, rank)
    
    I = Impurity.Imp(R[:-1,:], V[:-1],B[:-1,:])
    g = GreenF.GF(para.m, para.alpha, para.beta, para.B0, I, E, True,
                  nprocs, rank, comm)
    _, _, den_ell = calc_den_distr(g,E, x_part, y_part,
            x_dim, y_dim, rank)

    if rank == 0:
        den_diff = den_full - den_ell
        np.savez(prefix + ".npz", den_full, den_ell, den_diff, R, X, Y)

def den_scan(comm, rank, nprocs):
    m     = 10.0 * np.ones(5)
    alpha = np.array([1E-3, 1.0,  1E-3, 2.0,  1E-3 ])
    beta  = np.array([1E-3, 1E-3, 1.0,  1E-3, 1.0])
    B0    = np.array([1.0,  0.0,  0.0,  1.0,  2.0])

    para = Param(m[3], alpha[3], beta[3], B0[3])
    N = 81 


    E = np.linspace(-1.5, 1.5, 300) 
    lamb = 2*np.pi /40.0
    a = 0.64
    b = 4.0/5.0 * a 

    for i in range(E.shape[0]):
        prefix = "81_E=%1.5f"%(E[i])
        if rank == 0:
            print("%d  %1.5f"%(i, E[i]))
        calc_diff_den(E[i],a, b, N, para, prefix)

def latex(name, mag, Set):
    new_name = "img" + name[5:]
    print(r"\begin{figure}")
    print(r"\centering")
    print(r"\includegraphics[width=0.7\textwidth]{" + \
            new_name + "}")
    print(r"\caption{Set %d "%(Set+1) + " magnetization " +\
            str(mag) + " Bnone}")
    print("\end{figure}")

def mag_plot(comm,  rank, nprocs, Set, mag, E):
    m     = 10.0 * np.ones(5)
    alpha = np.array([1E-8, 1.0,  0.0,    2.0, 0.0 ])
    beta  = np.array([1E-8, 0.0,  1.0,    0.0, 1.0])
    B0    = np.array([1.0,  0.0,  0.0,    1.0, 2.0])

    N      = 1
    R      = np.array([[0.0, 0.0]])
    V      = 0.23 * np.ones(N)
    B      = np.zeros((N,3))
    #B[:,2] = V[0]
    x      = y = np.linspace(-1.2,1.2,100)

    name = "plots/magplot_Set_%d_mag"%(Set+1) +\
            str(mag) + \
            "_Bnone.pdf"
    # if rank == 0:
        # latex(name, mag, Set)
    I = Impurity.Imp(R, V,B)
    g = GreenF.GF(m[Set], alpha[Set], beta[Set], B0[Set], I, E, mag,
              nprocs, rank, comm)
    create_mag_plot_nice(g, x, y, Set, name, comm)

comm   = MPI.COMM_WORLD
rank   = comm.Get_rank()
nprocs = comm.Get_size()


for i in range(5):
    E = 2.5 
    mag_plot(comm, rank, nprocs, i, False, E)
    if rank == 0:
        print(i)
