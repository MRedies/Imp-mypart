#/usr/bin/ipython
import matplotlib.pyplot as plt
import numpy as np
import GreenF
import Impurity
from matplotlib import animation
import scipy.optimize as optimize
from scipy import integrate
import threading
from threading import Thread

class Param():
    def __init__(self, m, alpha, beta, B0):
        self.m     = m
        self.alpha = alpha
        self.beta  = beta
        self.B0    = B0

def circumf(a,b, end):
    U = lambda t:  np.sqrt( a**2 * ( np.sin(t)**2 + b**2/(a**2) * np.cos(t)**2))
    res, err = integrate.quad(U, 0, end)
    return res

def ellipse(a,b,n):
    global rank
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
        t        = optimize.bisect(f, 0.0, 2.0 * np.pi)
        x_arr[i] = x(t)
        y_arr[i] = y(t)

    return x_arr, y_arr

def foc(a,b):
    return np.sqrt(a**2 - b**2)

def setup_Gs(param,E, a, b, N):
    R                  = np.zeros((N, 2))
    R[:-1,0], R[:-1,1] = ellipse(a,b,N-1)
    R[-1,:]            = np.array([-foc(a,b), 0.0])

    V      = 0.23 * np.ones(N)
    B      = np.zeros((N,3))
    B[:,0] = 1e-6

    #without foc
    Iwithout = Impurity.Imp(R[:-1,:], V[:-1], B[:-1,:])
    Gwithout = GreenF.GF(para.m, para.alpha, para.beta,
            para.B0, Iwithout, E, True)

    #with foc
    Iwith = Impurity.Imp(R, V, B)
    Gwith = GreenF.GF(para.m, para.alpha, para.beta,
            para.B0, Iwith, E, True)

    return Gwithout, Gwith

def calc_den(g, x, E):
    r = np.array([0.0, 0.0])
    z = np.zeros(x.shape)

    for j in range(x.shape[0]):
        r[0] = x[j]
        z[j] = g.Roh(r)
    return z

# initialization function: plot the background of each frame
def init():
    for line in lines:
        line.set_data([],[])
    #ttl.set_text('')
    return lines#, ttl

# animation function.  This is called sequentially
def animate(i):
    global a, E, para, N
    
    print(i)
	
    bs = np.linspace(2.0/5.0, 1.0, 20)
    b  = bs[i]

    G_wo, G_w = setup_Gs(para, E, a, b, N)
    x = np.linspace(-1.2 * a, 1.2 * a, 300)

    z_wo = calc_den(G_wo, x, E)
    z_w  = calc_den(G_w,  x, E)
    z_d  = z_w - z_wo

    #lines[0].set_data(x, z_wo)
    #lines[1].set_data(x, z_w)
    lines[2].set_data(x, z_d)
    lines[3].set_data([-foc(a,b), foc(a,b)], [0.0, 0.0])
    #ttl.set_text("B = %f, exc = %f"%(b, np.sqrt(1- b**2)))
    return lines#, ttl


N = 151
a = 1.0
b = 2.0/5.0
E = 0
para = Param(10.0, # m
        2.0,  # alpha
        0.0,  # beta
        2.0)  # B0

fig   = plt.figure()
ax    = plt.axes(xlim=(-1.2*a, 1.2*a), ylim=(-5, 5))
ttl = plt.title("nana")

lines = []
l, = ax.plot([], [], label="without atom")
lines.append(l)
l, = ax.plot([], [], label="with atom")
lines.append(l)
l, = ax.plot([], [], label="diff")
lines.append(l)
l, = ax.plot([], [], 'r.', label="focal points")
lines.append(l)

# call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, init_func=init,
        frames=20, interval=1000, blit=True)

anim.save('move_b.mp4', fps=1, extra_args=['-vcodec', 'libx264'])

plt.show()
