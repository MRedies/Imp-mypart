import numpy as np
import scipy.special as sf
import numpy.linalg as la
import matplotlib.pyplot as plt
import GreenF

R     = 1e-4
eta   = 1e-2
m     = 10
alpha = 1.0
beta  = 0.0
B0    = 1.0

G   = GreenF.GF(m, alpha, beta, B0)
E   = np.linspace(0, 8, 300)

z  = lambda E: E + 1j * eta
f1 = lambda E: np.imag(1j * np.abs( G.kp(E))/ G.D_prim(z(E), G.kp(E)) \
        * sf.hankel1(0, R * G.kp(E)) * z(E))
f2 = lambda E: np.imag(1j * np.abs( G.kp(E))/ G.D_prim(z(E), G.kp(E)) \
        * sf.hankel1(0, R * G.kp(E)) * (- G.kp(E)**2 / (2.0 * m))) 


plt.plot(E, f1(E), label="z term")
plt.plot(E, f2(E), label="k**2 term")
plt.legend()
plt.show()
