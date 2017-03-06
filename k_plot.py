import numpy as np
import matplotlib.pyplot as plt
import GreenF

a = GreenF.GF(10.0,1.0, 0.0, 1.0)

E  = np.linspace(-10,10,1000)
k1 = np.zeros(E.shape, dtype=np.complex128)
k2 = np.zeros(E.shape, dtype=np.complex128)

for i in range(E.shape[0]):
    k1[i], k2[i] = a.find_ks(E[i] + 0.1 * 1j)

plt.plot(E,np.real(k1), label="K1 RE")
plt.plot(E,np.imag(k1), label="K1 IM")
plt.plot(E,np.real(k2), label="K2 RE")
plt.plot(E,np.imag(k2), label="K2 IM")
plt.xlabel("E")
plt.ylabel("k")
plt.legend()
plt.show()



