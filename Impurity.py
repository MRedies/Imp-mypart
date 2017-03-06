import numpy as np
import numpy.linalg as la


class Imp:
    def __init__(self, R, V, B, width = 0.1):
        self.R     = R
        self.V     = V
        self.B     = B
        self.width = width
        self.n_imp = V.shape[0]

        self.T     = np.zeros((2*self.n_imp, 2*self.n_imp),
                        dtype=np.complex_)

        self.sigma_0 = np.identity(2, dtype=np.complex_)
        self.sigma_x = np.array([[0, 1],[1, 0]], dtype=np.complex_)
        self.sigma_y = np.array([[0, -1j],[1j, 0]], dtype=np.complex_)
        self.sigma_z = np.array([[1, 0],[0, -1]], dtype=np.complex_)
    
    def d_H(self, r):
        result = np.zeros((2,2), dtype=np.complex_)
        
        for i in range(self.R.shape[0]):
            single = np.zeros((2,2), dtype=np.complex_)

            single += self.V[i]   * self.sigma_0
            single += self.B[i,0] * self.sigma_x
            single += self.B[i,1] * self.sigma_y
            single += self.B[i,2] * self.sigma_z

            dR = la.norm(r - self.R[i,:])
            single *= self.delta_lor(dR)

            result += single

        return result
    
    def An(self, n):
        result = np.zeros((2,2), dtype=np.complex_)
        
        result += self.V[n]   * self.sigma_0
        result += self.B[n,0] * self.sigma_x
        result += self.B[n,1] * self.sigma_y
        result += self.B[n,2] * self.sigma_z

        return result 
    def delta_lor(self, t):
        return 1.0/np.pi * self.width /(t**2 + self.width**2)

    def delta_gau(self, t):
        return 1.0/(np.sqrt(2.0 * np.pi) * self.width) \
                * np.exp(-t**2 /(2.0 * self.width**2))

    def set_diagT(self, m, mag):
        for i in range(0, 2*self.n_imp, 2):
            n = int(i/2)
            self.T[i:i+2,i:i+2] = self.inv_t(n, m, mag)

    def inv_t(self,n,m, mag):
        if(mag  == False):
            t = 1j * self.sigma_0
        else:
            b_norm  = 1.0 / la.norm(self.B[n,:])
            t       = 1j * b_norm * self.B[n,0] * self.sigma_x
            t      += 1j * b_norm * self.B[n,1] * self.sigma_y
            t      += 1j * b_norm * self.B[n,2] * self.sigma_z
        #print("m = %f + i %f"%(np.real(m), np.imag(m) ))
        return 2.0*t/(m)
