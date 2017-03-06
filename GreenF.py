import numpy as np
import scipy.special as sf
import numpy.linalg as la
import time
from sys import exit
import matplotlib.pyplot as plt
from mpi4py import MPI

def my_Hankel(n, k, abs_R):
    z = k*k
    if(np.imag(z) > 0):
        return  sf.hankel1(n, k * abs_R)
    else:
        return -sf.hankel2(n, k * abs_R)

class GF:
    def __init__(self, m, alpha, beta, B0, Impur, E, mag,
                nproc, rank, comm, 
            Ef =1.0, eta = 1e-5, R_to_0 = 1e-6):
        self.m      = complex(m,     0)
        self.alpha  = complex(alpha, 0)
        self.beta   = complex(beta,  0)
        self.B0     = complex(B0,    0)
        self.Impur  = Impur
        self.E      = complex(np.real(E), eta)
        self.Ef     = Ef
        self.eta    = eta
        self.R_to_0 = R_to_0
        self.E_so   = self.m * (alpha**2 + beta**2)
        self.E0     = - (self.E_so**2 + self.B0**2)/(2 * self.E_so)

        self.sigma_0 = np.identity(2)
        self.sigma_x = np.array([[0, 1],[1, 0]])
        self.sigma_y = np.array([[0, -1j],[1j, 0]])
        self.sigma_z = np.array([[1, 0],[0, -1]])

        self.nproc = nproc
        self.rank  = rank
        self.comm  = comm
        
        #time1 = time.time()
        self.set_invT(mag)
        #time2 = time.time()
        #if rank == 0:
        #    print("Time: %f"%(time2-time1))

        # print("n = %d"%(self.Impur.n_imp))

    def find_ks(self, z):
        inner_sqrt = np.sqrt(self.E_so**2 + 2*self.E_so*z + self.B0**2)
        k1 = np.sqrt(2*self.m * (z + self.E_so + inner_sqrt))
        k2 = np.sqrt(2*self.m * (z + self.E_so - inner_sqrt))
        return k1, k2

    def D(self, z, k):
        ksq_2m = (k**2)/(2.0 * self.m)
        return (z - ksq_2m)**2 - 2 * self.E_so * ksq_2m - self.B0**2

    def z_cross_R(self, R):
        return np.array([-R[1], R[0]])

    def N(self):
        E = self.E
        R = np.array([self.R_to_0, self.R_to_0])
        return - 1.0/np.pi * np.imag( np.trace(self.Gc(R, E)))

    
    def absk_Dpr(self):
        E = self.E
        k1, k2 = self.find_ks(E + self.eta)
        z1 = k1**2 /(2.0 * self.m)
        z2 = k2**2 /(2.0 * self.m)

        r1 = self.m/(z1 - z2) 
        r2 = self.m/(z2 - z1)
        return r1, r2

    def Gc(self, R):
        E = self.E
        z = E + self.eta * 1j
        k1, k2 = self.find_ks(z)
        R_hat = R / la.norm(R)
        abs_R = la.norm(R)
        ZxRpB = self.alpha * self.z_cross_R(R_hat) + self.beta * R_hat
        
        fraction1, fraction2 = self.absk_Dpr()

        res  = 0 * 1j
        res += 0.5 * 1j * fraction1 * my_Hankel(0, k1, abs_R) \
                * ((z - (k1**2)/(2.0 * self.m)) * self.sigma_0 + self.B0 * self.sigma_z)
        res += 0.5 * 1j * fraction2 * my_Hankel(0, k2, abs_R) \
                * ((z - (k2**2)/(2.0 * self.m)) * self.sigma_0 + self.B0 * self.sigma_z)
        
        res -= 0.5 * fraction1 * my_Hankel(1, k1, abs_R) \
                * k1 * (ZxRpB[0] * self.sigma_x + ZxRpB[1] * self.sigma_y)
        res -= 0.5 * fraction2 * my_Hankel(1, k2, abs_R) \
                * k2 * (ZxRpB[0] * self.sigma_x + ZxRpB[1] * self.sigma_y)
        return res


    def to_Rs(self, rx):
        result = np.zeros((2, 2*self.Impur.n_imp), 
                     dtype=np.complex_)
        for i in range(self.Impur.n_imp):
            mtx_i = 2 * i
            dR = self.Impur.R[i,:] - rx
            result[:,mtx_i:mtx_i+2] = self.Gc(dR)
        return result

    def from_Rs(self, ry):
        result = np.zeros((2*self.Impur.n_imp,2),
                    dtype=np.complex_)
        for i in range(self.Impur.n_imp):
            mtx_i = 2 * i
            dR = ry - self.Impur.R[i,:]
            result[mtx_i:mtx_i+2,:] = self.Gc(dR)
        return result

    def G(self, rx, ry):
        M = np.dot(self.to_Rs(rx), self.Impur.T)
        return np.dot(M, self.from_Rs(ry))

    def Roh(self, r):
        E = self.E
        return -1.0/np.pi * np.imag( np.trace(self.G(r, r)))

    def Ms(self, R):
        E = self.E
        result = np.zeros(3, dtype=np.float_)
        G_tmp = self.G(R,R)

        result[0] = np.imag(np.trace(np.dot(self.sigma_x, G_tmp)))
        result[1] = np.imag(np.trace(np.dot(self.sigma_y, G_tmp)))
        result[2] = np.imag(np.trace(np.dot(self.sigma_z, G_tmp)))

        result *= -1.0 / np.pi
        return result
    

    def I(self, R, V, m_tip):
        current = self.Roh(R)
        current += np.inner(m_tip, self.Ms(R))
        return current
    
    def set_invT(self, mag):
        E = self.E
        #init matrix 0
        self.Impur.T.fill(0.0) 
        self.set_Gpart_distr()
        #self.set_Gpart()
        self.Impur.set_diagT(self.m, mag)
        self.Impur.T = la.inv(self.Impur.T)

    def set_Gpart_distr(self):
        nproc = self.nproc
        rank  = self.rank
        comm  = self.comm

        mtx = np.zeros(self.Impur.T.shape, dtype=np.complex_)
        Rs  = self.Impur.R

        if rank   == 0:
            i_vec  = np.arange(0, self.Impur.n_imp)
            isp    = np.array_split(i_vec, nproc)
        else:
            i_vec = isp = None

        i_part = comm.scatter(isp, root=0)
        for i in i_part:
            mtx_i = 2*i
            for j in range(0, self.Impur.n_imp):
                mtx_j = 2*j
                if(i != j):
                    dR = Rs[i,:] - Rs[j,:]
                    mtx[mtx_i:mtx_i+2,mtx_j:mtx_j+2] = -self.Gc(dR)
        comm.Allreduce(mtx, self.Impur.T, op=MPI.SUM)



    def set_Gpart(self):
        E = self.E
        mtx = self.Impur.T
        Rs  = self.Impur.R
        for i in range(0, self.Impur.n_imp):
            mtx_i = 2*i
            for j in range(0, self.Impur.n_imp):
                mtx_j = 2*j
                if(i != j):
                    dR = Rs[i,:] - Rs[j,:]
                    mtx[mtx_i:mtx_i+2,mtx_j:mtx_j+2] = -self.Gc(dR)

# CODE THAT IS LIKE NOT TO BE USED AGAIN
    # def dRoh(self, R):
        # E = self.E
        # return -1.0/np.pi * np.imag( np.trace(self.dG(R,E)))

    # def dMs(self, R):
        # E = self.E
        # result = np.zeros(3, dtype=np.float_)
        # G_tmp = self.dG(R)

        # result[0] = np.imag(np.trace(np.dot(self.sigma_x, G_tmp)))
        # result[1] = np.imag(np.trace(np.dot(self.sigma_y, G_tmp)))
        # result[2] = np.imag(np.trace(np.dot(self.sigma_z, G_tmp)))

        # result *= -1.0 / np.pi
        # return result

    # def Npl(self, E):
        # R = np.array([self.R_to_0, self.R_to_0])
        # return - 1.0/np.pi * np.imag( np.trace(self.Gp(R, E)))

    # def Nmi(self, E):
        # R = np.array([self.R_to_0, self.R_to_0])
        # return - 1.0/np.pi * np.imag( np.trace(self.Gm(R, E)))

    # def Gp(self, R, E):
        # z = E + self.eta * 1j
        # k1, k2 = self.find_ks(z)
        # abs_R = la.norm(R)
        
        # fraction1, fraction2 = self.absk_Dpr(E)

        # return 0.5 * 1j * fraction1 * my_Hankel(0, k1, abs_R) \
                # * (z - (k1**2)/(2.0 * self.m)) * self.sigma_0
        
    # def Gm(self, R, E):
        # z = E + self.eta * 1j
        # k1, k2 = self.find_ks(z)
        # abs_R = la.norm(R)
        # fraction1, fraction2 = self.absk_Dpr(E)

        # return 0.5 * 1j * fraction2 * my_Hankel(0, k2, abs_R) \
                 # * (z - k2**2/(2.0 * self.m)) * self.sigma_0
    
    # def dG(self, r):
        # E = self.E
        # result = np.zeros((2,2), dtype=np.complex_)
        
        # for n in range(self.Impur.n_imp):
            # R       = self.Impur.R[n,:]
            # tmp     = np.dot(self.Impur.An(n), self.Gc(R-r))
            # result += np.dot(self.Gc(r-R), tmp)
        # return result
