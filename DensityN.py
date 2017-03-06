import numpy as np


def theta(x):
    thres = 1e-8
    if(np.abs(x.imag) < thres and x.real >= 0):
        return 1.0
    else:
        return 0.0

class DOS:
    def __init__(self,m,alpha,beta,B0):
        self.m      = complex(m,     0)
        self.alpha  = complex(alpha, 0)
        self.beta   = complex(beta,  0)
        self.B0     = complex(B0,    0)
        self.alpBet = self.alpha**2 + self.beta**2
        self.E_so   = self.m * (self.alpha**2 + self.beta**2)


    #derivative of delta function for delta(E-Eplus(k)) and delta(E-Eminus(k))
    #evaluated with k+(E) and k-(E)
    def NPlusKp(self,E):
        return np.abs(	self.kplus(E)*(-1./self.m -  self.alpBet/np.sqrt(self.alpBet*self.kplus(E)**2+self.B0**2))	)

    def NMinusKp(self,E):
        return np.abs(	self.kplus(E)*(-1./self.m + self.alpBet/np.sqrt(self.alpBet*self.kplus(E)**2+self.B0**2))	)



    def NPlusKm(self,E):
        return np.abs(	self.kminus(E)*(-1./self.m - self.alpBet/np.sqrt(self.alpBet*self.kminus(E)**2+self.B0**2))	)

    def NMinusKm(self,E):
        return np.abs(	self.kminus(E)*(-1./self.m + self.alpBet/np.sqrt(self.alpBet*self.kminus(E)**2+self.B0**2))	)


    #k+(E) and k-(E)
    #sol of E-Eminus(k)=0
    def kminus(self,E):
        return np.sqrt(	2*self.m*(   E + self.E_so - np.sqrt( (E+self.E_so)**2+self.B0**2-E**2)  )			)

    #sol of E-Eplus(k)=0
    def kplus(self,E):
        return np.sqrt(	2*self.m*(   E + self.E_so + np.sqrt( (E+self.E_so)**2+self.B0**2-E**2)  )			)
    

    
    def N(self,E):
        res = 0
        if E<0:
            res += theta(self.kplus(E)) *self.kplus(E) /self.NMinusKp(E)
            res += theta(self.kminus(E))*self.kminus(E)/self.NMinusKm(E)
        else:
            res += theta(self.kplus(E)) *self.kplus(E) /self.NMinusKp(E)
            res += theta(self.kminus(E))*self.kminus(E)/self.NPlusKm(E)

        return res/(2*np.pi)


    #contribution of kplus part to N
    def Npl(self,E):
        res = 0
        if E<0:
            res += theta(self.kplus(E)) *self.kplus(E) /self.NMinusKp(E)
        else:
            res += theta(self.kplus(E)) *self.kplus(E) /self.NMinusKp(E)

        return res/(2*np.pi)

    #contribution of kminus part to N
    def Nmi(self,E):
        res = 0
        if E<0:
            res += theta(self.kminus(E))*self.kminus(E)/self.NMinusKm(E)
        else:
            res += theta(self.kminus(E))*self.kminus(E)/self.NPlusKm(E)

        return res/(2*np.pi)
