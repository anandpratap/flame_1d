import numpy as np
from pylab import *
from schemes import diff, diff_up, diff2
from scipy.interpolate import UnivariateSpline
from chemistry import Chemistry

def func(x, xt, left, right):
    n = len(x)
    var = np.zeros(n, dtype=np.complex)
    f = left + (1.0 + np.tanh((x - xt)*1000.0))/2.0*(right-left)
    return f

class FlameBase(object):
    def __init__(self, chemistry, S_minf, S_pinf):
        self.c = chemistry
        self.S_minf = S_minf
        self.S_pinf = S_pinf

        self.mdot = 0.2
        self.alpha = self.D = 1.5e-5
        self.n = 21
        self.x = np.linspace(0.0, 30.0e-3, self.n)
        self.nspecies = self.c.n
        self.q = np.zeros((self.nspecies+1)*self.n, dtype=np.complex)
        
        self.dt = 1e-3
        self.maxiter = 100000
        self.tol = 1e-6

        T, Y = self.get_var(self.q)
        
        T[:] = func(self.x, 0.008, self.S_minf[0], self.S_pinf[0])
        for i in range(self.nspecies):
            Y[i,:] = func(self.x, 0.008, self.S_minf[i+1], self.S_pinf[i+1])
        self.q = self.get_var_inv(T, Y)

    def get_var(self, q):
        nspecies = self.nspecies
        n = self.n*(self.nspecies+1)
        T = q[0:n:nspecies+1]
        Y = np.zeros([nspecies, n/(nspecies+1)], dtype=np.complex)
        for i in range(nspecies):
            Y[i,:] = q[i+1:n:nspecies+1]
        return T, Y
    
    def get_var_inv(self, T, Y):
        nspecies = self.nspecies
        n = self.n*(self.nspecies+1)
        q = np.zeros(n, dtype=np.complex)
        q[0:n:nspecies+1] = T[:]
        for i in range(nspecies):
            q[i+1:n:nspecies+1] = Y[i,:]
        return q

    def massf_to_molef(self, Y):
        ybymw_sum = np.zeros(self.n, dtype=np.complex)
        for i in range(self.nspecies):
            ybymw_sum += Y[i,:]/self.c.mw[i]
        X = np.zeros_like(Y)
        for i in range(self.nspecies):
            X[i,:] = (Y[i,:]/self.c.mw[i])/ybymw_sum
        return X
        
    def calc_source_terms(self, T, Y):
        source_T = np.zeros(self.n, dtype=T.dtype)
        source_Y = np.zeros([self.nspecies, self.n], dtype=T.dtype)
        for idx, reaction in enumerate(self.c.reactions):
            Ta = reaction.Ta
            A = reaction.A
            b = reaction.b
            Q = reaction.Q
            Cp = 1005.0
            nu = reaction.nurhs - reaction.nulhs
            
            kf = A*T**b*np.exp(-Ta/T)

            mw = 0.029
            xprod = np.ones_like(kf)
            X = self.massf_to_molef(Y)
            for i in range(len(reaction.lhs_species)):
                p = reaction.nulhs[self.c.species.index(reaction.lhs_species[i])]
                xprod *= X[self.c.species.index(reaction.lhs_species[i]), :]**p
            q = kf*xprod
            w_dot = np.zeros([self.nspecies, self.n], dtype=np.complex)
            mw = self.c.mw
            for i in range(self.nspecies):
                w_dot[i,:] = nu[i]*q

            sum_w_dot = np.sum(w_dot, axis=0)
            
            w_dot_T = Q/Cp*q#(w_dot_p + w_dot_f + w_dot_o)
            #w_dot_T = Q/Cp*sum_w_dot
            source_T += w_dot_T
            for i in range(self.nspecies):
                source_Y[i,:] += w_dot[i,:]*mw[i]
        #ioff()
        #figure()
#        plot(source_T)
        #plot(source_Yo)
#        print nu
        #plot(source_Y.T)
        #show()
        
        return source_T, source_Y

    def calc_temperature_residual(self, T, source_T):
        x = self.x
        R = np.zeros(self.n, dtype=T.dtype)
        R = -diff_up(self.x, T)*self.mdot + self.alpha*diff2(self.x, T) + source_T
        R[0] = -(T[0] - self.S_minf[0])
        R[-1] = -1/(x[-1] - x[-2])*(T[-1] - T[-2])
        #R[-1] = -1/(x[-1] - x[-2])*(1.5*T[-1] - 2.0*T[-2] + 0.5*T[-3])
        return R
    
    def calc_species_residual(self, Y, source_Y, Yb):
        x = self.x
        R = np.zeros(self.n, dtype=Y.dtype)
        R = -diff_up(self.x, Y)*self.mdot + self.alpha*diff2(self.x, Y) + source_Y
        R[0] = -(Y[0] - Yb)
        R[-1] = -1/(x[-1] - x[-2])*(Y[-1] - Y[-2])
        #R[-1] = -1/(x[-1] - x[-2])*(1.5*Y[-1] - 2.0*Y[-2] + 0.5*Y[-3])
        return R

    def calc_residual(self, q):
        N = self.n*(self.nspecies+1)
        T, Y = self.get_var(q)
        source_T, source_Y = self.calc_source_terms(T, Y)
        R_T = self.calc_temperature_residual(T, source_T)
        R_Y = np.zeros([self.nspecies, self.n], dtype=q.dtype)
        for i in range(self.nspecies):
            R_Y[i,:] = self.calc_species_residual(Y[i,:], source_Y[i,:], self.S_minf[i+1])
        R = self.get_var_inv(R_T, R_Y)
        #figure()
        #plot(self.x, R_Yf)
        #plot(self.x, R_Yo)
        #plot(self.x, R_Yp)
        #figure()
        #plot(self.x, R_T)
        #show()
        return R

    def calc_residual_jacobian(self, q, dq=1e-25):
        n = np.size(q)
        print q.dtype
        print n
        dRdq = np.zeros([n, n], dtype=q.dtype)
        for i in range(n):
            q[i] = q[i] + 1j*dq
            R = self.calc_residual(q)
            dRdq[:,i] = np.imag(R[:])/dq
            q[i] = q[i] - 1j*dq
        return dRdq
        
    def step_implicit(self, q, dt):
        R = self.calc_residual(q)
        print np.shape(q)
        dRdq = self.calc_residual_jacobian(q)
        dt = self.calc_dt()
        A = np.zeros_like(dRdq)
        n = self.n*(self.nspecies+1)
        for i in range(0, n):
            A[i,i] = 1./dt[i]
        A = A - dRdq
        dq = linalg.solve(A, R)
        l2norm = np.sqrt(sum(R**2))/np.size(R)
        return dq, l2norm
    def step_explicit(self, q, dt):
        R = self.calc_residual(q)
        dt = self.calc_dt()
        dq = R*dt
        l2norm = np.sqrt(sum(R**2))/np.size(R)
        return dq, l2norm
    
    def step_rk4(self, q, dt):
        qi = q.copy()
        for i in range(4):
            R = self.calc_residual(q);
            q = q + R*dt/(5-i);
        dq = q - qi
        l2norm = np.sqrt(sum(R**2))/np.size(R)
        return dq, l2norm

    def calc_dt(self):
        dt = self.dt*np.ones(self.n*(self.nspecies+1))
        dt[0:self.n*(self.nspecies+1):self.nspecies+1] = self.dt/10
        return dt
    def solve(self):
        q = np.copy(self.q)
        dt = self.dt
        for i in range(self.maxiter):
            self.iterr = i
            dq, l2norm = self.step_implicit(q, dt)
            q[:] = q[:] + dq[:]
            q.clip(min=0)
            self.boundary(q)
            print "Iteration: %i Norm: %1.2e"%(i, l2norm)
            self.plot(q)
            if l2norm < self.tol:
                self.postprocess(q)
                break
        self.postprocess(q)
        self.q[:] = q[:]
    
    def plot(self, q):
        if self.iterr%10 == 0:
            plt.ion()
            plt.figure(1)
            clf()
            plt.plot(self.x, q[0:self.n*(self.nspecies+1):self.nspecies+1], 'r-')
            plt.figure(2)
            clf()
            for i in range(self.nspecies):
                plt.plot(self.x, q[i+1:self.n*(self.nspecies+1):self.nspecies+1], label=self.c.species[i])
            plt.legend()
            plt.pause(0.0000001)
            T, Y = self.get_var(q.astype(np.float64))
            np.savetxt("solution/sol.dat", np.c_[self.x, T, Y.T])

    def boundary(self, q):
        if self.iterr < -1 :
            n = self.n*(self.nspecies+1)
            q[-1] = q[-5]
            q[-2] = q[-6]
            q[-3] = q[-7]
            q[-4] = q[-8]
            print "boundary"
    def postprocess(self, q):
        pass
if __name__ == "__main__":
    #species = ['A', 'B', 'C', 'D']
    species = ['A2', 'B2', 'C2', 'D2']#, 'B2', 'C10A10']
    enthalpy = [0.0, 0.0, 1.5e6/2, 0.0]
    S_minf = [300.0, 0.1, 0.1, 0.1, 0.7]#, 0.3, 0.5]
    S_pinf = [900.0, 0.0, 0.0, 0.0, 1.0]#, 0.0, 0.0]
    c = Chemistry(species, enthalpy)

    # reaction = {'equation':'A = B', 'A': 1e9, 'b': 1.0, 'Ta':14000.0, 'Q': 1.5e6}
    # c.add_reaction(reaction)

    # reaction = {'equation':'D = C', 'A': 1e9, 'b': 1.0, 'Ta':14000.0, 'Q': 1.5e6}
    # c.add_reaction(reaction)
        
    # reaction = {'equation':'C = D', 'A': 1e9, 'b': 1.0, 'Ta':14000.0, 'Q': -1.5e6}
    # c.add_reaction(reaction)
        
    # reaction = {'equation':'B = A', 'A': 1e9, 'b': 1.0, 'Ta':14000.0, 'Q': -1.5e6}
    # c.add_reaction(reaction)

    #reaction = {'equation':'A2 = C2', 'A': 1e9, 'b': 1.0, 'Ta':14000.0}
    #c.add_reaction(reaction)
    reaction = {'equation':'C2 = A2', 'A': 1e9, 'b': 1.0, 'Ta':14000.0}
#    reaction = {'equation':'H2 + 0.5*O2 = H2O', 'A': 1e9, 'b': 1.0, 'Ta':14000.0, 'Q': -1.5e6}
    c.add_reaction(reaction)
    #reaction = {'equation':'2*C = A + B', 'A': 1e9, 'b': 1.0, 'Ta':14000.0, 'Q': -1.5e6}
    #c.add_reaction(reaction)
    
    flame = FlameBase(c, S_minf, S_pinf)
    flame.dt = 1e-3
    flame.solve()
