import numpy as np
from pylab import *
from schemes import diff, diff_up, diff2
from scipy.interpolate import UnivariateSpline
def get_var(q):
    nspecies = 3
    n = np.size(q)
    T = q[0:n:nspecies+1]
    Yf = q[1:n:nspecies+1]
    Yo = q[2:n:nspecies+1]
    Yp = q[3:n:nspecies+1]
    return T, Yf, Yo, Yp
    
def get_var_inv(T, Yf, Yo, Yp):
    nspecies = 3
    n = np.size(T)*(nspecies+1)
    q = np.zeros(n, dtype=np.complex)
    q[0:n:nspecies+1] = T[:]
    q[1:n:nspecies+1] = Yf[:]
    q[2:n:nspecies+1] = Yo[:]
    q[3:n:nspecies+1] = Yp[:]
    return q

def func(x, xt, left, right):
    n = len(x)
    var = np.zeros(n, dtype=np.complex)
    f = left + (1.0 + np.tanh((x - xt)*1000.0))/2.0*(right-left)
    return f

class FlameBase(object):
    def __init__(self):
        self.mdot = 0.2
        self.alpha = self.D = 1.5e-5
        self.n = 51
        self.x = np.linspace(0.0, 30.0e-3, self.n)
        self.nspecies = 3
        self.q = np.zeros((self.nspecies+1)*self.n, dtype=np.complex)
        
        self.dt = 1e-3
        self.maxiter = 100000
        self.tol = 1e-6

        T, Yf, Yo, Yp = get_var(self.q)
        
        T[:] = func(self.x, 0.008, 300.0, 900.0)
        Yf[:] = func(self.x, 0.008, 0.1, 0.0)
        Yo[:] = func(self.x, 0.008, 0.1, 0.0)
        Yp[:] = func(self.x, 0.008, 0.0, 0.2)

        if 0:
            d = np.loadtxt("base_solution/sol.dat")
            xb = d[:,0]
            Tb = d[:,1]
            Yfb = d[:,2]
            Yob = d[:,3]
            Ypb = d[:,4]

            kind = 0.001
            k = 1
            f = UnivariateSpline(xb, Tb, s=kind, k=k)
            T = f(self.x)
            f = UnivariateSpline(xb, Yfb, s=kind, k=k)
            Yf = f(self.x)
            f = UnivariateSpline(xb, Yob, s=kind, k=k)
            Yo = f(self.x)
            f = UnivariateSpline(xb, Ypb, s=kind, k=k)
            Yp = f(self.x)


        # T[:self.n/3] = 300.0
        # Yf[:self.n/3] = 0.1
        # Yo[:self.n/3] = 0.1
        # Yp[:self.n/3] = 0.0
        
        # T[self.n/3:] = 900.0
        # Yf[self.n/3:] = 0.0
        # Yo[self.n/3:] = 0.0
        # Yp[self.n/3:] = 0.2

        # T[self.n/3] = 900.0
        # Yf[self.n/3] = 0.05
        # Yo[self.n/3] = 0.05
        # Yp[self.n/3] = 0.2/2
        self.q = get_var_inv(T, Yf, Yo, Yp)
     #   figure()
        # print np.shape(self.x), np.shape(Yf)
     #   plot(self.x, Yf)
     #   plot(self.x, Yo)
     #   plot(self.x, Yp)
     #   figure()
     #   plot(self.x, T)
     #   show()

    def calc_source_terms(self, T, Yf, Yo, Yp):
        source_T = np.zeros(self.n, dtype=T.dtype)
        source_Yf = np.zeros(self.n, dtype=T.dtype)
        source_Yo = np.zeros(self.n, dtype=T.dtype)
        source_Yp = np.zeros(self.n, dtype=T.dtype)
        Ta = 14000.0
        A = 1e9
        Q = 1.5e6
        Cp = 1005.0


        nu_f = -1.0
        nu_o = -1.0
        nu_p = 2.0
        
        mw = 0.029

        Xf = Yf
        Xo = Yo
        kf = A*T*np.exp(-Ta/T)
        q = kf*Xf*Xo

        w_dot_f = nu_f*q
        w_dot_o = nu_o*q
        w_dot_p = nu_p*q

        w_dot_T = Q/Cp*q#(w_dot_p + w_dot_f + w_dot_o)
        if self.iterr > -1:
            self.dt = 1e-4
            source_T = w_dot_T
            source_Yf = w_dot_f*mw
            source_Yo = w_dot_o*mw
            source_Yp = w_dot_p*mw
            #ioff()
            #figure()
            #plot(source_T)
            #plot(source_Yo)
            #plot(source_Yp)
            #show()
        return source_T, source_Yf, source_Yo, source_Yp

    def calc_temperature_residual(self, T, source_T):
        x = self.x
        R = np.zeros(self.n, dtype=T.dtype)
        R = -diff_up(self.x, T)*self.mdot + self.alpha*diff2(self.x, T) + source_T
        R[0] = -(T[0] - 300.0)
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
        R = np.zeros(self.n*(self.nspecies+1), dtype=q.dtype)
        N = self.n*(self.nspecies+1)
        T, Yf, Yo, Yp = get_var(q)
        source_T, source_Yf, source_Yo, source_Yp = self.calc_source_terms(T, Yf, Yo, Yp)
        R_T = self.calc_temperature_residual(T, source_T)
        R_Yf = self.calc_species_residual(Yf, source_Yf, 0.1)
        R_Yo = self.calc_species_residual(Yo, source_Yo, 0.1)
        R_Yp = self.calc_species_residual(Yp, source_Yp, 0.0)
        nspecies = self.nspecies
        R[0:N:nspecies+1] = R_T[:]
        R[1:N:nspecies+1] = R_Yf[:]
        R[2:N:nspecies+1] = R_Yo[:]
        R[3:N:nspecies+1] = R_Yp[:]
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
        print np.linalg.cond(A)
        print norm(dRdq)
        plt.ion()
        plt.figure(3)
        plt.clf()
        plt.plot(real(np.linalg.eigvals(dRdq[0:n:4,0:n:4]))*dt[0:n:4], imag(np.linalg.eigvals(dRdq[0:n:4,0:n:4]))*dt[0:n:4], 'o')
        plt.plot(real(np.linalg.eigvals(dRdq[1:n:4,1:n:4]))*dt[1:n:4], imag(np.linalg.eigvals(dRdq[1:n:4,1:n:4]))*dt[1:n:4], 'o')
        plt.ylim(-1, 1)
#        plt.show()
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
            plt.plot(self.x, q[1:self.n*(self.nspecies+1):self.nspecies+1], 'rx')
            plt.plot(self.x, q[2:self.n*(self.nspecies+1):self.nspecies+1], 'g-')
            plt.plot(self.x, q[3:self.n*(self.nspecies+1):self.nspecies+1], 'b-')
            plt.pause(0.0000001)
            T, Yf, Yo, Yp = get_var(q.astype(np.float64))
            np.savetxt("solution/sol.dat", np.c_[self.x, T, Yf, Yo, Yp])

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
    flame = FlameBase()
    flame.solve()
