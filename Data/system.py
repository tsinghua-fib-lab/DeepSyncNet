import numpy as np


class FHN():
    def __init__(self, a, b, I, epsilon, delta1, delta2, du, xdim):
        self.a = a
        self.b = b
        self.I = I
        self.epsilon = epsilon
        self.delta1 = delta1
        self.delta2 = delta2
        self.du = du
        self.xdim = xdim
    
    def f(self, x, t):
        u = x[:self.xdim]
        v = x[self.xdim:]

        if self.xdim == 1:
            dudt = (u - 1/3*u**3 - v + self.I) / self.epsilon
            dvdt = u - self.b*v + self.a
            return np.concatenate([dudt, dvdt])
        
        dudt = (u - 1/3*u**3 - v + self.I + self.du*(np.roll(u, -1) + np.roll(u, 1) - 2*u)) / self.epsilon
        dvdt = u - self.b*v + self.a

        if self.du!=0 and self.xdim>2:  # Neumann boundary condition
            dudt[0] = dudt[-1] = 0
            dvdt[0] = dvdt[-1] = 0
        
        return np.concatenate([dudt, dvdt])
    
    def g(self, x, t):

        delta1 = [np.sqrt(self.epsilon)/self.epsilon*self.delta1]*self.xdim
        delta2 = [self.delta2]*self.xdim

        if self.du!=0 and self.xdim>2:  # Neumann boundary condition
            delta1[0], delta1[-1] = 0, 0
            delta2[0], delta2[-1] = 0, 0

        return np.diag(delta1 + delta2)


class FHNv():
    def __init__(self, xdim, Du, Dv, epsilon, a0, a1, delta1, delta2):
        self.xdim = xdim
        self.Du = Du
        self.Dv = Dv
        self.epsilon = epsilon
        self.a0 = a0
        self.a1 = a1
        self.delta1 = delta1
        self.delta2 = delta2

    def f(self, x, t):
        u = x[:self.xdim]
        v = x[self.xdim:]

        dudt = u - 1/3*u**3 - v + self.Du*(np.roll(u, -1) + np.roll(u, 1) - 2*u)
        dvdt = self.epsilon*(u - self.a1*v - self.a0) + self.Dv*(np.roll(v, -1) + np.roll(v, 1) - 2*v)

        # Neumann boundary condition
        dudt[0] = dudt[-1] = 0
        dvdt[0] = dvdt[-1] = 0

        return np.concatenate([dudt, dvdt])
    
    def g(self, x, t):

        delta_u = [self.delta1*1.]*self.xdim
        delta_v = [np.sqrt(self.epsilon)*self.delta2*1.]*self.xdim

        # Neumann boundary condition
        delta_u[0], delta_u[-1] = 0, 0
        delta_v[0], delta_v[-1] = 0, 0

        return np.diag(delta_u+delta_v)
    
    
class HalfMoon_2D():
    def __init__(self, a1=1e-3, a2=1e-3, a3=1e-1, a4=1e-1):
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3
        self.a4 = a4
    
    def f(self, x, t):
        return np.array([self.a1 * 1, self.a3 * (1-x[1])])
    
    def g(self, x, t):
        return np.array([[self.a2*1., 0.],
                         [0., self.a4*1.]])

    
class Coupled_Lorenz():
    def __init__(self, sigma=10, rho=29, beta=8/3):
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        self.xi = 0.01
    
    def f(self, x, t):
        x1,x2, y1, y2, z1, z2 = x
        
        dx1dt = self.sigma*(y1-x1) - self.xi * (x1-x2)
        dy1dt = self.rho*x1 - y1 - x1*z1
        dz1dt = - self.beta*z1 + x1*y1
        dx2dt = self.sigma*(y2-x2) - self.xi * (x2-x1)
        dy2dt = self.rho*x2 - y2 - x2*z2
        dz2dt = - self.beta*z2 + x2*y2
        
        return np.array([dx1dt, 0.05*dx2dt, dy1dt, 0.05*dy2dt, dz1dt, 0.05*dz2dt])
    
    def g(self, x, t):
        return np.diag([0.0]*6)