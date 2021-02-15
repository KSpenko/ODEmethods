import numpy 

# -----------------------------------------------------------------------------
# Coefficients
sym_crt =  +0.1259921049894873E+01

sym_eps1 = +0.1644986515575760E+00
sym_lam1 = -0.2094333910398989E-01
sym_ksi1 = +0.1235692651138917E+01

sym_eps2 = +0.1786178958448091E+00
sym_lam2 = -0.2123418310626054E+00
sym_ksi2 = -0.6626458266981849E-01

# Simplectic integrators
sym_euler = [[1],[1]]
sym_verlet = [[0, 1],[0.5, 0.5]]
sym_ruth = [[1, -2./3., 2./3.],[-1./24., 3./4., 7./24.]]
sym_forest_ruth = [[(1./(2.*(2. - sym_crt))), ((1. - sym_crt)/(2.*(2. - sym_crt))), ((1. - sym_crt)/(2.*(2. - sym_crt))), (1./(2.*(2. - sym_crt))) ],[(1./(2. - sym_crt)), (-sym_crt/(2. - sym_crt)), (1./(2. - sym_crt)), 0.]] #leapfrog_yoshida
sym_VEFRL = [[0., (1. - 2.*sym_lam1)/2., sym_lam1, sym_lam1, (1. - 2.*sym_lam1)/2.],[sym_eps1, sym_ksi1,(1. - 2.*(sym_ksi1 + sym_eps1)), sym_ksi1, sym_eps1]]
sym_PEFRL = [[sym_eps2, sym_ksi2, (1. - 2.*(sym_ksi2 + sym_eps2)), sym_ksi2, sym_eps2],[(1. -2.*sym_lam2)/2., sym_lam2, sym_lam2, (1. -2.*sym_lam2)/2., 0.]]

#===============================================================================================================
# Class for Simplectic integrator
class SymIntegrator:
    """ Class for Symplectic integrator techniques of handling numerical calculations of Hamiltonian systems."""
    def __init__(self, function, method, parameters):
        """ Symplectic integrator is specified by a 2D array of dimensions (2xn).
    The method is used to numericaly solve a Hamiltonian system which is specified by a system of 2 ODE of first order (generalized coordinates)!
    function - system of 2 ODE of first order (generalized coordinates),
    parameters - used and passed to the function,
    method - (2xn) array of coefficients c_i (first row for position) and coefficients d_i (second row for velocity/momentum)."""
        self.function = function
        self.parameters = parameters
        self.method = method

    def single_step(self, x, y, h):
        """ Calculates next step, estimated value of next step
    x, y - previous step,
    h - distance to the next step,
    returns: (x, y) next step and its value."""
        for i in range(len(self.method[0])): #for every column (every pair of c_i, d_i)
            y[0] = y[0] + h * y[1] * self.method[0][i]
            y[1] = y[1] + h * self.function(x, y, self.parameters)[-1] * self.method[1][i]
        return x + h, y

    def run(self, x0=0, y0=0, stepnum=100, stepsize=1):
        """ runs the simplectic integrator for predicting next steps:
    x0, y0 - initial step,
    stepnum - number of steps to predict,
    stepsize - distance between steps,
    returns: (x, y) arrays of steps and their values."""
        try:
            n = len(y0)
        except:
            n = 1
        x = numpy.arange(stepnum + 1, dtype=float)
        y = numpy.empty((stepnum + 1, n), dtype=float)

        x = x * stepsize + x0
        y[0] = y0

        for i in range(1, stepnum + 1):
            x[i], y[i] = self.single_step(x[i - 1], y[i - 1], stepsize)
        return x, y