import numpy  as np

# Class for Simplectic integrator
class SymIntegrator:
    """ Class for Symplectic integrator techniques of handling numerical calculations of Hamiltonian systems."""
    def __init__(self, method, function, parameters):
        """ Symplectic integrator is specified by a 2D array of dimensions (2xn).
    The method is used to numericaly solve a Hamiltonian system which is specified by a system of 2 ODE of first order (generalized coordinates)!
    method - (2xn) array of coefficients c_i (first row for position) and coefficients d_i (second row for velocity/momentum),
    function - system of 2xn(degrees of freedom) ODE of first order (generalized coordinates, formated in pairs [q_1, p_1, q_2, p_2, ...]),
    parameters - used and passed to the function."""
        self.method = method
        self.function = function
        self.parameters = parameters

    def single_step(self, x, y, h):
        """ Calculates next step, estimated values of next step:
    x - previous step 
    y - generalized coordinates (formated in pairs [q_1, p_1, q_2, p_2, ...]),
    h - distance to the next step,
    returns: 
        x -> next step,
        y -> values at the next step."""
        for i in range(len(self.method[0])): #for every column (every pair of c_i, d_i)
            y[::2] = y[::2] + h*np.array(y[1::2])*self.method[0][i]
            y[1::2] = y[1::2] + h*np.array(self.function(x, y, self.parameters)[1::2])*self.method[1][i]
        return x+h, y

    def run(self, x0=0, y0=0, stepnum=100, stepsize=1):
        """ runs the simplectic integrator for predicting next steps:
    x0 - initial step 
    y0 - initial values, generalized coordinates (formated in pairs [q_1, p_1, q_2, p_2, ...]),
    stepnum - number of steps to predict,
    stepsize - distance between steps,
    runcorr - number of how many times to correct each step prediction,
    returns: 
        x -> array of steps,
        y -> multiple arrays of values in each step."""
        if len(y0) % 2 != 0:
            raise Exception("Generalized coordinates must be formated in pairs [q_1, p_1, q_2, p_2, ...]),")  
        
        # prepare for iterating steps
        # initial values
        x = np.arange((stepnum+1), dtype=float)
        x = x * stepsize + x0
        y = np.empty((len(y0), stepnum+1), dtype=float)
        y[:,0] = y0
        
        for i in range(1, stepnum + 1):
            x[i], y[:,i] = self.single_step(x[i-1], y[:,i-1], stepsize)
        return x, y