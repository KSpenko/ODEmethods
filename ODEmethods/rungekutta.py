import numpy as np

# Class for running RK methods with Butcher tableu:
class RKMethod:
    """ Class for Runge-Kutta techniques of handling numerical calculations of ordinary differential equations: initial value problems. """
    def __init__(self, method, function, parameters):
        """ RKMethod is specified by a Butcher tableau which are defined as a 2D array.
    The method is used to numericaly solve an ODE:IVP which is specified by a system of N ODE of first order!
    function - system of N ODE of 1. order, excepts (x, y, par) returns dy/dx -> len(y) = len(dy/dx) must match!
    parameters - used and passed to the function,
    method - 2D array of the Butcher tableau defined in dictionary "rk_methods" (so user can iterate through keys), 
            with the assumptions: 2D array is either a sqaure matrix (nxn), or has dimensions (nx(n+1)) for adaptive stepsize (n is the order of the RK methods)."""
        self.method = method
        self.function = function
        self.parameters = parameters

    def single_step(self, x, y, h, calc_error):
        """ Calculates next step, estimated values of next step (and conditionaly error):
    x, y - previous step (y can be multidimensional),
    h - distance to the next step,
    calc_error - switch for calculating error of step,
    returns: 
        x -> next step,
        y -> value/values at the next step,
        error -> error/errors of values at next step (if the method calculates it)."""
        k = np.zeros((len(y), self.method.shape[1]-1))
        for i in range(self.method.shape[1]-1):
            k[:,i] = self.function(x+self.method[i][0], y+h*k.dot(self.method[i,1:]), self.parameters)

        y_new1 = y + h*k.dot(self.method[-1,1:])
        if(calc_error):
            y_new2 = y + h*k.dot(self.method[-2,1:])
            error = np.absolute(y_new1 - y_new2)
            return x + h, y_new2, error
        else:
            return x + h, y_new1

    def run(self, x0=0., y0=0., stepnum=100, stepsize=1., adaptive=False, tolerance=1.0e-20):
        """ runs the RK method for predicting next steps:
    x0, y0 - initial step (y0 can be multidimensional),
    stepnum - number of steps to predict,
    stepsize - distance between steps,
    adaptive - switch for enabling adaptive stepsize (works only if it calculates error),
    tolerance - parameter for the condition of the adaptive stepsize (can cause instability),
    returns: 
        x -> array of steps,
        y -> array/multiple arrays of value/values in each step,
        error -> array/multiple arrays of error estimate/estimates of values (if the method calculates it)."""
        # generalizing input dimensions
        # we could avoid the use of "n" variable, if we would assume "y" would ALWAYS be an array (also input and output even if only one quantity n=1)!!
        try:
            n = len(y0)
        except:
            n = 1

        # check if method can estimate errors
        calc_error = False
        if(self.method.shape[0] == self.method.shape[1]+1):
            calc_error = True
        
        # prepare for iterating steps
        y = np.empty((n, stepnum+1), dtype=float)
        if(calc_error):
            error = np.zeros((n, stepnum+1), dtype=float)

        # initial values
        x = np.arange((stepnum+1), dtype=float)
        x = x * stepsize + x0 # adaptive step gets adjusted on top
        y[:,0] = y0

        # iteration LOOP
        for i in range(1, stepnum + 1):
            if(calc_error):
                if(adaptive):
                    # condition for adaptive stepsize
                    # https://en.wikipedia.org/wiki/Adaptive_step_size 
                    stepsize = stepsize * 0.9 * np.minimum(np.maximum(np.sqrt(0.5*np.amin(np.divide(tolerance,np.amax(error[:,i-1])))), 0.3), 2.)
                x[i], y[:,i], error[:,i] = self.single_step(x[i-1], y[:,i-1], stepsize, calc_error)
            else:
                x[i], y[:,i] = self.single_step(x[i-1], y[:,i-1], stepsize, calc_error)

        # reformat data for user friendly output
        if n == 1: 
            y = y[0]
            if(calc_error):
                error = error[0]
        if(calc_error):
            return x, y, error
        else:
            return x, y