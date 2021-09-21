import numpy as np
from ODEmethods.methods import rk_methods
from ODEmethods.rungekutta import RKMethod

# Class for PECE/PECECE/PECECECE...
class PECE:
    """ Class for PECE techniques of handling numerical calculations of ordinary differential equations for initial value problems."""
    def __init__(self, predictor, corrector, function, parameters, rk_start=rk_methods["original_rk"]):
        """ PECE Method is specified by a combination of a predictor and a corrector, which are both defined by 1D arrays.
    The method is used to numericaly solve an ODE:IVP which is specified by a system of N ODE of first order!
    predictor/corrector - 1D arrays (defined in 2 arrays (index+1 marks the order) in ODEmethods.methods), 
                        with the assumption that len(corrector) == [len(predictor), len(predictor)+1],
    function - system of N ODE of 1. order,
    parameters - used and passed to the function."""
        self.predictor = predictor
        self.corrector = corrector
        self.function = function
        self.parameters = parameters
        self.rk_start = rk_start

    def predict(self, x, y, h):
        """ predicts the next step, based on the predictor method:
    x, y - previous steps (y can be multidimensional),
    h - distance to the next step,
    returns: 
        x -> next step,
        y -> predicted value/values at the next step."""
        return x[-1]+h, y[-1]+h*np.dot(self.function(x, np.transpose(y), self.parameters), np.flip(self.predictor))

    def correct(self, x, y, h):
        """ corrects the current step, based on the corrector method:
    x, y - previous steps (y can be multidimensional),
    h - distance to the next step,
    returns: 
        x -> next step,
        y -> corrected value/values at the next step."""
        return y[-2]+h*np.dot(self.function(x, np.transpose(y), self.parameters), np.flip(self.corrector))

    def run(self, x0=0, y0=0, stepnum=100, stepsize=1, runcorr=1):
        """ runs the PECE method for predicting next steps:
    x0, y0 - initial step (y0 can be multidimensional),
    stepnum - number of steps to predict,
    stepsize - distance between steps,
    runcorr - number of how many times to correct each step prediction,
    returns: 
        x -> array of steps,
        y -> array/multiple arrays of value/values in each step."""
        pred_len = len(self.predictor)
        corr_len = len(self.corrector)
        if (corr_len < pred_len) or (pred_len < corr_len-1):
            raise Exception("len(corrector) == [len(predictor), len(predictor)+1] must be satisfied!!")
        # generalizing input dimensions
        # we could avoid the use of "n" variable, if we would assume "y" would ALWAYS be an array (also input and output even if only one quantity n=1)!!
        try:
            n = len(y0)
        except:
            n = 1
        
        # prepare for iterating steps
        # initial values
        x = np.arange((stepnum+1), dtype=float)
        x = x * stepsize + x0
        y = np.empty((stepnum+1, n), dtype=float)
        y[0] = y0

        # calculate first couple of elements (RK4)
        PECEstart = RKMethod(self.rk_start, self.function,  self.parameters)
        x[:pred_len], y[:pred_len] = PECEstart.run(x0=x0, xf=stepsize*(pred_len-1), y0=y0, init_step=stepsize, adaptive=False)
        
        for i in range(pred_len, stepnum+1):
            # predict values
            x[i], y[i] = self.predict(x[i-pred_len:i], y[i-pred_len:i], stepsize)
            for j in range(runcorr):  # correct estimate (multiple times)
                y[i] = self.correct(x[i+1-corr_len:i+1], y[i+1-corr_len:i+1], stepsize)
        
         # reformat data for user friendly output
        if n == 1: 
            y = y[:,0]
        return x, y