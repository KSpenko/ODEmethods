import numpy
from . import rungekutta

# -----------------------------------------------------------------------------
# Predictors: (Adams-Bashforth)
pred1 = [1]
pred2 = [3. / 2., -1. / 2.]
pred3 = [23. / 12., -16. / 12., 5. / 12.]
pred4 = [55. / 24., -59. / 24., 37. / 24., -9. / 24.]
pred5 = [1901. / 720., -2774. / 720., 2616. / 720., -1274. / 720., 251. / 720.]

#Correctors: (Adams-Moulton)
correc1 = [0.5, 0.5]
correc2 = [5. / 12., 2. / 3., -1. / 12.]
correc3 = [9. / 24., 19. / 24., -5. / 24., 1. / 24.]
correc4 = [251. / 720., 646. / 720., -264. / 720., 106. / 720., -19. / 720.]

#===============================================================================================================
# Class for PECE/PECECE/PECECECE
class PECE:
    """ Class for PECE techniques of handling numerical calculations of ordinary differential equations. """
    def __init__(self, function, parameters, predictor, corrector):
        """ PECE Method is specified by a combination of a predictor and a corrector, which are both defined by 1D arrays.
    The method is used to numericaly solve an ODE which is specified by a system of N ODE of first order!
    function - system of N ODE of 1. order,
    parameters - used and passed to the function,
    predictor/corrector - 1D arrays specifying the method, with the assumption that len(corrector) >= len(predictor)."""
        self.function = function
        self.parameters = parameters
        self.predictor = predictor
        self.corrector = corrector

    def predict(self, x, y, h):
        """ predicts the next step, based on the predictor method:
    x, y - past steps used for prediction,
    h - distance to the next step,
    returns: (x, y) of the next step"""
        vsota = numpy.zeros(y.shape[1])
        for i in range(len(self.predictor)):
            vsota += numpy.array(self.function(x[i], y[i], self.parameters)) * self.predictor[-i - 1]
        return x[-1]+h, y[-1]+h*vsota

    def correct(self, x, y, h):
        """ corrects the current step, based on the corrector method:
    x, y - past steps used for correction,
    h - distance to the next step,
    returns: corrected prediction of current step y"""
        vsota = numpy.zeros(y.shape[1])
        for i in range(len(self.corrector)):
            vsota += numpy.array(self.function(x[i], y[i], self.parameters)) * self.corrector[-i - 1]
        return y[-2] + h * vsota

    def run(self, x0=0, y0=0, stepnum=100, stepsize=1, runcorr=1):
        """ runs the PECE method for predicting next steps:
    x0, y0 - initial step,
    stepnum - number of steps to predict,
    stepsize - distance between steps,
    runcorr - number of how many times to correct each step prediction,
    returns: (x, y) arrays of steps and their values."""
        x = numpy.arange(stepnum + 1, dtype=float)
        y = numpy.empty([stepnum + 1, y0.shape[0]], dtype=float)
        pred_len = len(self.predictor)
        corr_len = len(self.corrector)

        # calculate first couple of elements (RK4)
        PECEstart = rungekutta.RKMethod(self.function, rungekutta.original_rk, self.parameters)
        x[:pred_len], y[:pred_len] = PECEstart.run(x0, y0, pred_len - 1, stepsize)[:-1]

        for i in range(pred_len, stepnum + 1):
            # predict values
            x[i], y[i] = self.predict(x[i - pred_len:i], y[i - pred_len:i], stepsize)
            for j in range(runcorr):  # correct estimate (multiple times)
                y[i] = self.correct(x[i + 1 - corr_len:i + 1], y[i + 1 - corr_len:i + 1], stepsize)
        return x, y