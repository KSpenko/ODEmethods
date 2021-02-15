import numpy

#-------------------------------------------------------------------------------
# Runge-Kutta Coefficients
a1 = 2.000000000000000e-01  # 1/5
a2 = 2.500000000000000e-01  # 1/4
a3 = 3.750000000000000e-01  # 3/8
a4 = 9.230769230769231e-01  # 12/13
a5 = 1.000000000000000e+00  # 1
a6 = 5.000000000000000e-01  # 1/2
a7 = 7.500000000000000e-01  # 3/4
a8 = 1.250000000000000e-01  # 1/8
a9 = 3.333333333333333e-01  # 1/3
a10 = 6.666666666666667e-01  # 2/3
a11 = 2.000000000000000e+00  # 2
a12 = 1.666666666666667e-01  # 1/6
a13 = 2.222222222222222e-01  # 2/9
a14 = 4.444444444444444e-01  # 4/9
a15 = 1.414213562373095e+00  # sqrt(2)
a16 = 3.906250000000000e-03  # 1/256
a17 = 9.960937500000000e-01  # 255/256
a18 = 1.953125000000000e-03  # 1/512
a19 = 2.916666666666667e-01  # 7/24
a20 = 2.500000000000000e-02  # 1/40

b21 = 2.500000000000000e-01  # 1/4
b31 = 9.375000000000000e-02  # 3/32
b32 = 2.812500000000000e-01  # 9/32
b41 = 8.793809740555303e-01  # 1932/2197
b42 = -3.277196176604461e+00  # -7200/2197
b43 = 3.320892125625853e+00  # 7296/2197
b51 = 2.032407407407407e+00  # 439/216
b52 = -8.000000000000000e+00  # -8
b53 = 7.173489278752436e+00  # 3680/513
b54 = -2.058966861598441e-01  # -845/4104
b61 = -2.962962962962963e-01  # -8/27
b62 = 2.000000000000000e+00  # 2
b63 = -1.381676413255361e+00  # -3544/2565
b64 = 4.529727095516569e-01  # 1859/4104
b65 = -2.750000000000000e-01  # -11/40

r1 = 2.777777777777778e-03  # 1/360
r3 = -2.994152046783626e-02  # -128/4275
r4 = -2.919989367357789e-02  # -2197/75240
r5 = 2.000000000000000e-02  # 1/50
r6 = 3.636363636363636e-02  # 2/55

p1 = 1.185185185185185e-01  # 16/135
p3 = 5.189863547758285e-01  # 6656/12825
p4 = 5.061314903420167e-01  # 28561/56430
p5 = -1.800000000000000e-01  # -9/50

c1 = 1.157407407407407e-01  # 25/216
c3 = 5.489278752436647e-01  # 1408/2565
c4 = 5.353313840155945e-01  # 2197/4104
c5 = -2.000000000000000e-01  # -1/5

ck1 = 3.000000000000000e-01  # 3/10
ck2 = 7.500000000000000e-02  # 3/40
ck3 = 2.250000000000000e-01  # 9/40
ck4 = 6.000000000000000e-01  # 3/5
ck5 = -9.000000000000000e-01  # -9/10
ck6 = 1.200000000000000e+00  # 6/5
ck7 = -2.037037037037037e-01  # -11/54
ck8 = 2.500000000000000e+00  # 5/2
ck9 = -2.592592592592592e+00  # -70/27
ck10 = 1.296296296296296e+00  # 35/27
ck11 = 8.750000000000000e-01  # 7/8
ck12 = 2.949580439814815e-02  # 1631/55296
ck13 = 3.417968750000000e-01  # 175/512
ck14 = 4.159432870370370e-02  # 575/13824
ck15 = 4.003454137731482e-01  # 44275/110592
ck16 = 6.176757812500000e-02  # 253/4096
ck17 = 9.788359788359788e-02  # 37/378
ck18 = 4.025764895330113e-01  # 250/621
ck19 = 2.104377104377104e-01  # 125/594
ck20 = 2.891022021456804e-01  # 512/1771
ck21 = 1.021773726851852e-01  # 2825/27648
ck22 = 3.839079034391534e-01  # 18575/48384
ck23 = 2.445927372685185e-01  # 13525/55296
ck24 = 1.932198660714286e-02  # 277/14336

dp1 = 8.000000000000000e-01  # 4/5
dp2 = 9.777777777777778e-01  # 44/45
dp3 = -3.733333333333333e+00  # -56/15
dp4 = 3.555555555555556e+00  # 32/9
dp5 = 8.888888888888889e-01  # 8/9
dp6 = 2.952598689224204e+00  # 19372/6561
dp7 = -1.159579332418839e+01  # -25360/2187
dp8 = 9.822892851699436e+00  # 64448/6561
dp9 = -2.908093278000000e-01  # -212/729
dp10 = 2.846275252525253e+00  # 9017/3168
dp11 = -1.075757575757576e+01  # -355/33
dp12 = 8.906422717743473e+00  # 46732/5247
dp13 = 2.784090909090909e-01  # 49/176
dp14 = -2.735313036020583e-01  # -5103/18656
dp15 = 9.114583333333333e-02  # 35/384
dp16 = 4.492362982929021e-01  # 500/1113
dp17 = 6.510416666666667e-01  # 125/192
dp18 = -3.223761792452830e-01  # -2187/6784
dp19 = 1.309523809523810e-01  # 11/84
dp20 = 8.991319444444444e-02  # 5179/57600
dp21 = 4.534890685834082e-01  # 7571/16695
dp22 = 6.140625000000000e-01  # 393/640
dp23 = -2.715123820754717e-01  # -92097/339200
dp24 = 8.904761904761905e-02  # 187/2100

#-------------------------------------------------------------------------------
# eksplicit RK methods
euler = [[0, 0], [None, a5]]
midpoint = [[0, 0, 0], [a6, a6, 0], [None, 0, a5]]
heun = [[0, 0, 0], [a5, a5, 0], [None, a6, a6]]
ralston = [[0, 0, 0], [a10, a10, 0], [None, a2, a7]]
kutta_second = [[0, 0, 0], [a6, a6, 0], [None, 0, a5]]
kutta_third = [[0, 0, 0, 0], [a6, a6, 0, 0], [a5, -a5, a11, 0], [None, a12, a10, a12]]
heun_third = [[0, 0, 0, 0], [a9, a9, 0, 0], [a10, 0, a10, 0], [None, a2, 0, a7]]
ralston_third = [[0, 0, 0, 0], [a6, a6, 0, 0], [a7, 0, a7, 0], [None, a13, a9, a14]]
ssprk3 = [[0, 0, 0, 0], [a5, a5, 0, 0], [a6, a2, a2, 0], [None, a12, a12, a10]]
original_rk = [[0, 0, 0, 0, 0], [a6, a6, 0, 0, 0], [a6, 0, a6, 0, 0], [a5, 0, 0, a5, 0], [None, a12, a9, a9, a12]]
third_eight = [[0, 0, 0, 0, 0], [a9, a9, 0, 0, 0], [a10, -a9, a5, 0, 0], [a5, a5, -a5, a5, 0], [None, a8, a3, a3, a8]]
gill = [[0, 0, 0, 0, 0], [a6, a6, 0, 0, 0], [a6, a6 * (-a5 + a15), a5 - a6 * a15, 0, 0],
        [a5, 0, -a6 * a15, a5 + a6 * a15, 0], [None, a12, (a11 - a15) * a12, (a11 + a15) * a12, a12]]

# Adaptive step RK methods
heun_euler = [[0, 0, 0], [a5, a5, 0], [None, a6, a6], [None, a5, 0]]
fehlberg_rk12 = [[0, 0, 0, 0], [a6, a6, 0, 0], [a5, a16, a17, 0], [None, a16, a17, 0], [None, a18, a17, a18]]
bogacki_shampine = [[0, 0, 0, 0, 0], [a6, a6, 0, 0, 0], [a7, 0, a7, 0, 0], [a5, a13, a9, a14, 0],
                    [None, a13, a9, a14, 0], [None, a19, a2, a9, a8]]
fehlberg = [[0, 0, 0, 0, 0, 0, 0], [a2, a2, 0, 0, 0, 0, 0], [a3, b31, b32, 0, 0, 0, 0], [a4, b41, b42, b43, 0, 0, 0],
            [a5, b51, b52, b53, b54, 0, 0], [a6, b61, b62, b63, b64, b65, 0], [None, p1, 0, p3, p4, p5, r6],
            [None, c1, 0, c3, c4, c5, 0]]
cash_karp = [[0, 0, 0, 0, 0, 0, 0], [a1, a1, 0, 0, 0, 0, 0], [ck1, ck2, ck3, 0, 0, 0, 0], [ck4, ck1, ck5, ck6, 0, 0, 0],
             [a5, ck7, ck8, ck9, ck10, 0, 0], [ck11, ck12, ck13, ck14, ck15, ck16, 0],
             [None, ck17, 0, ck18, ck19, 0, ck20], [None, ck21, 0, ck22, ck23, ck24, a2]]
dormand_prince = [[0, 0, 0, 0, 0, 0, 0, 0], [a1, a1, 0, 0, 0, 0, 0, 0], [ck1, ck2, ck3, 0, 0, 0, 0, 0],
                  [dp1, dp2, dp3, dp4, 0, 0, 0, 0], [dp5, dp6, dp7, dp8, dp9, 0, 0, 0],
                  [a5, dp10, dp11, dp12, dp13, dp14, 0, 0], [a5, dp15, 0, dp16, dp17, dp18, dp19, 0],
                  [None, dp15, 0, dp16, dp17, dp18, dp19, 0], [None, dp20, 0, dp21, dp22, dp23, dp24, a20]]

#===================================================================================================================
# Class for running RK methods with Butcher tableu:
class RKMethod:
    """ Class for Runge-Kutta techniques of handling numerical calculations of ordinary differential equations. """
    def __init__(self, function, method, parameters):
        """ RKMethod is specified by a Butcher tableau which are defined as a 2D array.
    The method is used to numericaly solve an ODE which is specified by a system of N ODE of first order!
    function - system of N ODE of 1. order,
    parameters - used and passed to the function,
    method - 2D array of the Butcher tableau, with the assumptions: 2D array is either a sqaure matrix (nxn), or has dimensions (nx(n+1)) for adaptive stepsize (n is the order of the RK methods)."""
        self.function = function
        self.parameters = parameters
        self.method = method

    def single_step(self, x, y, h):
        """ Calculates next step, estimated values of next step (and conditionaly error):
    x, y - previous step,
    h - distance to the next step,
    returns: (x, y, error) next step, its value, and its error estimate (if the method calculates it)."""
        error = 0  # set to zero if method doesnt estimate it itself
        k = numpy.zeros([len(self.method), y.shape[0]])  # Rows of Butcher tableu
        for i in range(len(self.method)):  # for every row in Butcher tableu
            yadd = 0.
            for j in range(1, len(self.method[0])):  # for every column in BT (except first one)
                yadd = yadd + self.method[i][j] * k[j - 1]

            if i < len(self.method[0]) - 1:  # calculate coefficients
                k[i] = (self.function(x + h * (self.method[i][0]), y + h * yadd, self.parameters))
            else:  # calculate estimate
                k[i] = (y + h * yadd)

        if len(self.method) == 1 + len(self.method[0]):  # (optional) calculate error of estimate
            error = numpy.absolute(k[-1] - k[-2])
        return x + h, k[len(self.method) - 1], error

    def run(self, x0=0, y0=0, stepnum=100, stepsize=1, adaptive=False, tolerance=1.0e-20):
        """ runs the RK method for predicting next steps:
    x0, y0 - initial step,
    stepnum - number of steps to predict,
    stepsize - distance between steps,
    adaptive - switch for enabling adaptive stepsize
    tolerance - parameter for the condition of the adaptive stepsize
    returns: (x, y, error) arrays of steps and their values and error estimate of values (if the method calculates it)."""
        x = numpy.arange(stepnum + 1, dtype=float)
        y = numpy.empty([stepnum + 1, numpy.array(y0).shape[0]], dtype=float)
        error = numpy.empty([stepnum + 1, numpy.array(y0).shape[0]], dtype=float)

        x = x * stepsize + x0
        y[0] = y0
        error[0] = 0

        for i in range(1, stepnum + 1):
            # condition for adaptive stepsize
            # https://en.wikipedia.org/wiki/Adaptive_step_size
            if(adaptive): stepsize = stepsize * 0.9 * numpy.minimum(numpy.maximum(numpy.sqrt(0.5*numpy.amin(numpy.divide(tolerance,error[i-1]))), 0.3), 2.)
            # single_step
            x[i], y[i], error[i] = self.single_step(x[i - 1], y[i - 1], stepsize)
        return x, y, error