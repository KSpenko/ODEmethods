import numpy as np
import time
import matplotlib.pyplot as plt
from ODEmethods import rungekutta as rk
from ODEmethods import pece

def toplotna1(t, Temp, par=[0.1, -5.]):
    return (-par[0] * (Temp - par[1]))

def analytical_toplotna(t, T0, par=[0.1, -5]):
    return (par[1] + np.exp(-par[0] * t) * (T0 - par[1]))

par=[0.1,-5.]

test=np.zeros([4,2])

# RK methods
problem_rk = rk.RKMethod(toplotna1, rk.euler, par)
t0 = time.time()
rk1 = problem_rk.run(0, 21)
t1 = time.time()
test[0][0] = np.absolute(t1-t0)
error_rk1 = np.absolute(rk1[1]-analytical_toplotna(rk1[0], 21))
test[0][1] = np.amax(error_rk1)

problem_rk.method = rk.cash_karp
t0 = time.time()
rk2 = problem_rk.run(0, 21)
t1 = time.time()
test[1][0] = np.absolute(t1-t0)
error_rk2 = np.absolute(rk2[1]-analytical_toplotna(rk2[0], 21))
test[1][1] = np.amax(error_rk2)

# PECE methods
problem_pc = pece.PECE(toplotna1, par, pece.pred1, pece.correc1)
t0 = time.time()
pc1 = problem_pc.run(0, 21, runcorr=1)
t1 = time.time()
test[2][0] = np.absolute(t1-t0)
error_pc1 = np.absolute(pc1[1]-analytical_toplotna(pc1[0], 21))
test[2][1] = np.amax(error_pc1)

problem_pc.predictor = pece.pred4
problem_pc.corrector = pece.correc3
t0 = time.time()
pc2 = problem_pc.run(0, 21, runcorr=3)
t1 = time.time()
test[3][0] = np.absolute(t1-t0)
error_pc2 = np.absolute(pc2[1]-analytical_toplotna(pc2[0], 21))
test[3][1] = np.amax(error_pc2)

plt.figure()
plt.subplot(2, 1, 1)
plt.plot(rk1[0], rk1[1])
plt.plot(rk2[0], rk2[1])
plt.plot(pc1[0], pc1[1])
plt.plot(pc2[0], pc2[1])
plt.show()