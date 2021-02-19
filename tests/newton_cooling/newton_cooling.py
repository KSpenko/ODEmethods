import numpy as np
import matplotlib.pyplot as plt
from ODEmethods.methods import rk_methods, predictor, corrector
from ODEmethods.rungekutta import RKMethod 
from ODEmethods.pece import PECE

# Newton's law of cooling
def heat(t, Temp, par=[0.1, -5.]):
    return (-par[0] * (Temp - par[1]))

def analytical_heat(t, T0, par=[0.1, -5]):
    return (par[1] + np.exp(-par[0] * t) * (T0 - par[1]))

par=[0.1,-5.]

# RK methods
problem_rk = RKMethod(rk_methods["euler"], heat, par)
rk1 = problem_rk.run(x0=0, y0=21, stepnum=100)

problem_rk.method = rk_methods["cash_karp"]
rk2 = problem_rk.run(x0=0, y0=21, stepnum=100, adaptive=True, tolerance=10e-9)

# PECE methods
problem_pc = PECE(predictor[0], corrector[0], heat, par)
pc1 = problem_pc.run(x0=0, y0=21, stepnum=100, runcorr=1)

problem_pc.predictor = predictor[3]
problem_pc.corrector = corrector[2]
pc2 = problem_pc.run(x0=0, y0=21, stepnum=100, runcorr=3)

# Script for plotting graphs
fig = plt.figure()
fig.set_size_inches(12, 7)

plt.subplot(2, 2, 1)
plt.errorbar(rk2[0], rk2[1], yerr=rk2[2], label='cash-karp', color='red')
plt.plot(rk1[0], rk1[1], label='euler')
plt.plot(pc1[0], pc1[1], label='p1c2_1')
plt.plot(pc2[0], pc2[1], label='p4c4_3')
plt.plot(rk2[0], analytical_heat(rk2[0], 21, par), label='analytical', color='black', linestyle='--')
plt.xlabel(r'$t$')
plt.ylabel(r'$T(t)$')
plt.legend()

plt.subplot(2, 2, 2)
plt.errorbar(rk2[0], rk2[1], yerr=rk2[2], label='cash-karp', color='red')
plt.plot(rk1[0], rk1[1], label='euler')
plt.plot(pc1[0], pc1[1], label='p1c2_1')
plt.plot(pc2[0], pc2[1], label='p4c4_3')
plt.plot(rk2[0], analytical_heat(rk2[0], 21, par), label='analytical', color='black', linestyle='--')
plt.xlabel(r'$t$')
plt.ylabel(r'$T(t)$')
plt.xlim(-1., 50.)
plt.legend()

plt.subplot(2, 2, 3)
plt.errorbar(rk2[0], np.absolute(rk2[1]-analytical_heat(rk2[0], 21, par)), yerr=rk2[2], label='cash-karp', color='red')
plt.plot(rk1[0], np.absolute(rk1[1]-analytical_heat(rk1[0], 21, par)), label='euler')
plt.plot(pc1[0], np.absolute(pc1[1]-analytical_heat(pc1[0], 21, par)), label='p1c2_1')
plt.plot(pc2[0], np.absolute(pc2[1]-analytical_heat(pc2[0], 21, par)), label='p4c4_3')
plt.plot(rk2[0], rk2[2], label='ck_estimate', color='salmon', linestyle='--')
plt.xlabel(r'$t$')
plt.ylabel(r'$\delta T(t)$')
plt.legend()

plt.subplot(2, 2, 4)
plt.errorbar(rk2[0], np.absolute(rk2[1]-analytical_heat(rk2[0], 21, par)), yerr=rk2[2], label='cash-karp', color='red')
plt.plot(rk1[0], np.absolute(rk1[1]-analytical_heat(rk1[0], 21, par)), label='euler')
plt.plot(pc1[0], np.absolute(pc1[1]-analytical_heat(pc1[0], 21, par)), label='p1c2_1')
plt.plot(pc2[0], np.absolute(pc2[1]-analytical_heat(pc2[0], 21, par)), label='p4c4_3')
plt.plot(rk2[0], rk2[2], label='ck_estimate', color='salmon', linestyle='--')
plt.xlabel(r'$t$')
plt.ylabel(r'$\delta T(t)$')
plt.xlim(-1., 50.)
plt.legend()

plt.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.95, wspace=0.25, hspace=0.3)
plt.show()