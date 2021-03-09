import numpy as np
import matplotlib.pyplot as plt
from ODEmethods.methods import rk_methods, predictor, corrector
from ODEmethods.rungekutta import RKMethod 
from ODEmethods.pece import PECE

# SIR model: compartmental models for mathematical modelling of infectious diseases.
def SIRmodel(t, y, par=[0.01, 0.02]):
    return [-par[0]*y[0]*y[1], par[0]*y[0]*y[1] - par[1]*y[1], par[1]*y[1]]

par=[1., 0.15]

# RK methods
problem_rk = RKMethod(rk_methods["heun_euler"], SIRmodel, par)
rk = problem_rk.run(x0=0, y0=[0.99, 0.01, 0.], stepnum=1000, stepsize=0.025, adaptive=True, tolerance=10e-6)

# PECE methods
problem_pc = PECE(predictor[2], corrector[1], SIRmodel, par)
pc = problem_pc.run(x0=0, y0=[0.99, 0.01, 0.], stepnum=1000, stepsize=0.025, runcorr=1)

# Script for plotting graphs
fig = plt.figure()
plt.suptitle("SIR epidemiological model")
fig.set_size_inches(10, 5)

plt.subplot(1, 2, 1)
plt.errorbar(rk[0], rk[1][0], yerr=rk[2][0], color='lightpink', label='S - heun-euler')
plt.errorbar(rk[0], rk[1][1], yerr=rk[2][1], color='lightgreen', label='I - heun-euler')
plt.errorbar(rk[0], rk[1][2], yerr=rk[2][2], color='lightblue', label='R - heun-euler')

plt.plot(pc[0], pc[1][0], 'r--', label='S - p3c3_1')
plt.plot(pc[0], pc[1][1], 'g--', label='I - p3c3_1')
plt.plot(pc[0], pc[1][2], 'b--', label='R - p3c3_1')
plt.xlabel(r'$t$')
plt.ylabel(r'$N(t)$')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(rk[0], rk[2][0], color='lightpink', label='S - heun-euler')
plt.plot(rk[0], rk[2][1], color='lightgreen', label='I - heun-euler')
plt.plot(rk[0], rk[2][2], color='lightblue', label='R - heun-euler')
plt.xlabel(r'$t$')
plt.ylabel(r'$\delta N(t)$')
plt.legend()

plt.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.92, wspace=0.25, hspace=0.3)
plt.savefig("examples/SIRmodel/SIRmodel.png")
plt.show()