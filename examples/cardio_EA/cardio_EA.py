import copy
import numpy as np
from scipy.integrate import odeint, DOP853, RK45
import matplotlib.pyplot as plt

from ODEmethods.methods import rk_methods, sym_methods
from ODEmethods.rungekutta import RKMethod 
from ODEmethods.symplectic import SymIntegrator

def cardio(t, y, par):
    # Siroos Nazari
    d1, d2, d3 = par[0] 
    c1, c2, c3 = par[1]
    r35, r13, r15 = par[2]
    r31, r51, r53 = par[3]
    a1, a2, a3, w = par[4]
    # SA
    dy1 = y[1]
    dy2 = -d1*(y[0]**2.-1.)*y[1] - c1*y[0] + a1*np.cos(w*t) + r13*(y[0]-y[2]) + r15*(y[0]-y[4])
    # AV
    dy3 = y[3]
    dy4 = -d2*(y[2]**2.-1.)*y[3] - c2*y[2] + a2*np.cos(w*t) + r31*(y[2]-y[0]) + r35*(y[2]-y[4])
    # HP
    dy5 = y[5]
    dy6 = -d3*(y[4]**2.-1.)*y[5] - c3*y[4] + a3*np.cos(w*t) + r51*(y[4]-y[0]) + r53*(y[4]-y[2])
    return [dy1, dy2, dy3, dy4, dy5, dy6]

par = [
    [5., 6., 7.],
    [1.7, 1.7, 1.7],
    [1., 1., 1.],
    [0.0001, 0.0002, 0.0003],
    [5., 6., 4., 2.002],
]
y0 = np.ones(6)*0.01

sct = np.linspace(0., 20., 1000)
sc = odeint(cardio, t=sct, y0=y0, args=(par,), tfirst=True)

fig, axs = plt.subplots(1, 2)
plt.suptitle("Cardiac Electrical Activity")
fig.set_size_inches(10, 6)

axs[0].plot(sct, sc[:,0])
axs[0].plot(sct, sc[:,2])
axs[0].plot(sct, sc[:,4])

axs[1].plot(sct, sc[:,1])
axs[1].plot(sct, sc[:,3])
axs[1].plot(sct, sc[:,5])

plt.savefig("cardio_EA.png")
plt.close()