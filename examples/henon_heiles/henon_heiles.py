import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from ODEmethods.methods import sym_methods
from ODEmethods.symplectic import SymIntegrator

# Henon-Heiles galactic potential:
def hehe(t, y, par=None):
    # y is of the form [x, vx, y, vy]
    v = y[1::2]
    ax1 = - y[0] - 2.*y[0]*y[2]
    ay1 = - y[2] - np.power(y[0],2.) + np.power(y[2],2.)
    return [v[0], ax1, v[1], ay1]

def ymax(y, e):
    return 2.*e - y**2. + (2./3.)*y**3.

def poincare(si):
    # poincare map for x=0
    si = np.array(si)
    y = []
    for i in range(len(si)):
        if (si[i,0] > 0. and si[i-1,0] <= 0.):
            dx = si[i,0] - si[i-1,0]
            y_poincare = si[i-1] + ((si[i] - si[i-1])/dx)*(0.-si[i-1,0])
            y.append(y_poincare)
    return np.array(y)

n = 20000
h = 0.2
eps = 1e-3
e_ar = [1./10., 1./8., (1./6.)-eps]

fig, axs = plt.subplots(1, len(e_ar))
plt.suptitle("Hénon-Heiles: Poincare section")
fig.set_size_inches(16, 5)

for i in range(len(e_ar)):
    #print(i)
    e = e_ar[i]
    y_init = np.linspace(0., fsolve(ymax, 0.5, e), 20)
    v_init = np.sqrt(2.*e - np.power(y_init, 2.) + (2./3.)*np.power(y_init, 3.))
    for y0, v0 in zip(y_init, v_init):
        problem_si = SymIntegrator(sym_methods["VEFRL"], hehe, parameters=None)
        si = problem_si.run(x0=0, y0=[0., v0, y0, 0.], stepnum=n, stepsize=h)
        pmap = poincare(si[1])             # linear interpolation
        axs[i].scatter(pmap[:,2], pmap[:,3], marker='.')
    axs[i].set_title(r'$E=$'+str(e))
    axs[i].set_xlabel(r'$y$')
    axs[i].set_ylabel(r'$dy/dt$')
    axs[i].set_aspect(1)

plt.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.9, wspace=0.25, hspace=0.3)
plt.savefig("henon_heiles.png")
plt.show()