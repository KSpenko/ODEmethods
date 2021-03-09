import numpy as np
import matplotlib.pyplot as plt
from ODEmethods.methods import rk_methods, predictor, corrector, sym_methods
from ODEmethods.rungekutta import RKMethod 
from ODEmethods.pece import PECE
from ODEmethods.symplectic import SymIntegrator

# Mathematical pendulum
def matpend(t, u, w=1.):
    return [u[1], -w**2. * np.sin(u[0])]

def kinetic_matpend(p, w=1.):
    return 0.5*((p/w)**2.)

def potential_matpend(q):
    return (1.-np.cos(q))

par=1.
n=10000
h=0.1

fig, axs = plt.subplots(3, 1)
plt.suptitle("Mathematical pendulum")
fig.set_size_inches(8, 8)

for y0 in [[-3.*np.pi, 1.],[-3.*np.pi, 2.5],[3.*np.pi, -1.],[3.*np.pi, -2.5],[-2.*np.pi, 1.9],[-2.*np.pi, 1.7],[-2.*np.pi, 1.],[-2.*np.pi, 0.5],[2.*np.pi, 1.9],[2.*np.pi, 1.7],[2.*np.pi, 1.],[2.*np.pi, 0.5],[0., 1.9],[0., 1.7],[0., 1.],[0., 0.5]]:
    # RK methods
    problem_rk = RKMethod(rk_methods["euler"], matpend, par)
    rk = problem_rk.run(x0=0, y0=y0, stepnum=n, stepsize=h)

    # PECE methods
    problem_pc = PECE(predictor[0], corrector[0], matpend, par)
    pc = problem_pc.run(x0=0, y0=y0, stepnum=n, stepsize=h, runcorr=1)

    # SymInt methods
    problem_si = SymIntegrator(sym_methods["PEFRL"], matpend, par)
    si = problem_si.run(x0=0, y0=y0, stepnum=n, stepsize=h)

    # Plotting
    axs[0].plot(rk[1][0], potential_matpend(rk[1][0]), color='green')
    axs[0].plot(pc[1][0], potential_matpend(pc[1][0]), color='red')
    axs[0].plot(si[1][0], potential_matpend(si[1][0]), color='blue')

    axs[1].plot(rk[1][0], rk[1][1], color='green')
    axs[1].plot(pc[1][0], pc[1][1], color='red')
    axs[1].plot(si[1][0], si[1][1], color='blue')

    axs[2].plot(rk[0], potential_matpend(rk[1][0])+kinetic_matpend(rk[1][1]), color='green')
    axs[2].plot(pc[0], potential_matpend(pc[1][0])+kinetic_matpend(pc[1][1]), color='red')
    axs[2].plot(si[0], potential_matpend(si[1][0])+kinetic_matpend(si[1][1]), color='blue')
    
axs[0].set_title("Potential energy")
axs[0].set_xlim([-3.*np.pi, 3.*np.pi])
axs[0].set_xlabel(r'$\Theta$')
axs[0].set_ylabel(r'$W_p(\Theta )$')
axs[0].legend(["euler_rk","p1c2_1", "PEFRL"])

axs[1].set_title("Phase portrait")
Y1, Y2 = np.meshgrid(np.linspace(-3.*np.pi, 3.*np.pi, 40), np.linspace(-3.5, 3.5, 40))
u, v = np.zeros(Y1.shape), np.zeros(Y2.shape)
for i in range(Y1.shape[0]):
    for j in range(Y1.shape[1]):
        u[i,j], v[i,j] = matpend(0., [Y1[i,j], Y2[i,j]], par)
axs[1].quiver(Y1, Y2, u, v, color='lightgray')

axs[1].set_xlim([-3.*np.pi, 3.*np.pi])
axs[1].set_ylim([-3.5, 3.5])
axs[1].set_xlabel(r'$\Theta$')
axs[1].set_ylabel(r'$d\Theta /dt$')
axs[1].legend(["euler_rk","p1c2_1", "PEFRL"])

axs[2].set_title("Conservation of energy")
axs[2].set_xlabel(r'$t$')
axs[2].set_ylabel(r'$W_p(t)+W_k(t)$')
axs[2].legend(["euler_rk","p1c2_1", "PEFRL"])

plt.subplots_adjust(left=0.1, right=0.95, bottom=0.07, top=0.92, wspace=0.25, hspace=0.4)
plt.savefig("examples/math_pendulum/math_pendulum.png")
plt.show()