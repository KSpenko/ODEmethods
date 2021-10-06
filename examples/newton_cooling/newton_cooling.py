import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
from ODEmethods.methods import rk_methods, predictor, corrector
from ODEmethods.rungekutta import RKMethod 
from ODEmethods.pece import PECE

# Newton's law of cooling
def heat(t, Temp, par=[0.1, -5.]):
    return (-par[0] * (Temp - par[1]))

def analytical_heat(t, T0, par=[0.1, -5]):
    return (par[1] + np.exp(-par[0] * np.array(t)) * (T0 - par[1]))

par=[0.1,-5.]

# RK methods
problem_rk = RKMethod(rk_methods["euler"], heat, par)
rk1 = problem_rk.run(x0=0, xf=100., y0=21, init_step=1.)

problem_rk.method = rk_methods["cash_karp"]
rk2 = problem_rk.run(x0=0, xf=100., y0=21, init_step=1., adaptive=True, tolerance=1e-6)

problem_rk.method = rk_methods["dormand_prince"]
rk3 = problem_rk.run(x0=0, xf=100., y0=21, init_step=1., adaptive=True, tolerance=1e-6)

# PECE methods
problem_pc = PECE(predictor[0], corrector[0], heat, par)
pc1 = problem_pc.run(x0=0, y0=21, stepnum=100, stepsize=1., runcorr=1)

problem_pc.predictor = predictor[3]
problem_pc.corrector = corrector[2]
pc2 = problem_pc.run(x0=0, y0=21, stepnum=100, stepsize=1., runcorr=3)

# Scipy methods
sct = np.arange(0, 100, 1)
sc1 = scipy.integrate.solve_ivp(heat, t_span=(0, 100), y0=[21], t_eval=sct)
sc2 = scipy.integrate.odeint(heat, t=sct, y0=[21], tfirst=True)
sc2 = np.transpose(sc2)[0]

sc3_solver = scipy.integrate.RK45(heat, 0, [21], t_bound=100., first_step=1.)
sc4_solver = scipy.integrate.DOP853(heat, 0, [21], t_bound=100., first_step=1.)
def scipy_run(sc_solver):
    sc = [[0.],[21.]]
    while True:
        try:
            sc_solver.step()
        except Exception as e:
            print(e)
            break
        sc[0].append(sc_solver.t)
        sc[1].append(sc_solver.y)
    return sc
sc3 = scipy_run(sc3_solver)
sc4 = scipy_run(sc4_solver)

# Script for plotting graphs
fig = plt.figure()
plt.suptitle("Newton's law of cooling")
fig.set_size_inches(12, 7)

plt.subplot(2, 2, 1)
plt.errorbar(rk2[0], rk2[1], yerr=rk2[2], label='cash-karp', color='red')
plt.errorbar(rk3[0], rk3[1], yerr=rk3[2], label='dormand-prince', color='darkturquoise')
plt.plot(rk1[0], rk1[1], label='euler', color='orange')
plt.plot(pc1[0], pc1[1], label='p1c2_1', color='lime')
plt.plot(pc2[0], pc2[1], label='p4c4_3', color='blue')
plt.plot(sc1['t'], sc1['y'][0], label='scipy.solve_ivp', color='purple')
plt.plot(sct, sc2, label='scipy.odeint', color='fuchsia')
plt.plot(sc3[0], sc3[1], label='scipy.RK45', color='brown')
plt.plot(sc4[0], sc4[1], label='scipy.DOP853', color='green')
plt.plot(rk2[0], analytical_heat(rk2[0], 21, par), label='analytical', color='black', linestyle='--')
plt.xlabel(r'$t$')
plt.ylabel(r'$T(t)$')
plt.legend(fontsize="xx-small")

plt.subplot(2, 2, 2)
plt.errorbar(rk2[0], rk2[1], yerr=rk2[2], label='cash-karp', color='red')
plt.errorbar(rk3[0], rk3[1], yerr=rk3[2], label='dormand-prince', color='darkturquoise')
plt.plot(rk1[0], rk1[1], label='euler', color='orange')
plt.plot(pc1[0], pc1[1], label='p1c2_1', color='lime')
plt.plot(pc2[0], pc2[1], label='p4c4_3', color='blue')
plt.plot(sc1['t'], sc1['y'][0], label='scipy.solve_ivp', color='purple')
plt.plot(sct, sc2, label='scipy.odeint', color='fuchsia')
plt.plot(sc3[0], sc3[1], label='scipy.RK45', color='brown')
plt.plot(sc4[0], sc4[1], label='scipy.DOP853', color='green')
plt.plot(rk2[0], analytical_heat(rk2[0], 21, par), label='analytical', color='black', linestyle='--')
plt.xlabel(r'$t$')
plt.ylabel(r'$T(t)$')
plt.xlim(-1., 50.)
plt.legend(fontsize="xx-small")

plt.subplot(2, 2, 3)
plt.semilogy(rk2[0], rk2[2], label='ck_estimate', color='salmon', linestyle='--')
plt.semilogy(rk3[0], rk3[2], label='dp_estimate', color='paleturquoise', linestyle='--')
plt.semilogy(rk2[0], np.absolute(rk2[1]-analytical_heat(rk2[0], 21, par)), label='cash-karp', color='red')
plt.semilogy(rk3[0], np.absolute(rk3[1]-analytical_heat(rk3[0], 21, par)), label='dormand-prince', color='darkturquoise')
plt.semilogy(rk1[0], np.absolute(rk1[1]-analytical_heat(rk1[0], 21, par)), label='euler', color='orange')
plt.semilogy(pc1[0], np.absolute(pc1[1]-analytical_heat(pc1[0], 21, par)), label='p1c2_1', color='lime')
plt.semilogy(pc2[0], np.absolute(pc2[1]-analytical_heat(pc2[0], 21, par)), label='p4c4_3', color='blue')
plt.semilogy(sc1['t'], np.absolute(sc1['y'][0]-analytical_heat(sc1['t'], 21, par)), label='scipy.solve_ivp', color='purple')
plt.semilogy(sct, np.absolute(sc2-analytical_heat(sct, 21, par)), label='scipy.odeint', color='fuchsia')
plt.semilogy(sc3[0], np.absolute(sc3[1]-analytical_heat(sc3[0], 21, par)), label='scipy.RK45', color='brown')
plt.semilogy(sc4[0], np.absolute(sc4[1]-analytical_heat(sc4[0], 21, par)), label='scipy.DOP853', color='green')
plt.xlabel(r'$t$')
plt.ylabel(r'$\Delta T(t)$')
plt.legend(fontsize="xx-small")

plt.subplot(2, 2, 4)
plt.loglog(rk2[0], rk2[2], label='ck_estimate', color='salmon', linestyle='--')
plt.loglog(rk3[0], rk3[2], label='dp_estimate', color='paleturquoise', linestyle='--')
plt.loglog(rk2[0], np.absolute(rk2[1]-analytical_heat(rk2[0], 21, par)), label='cash-karp', color='red')
plt.loglog(rk3[0], np.absolute(rk3[1]-analytical_heat(rk3[0], 21, par)), label='dormand-prince', color='darkturquoise')
plt.loglog(rk1[0], np.absolute(rk1[1]-analytical_heat(rk1[0], 21, par)), label='euler', color='orange')
plt.loglog(pc1[0], np.absolute(pc1[1]-analytical_heat(pc1[0], 21, par)), label='p1c2_1', color='lime')
plt.loglog(pc2[0], np.absolute(pc2[1]-analytical_heat(pc2[0], 21, par)), label='p4c4_3', color='blue')
plt.loglog(sc1['t'], np.absolute(sc1['y'][0]-analytical_heat(sc1['t'], 21, par)), label='scipy.solve_ivp', color='purple')
plt.loglog(sct, np.absolute(sc2-analytical_heat(sct, 21, par)), label='scipy.odeint', color='fuchsia')
plt.loglog(sc3[0], np.absolute(sc3[1]-analytical_heat(sc3[0], 21, par)), label='scipy.RK45', color='brown')
plt.loglog(sc4[0], np.absolute(sc4[1]-analytical_heat(sc4[0], 21, par)), label='scipy.DOP853', color='green')
plt.xlabel(r'$t$')
plt.ylabel(r'$\Delta T(t)$')
plt.legend(fontsize="xx-small")

plt.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.92, wspace=0.25, hspace=0.3)
plt.savefig("newton_cooling.png")
plt.show()