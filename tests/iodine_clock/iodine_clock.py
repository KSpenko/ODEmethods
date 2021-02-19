import numpy as np
import matplotlib.pyplot as plt
from ODEmethods.methods import rk_methods
from ODEmethods.rungekutta import RKMethod 

# Chemical reactions: Iodine clock - Persulfate variation
def iodine_clock(t, y, par):  
    y0 = - par[0]*y[0]*y[1]
    y1 = - par[0]*y[0]*y[1] - par[1]*y[1]*y[2] + par[2]*y[3]*y[5] + par[3]*y[5]*y[6]
    y2 = par[0]*y[0]*y[1] - par[1]*y[1]*y[2]
    y3 = par[1]*y[1]*y[2] - par[2]*y[3]*y[5]
    y4 = 2*par[1]*y[1]*y[2]
    y5 = - par[2]*y[3]*y[5] - par[3]*y[5]*y[6]
    y6 = par[2]*y[3]*y[5] - par[3]*y[5]*y[6]
    y7 = par[3]*y[5]*y[6]
    return [y0, y1, y2, y3, y4, y5, y6, y7]

par = [1, 10, 100, 1000]

# RK methods
problem_rk = RKMethod(rk_methods["original_rk"], iodine_clock, par)
rk = problem_rk.run(x0=0, y0=[1., 0., 0.25, .09, 0., 1., 0., 0.], stepnum=10000, stepsize=0.001)

# Script for plotting graphs
fig = plt.figure()
fig.set_size_inches(13, 5)

plt.subplot(1, 2, 1)
plt.plot(rk[0], rk[1][0])
plt.plot(rk[0], rk[1][1])
plt.plot(rk[0], rk[1][2])
plt.plot(rk[0], rk[1][3])
plt.plot(rk[0], rk[1][4])
plt.plot(rk[0], rk[1][5])
plt.plot(rk[0], rk[1][6])
plt.plot(rk[0], rk[1][7])

plt.xlabel(r'$t$')
plt.ylabel(r'$N(t)$')
plt.legend(('$S_2 O^{2-}_8$', '$I^-$', '$IS_2 O^{2-}_8$', '$I_2$', '$SO^{2-}_4$', '$S_2 O^{2-}_3$', '$IS_2 O^{-}_3$', '$S_4 O^{2-}_6$'), loc='upper right')

plt.subplot(1, 2, 2)
plt.semilogx(rk[0], rk[1][0])
plt.semilogx(rk[0], rk[1][1])
plt.semilogx(rk[0], rk[1][2])
plt.semilogx(rk[0], rk[1][3])
plt.semilogx(rk[0], rk[1][4])
plt.semilogx(rk[0], rk[1][5])
plt.semilogx(rk[0], rk[1][6])
plt.semilogx(rk[0], rk[1][7])

plt.xlabel(r'$t$')
plt.ylabel(r'$N(t)$')
plt.suptitle("Iodine clock: Persulfate variation")
plt.legend(('$S_2 O^{2-}_8$', '$I^-$', '$IS_2 O^{2-}_8$', '$I_2$', '$SO^{2-}_4$', '$S_2 O^{2-}_3$', '$IS_2 O^{-}_3$', '$S_4 O^{2-}_6$'), loc='upper left')

plt.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.9, wspace=0.25, hspace=0.3)
plt.savefig("tests/iodine_clock/iodine_clock.png")
plt.show()