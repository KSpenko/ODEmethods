import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from ODEmethods.methods import sym_methods, rk_methods, predictor, corrector
from ODEmethods.symplectic import SymIntegrator
from ODEmethods.rungekutta import RKMethod
from ODEmethods.pece import PECE

# Two-body 2D (planar) problem:
def two_body(t, y, par=[1., 0.1, 2.]):
    # par = [G, m1, m2]
    # y is of the form [x1, vx1, y1, vy1, x2, vx2, y2, vy2]
    r = np.sqrt(np.power(y[0]-y[4], 2)+np.power(y[2]-y[6], 2))
    fg = -par[0]*np.divide(1., np.power(r, 2))
    
    v = y[1::2]
    ax1 = par[2]*fg*np.divide((y[0]-y[4]), r)
    ay1 = par[2]*fg*np.divide((y[2]-y[6]), r)
    ax2 = par[1]*fg*np.divide((-y[0]+y[4]), r)
    ay2 = par[1]*fg*np.divide((-y[2]+y[6]), r)
    return [v[0], ax1, v[1], ay1, v[2], ax2, v[3], ay2]

par=[1, 0.1, 2]
n=1000
h=0.2
y0=[5.,-0.3,-5.,0.,1.,-0.0001,1.,0.]

# SymIntegrator methods
problem_si = SymIntegrator(sym_methods["PEFRL"], two_body, par)
si = problem_si.run(x0=0, y0=y0, stepnum=n, stepsize=h)

# Runge-Kutta methods
problem_rk = RKMethod(rk_methods["original_rk"], two_body, par)
rk = problem_rk.run(x0=0, y0=y0, stepnum=n, stepsize=h)

# PECE methods
problem_pc = PECE(predictor[3], corrector[3], two_body, par)
pc = problem_pc.run(x0=0, y0=y0, stepnum=n, stepsize=h, runcorr=3)

# Animation Script:--------------------------------------------------------------------
def update_lines(num, dataLines, lines):
    for line, data in zip(lines, dataLines):
        line.set_data(data[..., :num])
    return lines

fig, ax = plt.subplots()

data = np.empty((6, 2, n+1))
data[0] = rk[1][:4:2]
data[1] = rk[1][4::2]
data[2] = pc[1][:4:2]
data[3] = pc[1][4::2]
data[4] = si[1][:4:2]
data[5] = si[1][4::2]

lines = []
colors = ["green","lightgreen","red","pink","blue","lightblue"]
for i in range(len(data)):
    lines.append(ax.plot(data[i][0, 0:1], data[i][1, 0:1])[0])
    lines[-1].set_color(colors[i])

ax.set_xlim(-10,10)
ax.set_ylim(-10,10)
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$')
ax.legend((lines[0],lines[2],lines[4]),("original_rk","p4c5_3", "PEFRL"))
plt.suptitle("Two-body problem")

line_ani = animation.FuncAnimation(fig, update_lines, n+1, fargs=(data, lines), interval=1, blit=False, save_count=n+1)
writergif = animation.PillowWriter(fps=30)
line_ani.save("tests/two_body/two_body.gif", writer=writergif)
plt.show()