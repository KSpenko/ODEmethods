import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
from ODEmethods.methods import sym_methods
from ODEmethods.symplectic import SymIntegrator

# Three-body 3D problem:
def three_body(t, y, par=[1., 0.1, 2., 1.]):
    # par = [G, m1, m2, m3]
    # y is of the form [x1, vx1, y1, vy1, z1, vz1, x2, vx2, y2, vy2, z2, vz2, x3, vx3, y3, vy3, z3, vz3]
    r12 = np.sqrt(np.power(y[0]-y[6], 2)+np.power(y[2]-y[8], 2)+np.power(y[4]-y[10], 2))
    r23 = np.sqrt(np.power(y[6]-y[12], 2)+np.power(y[8]-y[14], 2)+np.power(y[10]-y[16], 2))
    r13 = np.sqrt(np.power(y[0]-y[12], 2)+np.power(y[2]-y[14], 2)+np.power(y[4]-y[16], 2))
    fg12 = -par[0]*np.divide(1., np.power(r12, 2))
    fg23 = -par[0]*np.divide(1., np.power(r23, 2))
    fg13 = -par[0]*np.divide(1., np.power(r13, 2))
    
    v = y[1::2]
    ax1 = par[2]*fg12*np.divide((y[0]-y[6]), r12) + par[3]*fg13*np.divide((y[0]-y[12]), r13)
    ay1 = par[2]*fg12*np.divide((y[2]-y[8]), r12) + par[3]*fg13*np.divide((y[2]-y[14]), r13)
    az1 = par[2]*fg12*np.divide((y[4]-y[10]), r12) + par[3]*fg13*np.divide((y[4]-y[16]), r13)

    ax2 = par[1]*fg12*np.divide((-y[0]+y[6]), r12) + par[3]*fg23*np.divide((y[6]-y[12]), r23)
    ay2 = par[1]*fg12*np.divide((-y[2]+y[8]), r12) + par[3]*fg23*np.divide((y[8]-y[14]), r23)
    az2 = par[1]*fg12*np.divide((-y[4]+y[10]), r12) + par[3]*fg23*np.divide((y[10]-y[16]), r23)

    ax3 = par[1]*fg13*np.divide((-y[0]+y[12]), r13) + par[2]*fg23*np.divide((-y[6]+y[12]), r23)
    ay3 = par[1]*fg13*np.divide((-y[2]+y[14]), r13) + par[2]*fg23*np.divide((-y[8]+y[14]), r23)
    az3 = par[1]*fg13*np.divide((-y[4]+y[16]), r13) + par[2]*fg23*np.divide((-y[10]+y[16]), r23)
    return [v[0], ax1, v[1], ay1, v[2], az1, v[3], ax2, v[4], ay2, v[5], az2, v[6], ax3, v[7], ay3, v[8], az3]

par=[2.5, 1., 2., 3.]
n=1000
h=0.2
y0_1 = [-7.,-0.7,7.,0.2,0.,-0.1]
y0_2 = [-12.,0.5,17.,-0.2,-4.,0.2]
y0_3 = [-17.,0.2,12.,0.,5.,-0.3]
y0 = np.concatenate((y0_1, y0_2, y0_3))

# SymIntegrator methods
problem_si = SymIntegrator(sym_methods["PEFRL"], three_body, par)
si1 = problem_si.run(x0=0, y0=y0, stepnum=n, stepsize=h)
y0[0] += 0.0001
si2 = problem_si.run(x0=0, y0=y0, stepnum=n, stepsize=h)
y0[0] += 0.0001
si3 = problem_si.run(x0=0, y0=y0, stepnum=n, stepsize=h)

# 3D Plotting Script:----------------------------------------------------------------
def update_lines(num, dataLines, lines):
    for line, data in zip(lines, dataLines):
        line.set_data(data[0:2, :num])
        line.set_3d_properties(data[2,:num])
    return lines

# Attaching 3D axis to the figure
fig = plt.figure()
ax = p3.Axes3D(fig)

data = np.empty((9, 3, n+1))
data[0] = si1[1][:6:2]
data[1] = si1[1][6:12:2]
data[2] = si1[1][12::2]
data[3] = si2[1][:6:2]
data[4] = si2[1][6:12:2]
data[5] = si2[1][12::2]
data[6] = si3[1][:6:2]
data[7] = si3[1][6:12:2]
data[8] = si3[1][12::2]

lines = []
colors = ["darkgreen","green","limegreen","darkred","red","lightcoral","blue","royalblue","cornflowerblue"]
for i in range(len(data)):
    lines.append(ax.plot(data[i][0, 0:1], data[i][1, 0:1], data[i][2, 0:1])[0])
    lines[-1].set_color(colors[i])

ax.set_xlim3d([-20.0, 20.0])
ax.set_ylim3d([-20.0, 20.0])
ax.set_zlim3d([-20.0, 20.0])

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Three-body problem')

line_ani = animation.FuncAnimation(fig, update_lines, n+1, fargs=(data, lines), interval=1, blit=False)
plt.show()