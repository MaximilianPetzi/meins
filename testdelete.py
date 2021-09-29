from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

#mpl.rcParams["legend.fontsize"] = 10
fig = plt.figure()
ax = fig.gca(projection="3d")
theta = np.linspace(-4 * np.pi, 4 * np.pi, 100)
z = np.linspace(-2, 2, 100)
r = z**2 + 1
x = r * np.sin(theta)
y = r * np.cos(theta)
ax.plot(x, y, z, label="parametric curve")
#ax.plot(x, y, 2*z, label="second thingy")
ax.plot([0,0.001],[0,0],[0,0],linewidth=10)
ax.legend()
plt.show()