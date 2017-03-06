import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import colors, ticker, cm
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import Impurity


R = np.array( [[0,0],
               [1.0,0.7]])
V = np.array( [0.25, 3.2])
B = np.array( [[ 1,1,1],[ 1,1,1]])

a = Impurity.Imp(R, V, B)


fig = plt.figure()
ax = fig.gca(projection='3d')
X = np.linspace(-0.5, 1.5, 200)
Y = np.linspace(-0.5, 1.5, 200)
X, Y = np.meshgrid(X, Y)
print(X.shape)
r = np.array([0.0,0.0])

Z = np.zeros(X.shape)
for i in range(Z.shape[0]):
    for j in range(Z.shape[1]):
        r[0] = X[i,j]
        r[1] = Y[i,j]
        Z[i,j] = np.abs(a.d_H(r)[0,0])

# Plot the surface.

surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()


