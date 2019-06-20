import json

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

from svm.svm_roc import get_axes

mpl.rcParams['legend.fontsize'] = 10

fig = plt.figure()
ax = fig.gca(projection='3d')
theta = np.linspace(-4 * np.pi, 4 * np.pi, 100)
z = np.linspace(-2, 2, 100)
r = z**2 + 1
files = open('/Users/preethi/Allclass/297/data1/pree2.json')
f1 = json.load(files)
x, y, z = get_axes(f1)
x = x[3:20]
y = y[3:20]
z = z[3:20]
ax.plot(x, y, z, label='authentic user')
ax.legend()

plt.show()
