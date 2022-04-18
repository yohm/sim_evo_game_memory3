#%%
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)

# %%
plt.rcParams['font.family'] ='sans-serif'
plt.rcParams["figure.subplot.left"] = 0.2
plt.rcParams["figure.subplot.right"] = 0.95
plt.rcParams["figure.subplot.bottom"] = 0.20
plt.rcParams["figure.subplot.top"] = 0.95
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.major.width'] = 1.0
plt.rcParams['ytick.major.width'] = 1.0
plt.rcParams['axes.labelsize'] = 30
plt.rcParams['font.size'] = 22
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['savefig.dpi'] = 200
plt.rcParams['figure.facecolor'] = 'white'
# %%
dat = np.loadtxt("frequency.dat")
# %%
plt.clf()

label = [""] * 16
label[6] = "WSLS"
label[10] = "TFT"
label[0] = "ALLC"
label[15] = "ALLD"
label[14] = "GRIM"
x_position = np.arange(0, 16)
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.bar(x_position, dat)
ax.set_xticks(x_position)
ax.set_xticklabels(label, rotation=90)

# %%
plt.savefig("frequency.png")
