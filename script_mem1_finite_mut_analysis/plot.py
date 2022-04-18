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
dat = np.loadtxt("timeseries.dat")
# %%
plt.clf()

x = np.arange(0, dat.shape[0])
plt.ylim((0,1.0))
plt.ylabel(r'fraction')
labels = [str(i) for i in range(16)]
labels[0] = '0 AllC'
labels[6] = '6 WSLS'
labels[10] = '10 TFT'
labels[15] = '15 AllD'
for i in range(16):
  plt.plot(x, dat[:,i], '-', label=labels[i])
plt.legend(loc='upper left')
#plt.show()
# %%
plt.savefig("timeseries.png")

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
ax.bar(x_position, dat[-1])
ax.set_xticks(x_position)
ax.set_xticklabels(label, rotation=90)
#plt.show()

#%%
plt.savefig("abundance.png")
# %%
delta_p = np.loadtxt("delta_p.dat")
delta_p

# %%
plt.clf()
plt.rcParams["figure.subplot.left"] = 0.18
plt.rcParams["figure.subplot.right"] = 0.95
plt.rcParams["figure.subplot.bottom"] = 0.05
plt.rcParams["figure.subplot.top"] = 0.8
fig, ax = plt.subplots()
heatmap = ax.pcolor(delta_p, cmap=plt.cm.RdBu, vmin=-1, vmax=1)
ax.set_xticks(np.arange(delta_p.shape[0]) + 0.5, minor=False)
ax.set_yticks(np.arange(delta_p.shape[1]) + 0.5, minor=False)
ax.invert_yaxis()
ax.xaxis.tick_top()
ax.set_xticklabels(label, minor=False, rotation=90)
ax.set_yticklabels(label, minor=False)
#plt.show()
# %%
plt.savefig("delta_p.png")

# %%
rho = np.loadtxt("rho.dat")
rho


# %%
plt.clf()
plt.rcParams["figure.subplot.left"] = 0.18
plt.rcParams["figure.subplot.right"] = 0.95
plt.rcParams["figure.subplot.bottom"] = 0.05
plt.rcParams["figure.subplot.top"] = 0.8
fig, ax = plt.subplots()
heatmap = ax.pcolor(rho, cmap=plt.cm.Blues, vmin=0, vmax=1)
ax.set_xticks(np.arange(rho.shape[0]) + 0.5, minor=False)
ax.set_yticks(np.arange(rho.shape[1]) + 0.5, minor=False)
ax.invert_yaxis()
ax.xaxis.tick_top()
ax.set_xticklabels(label, minor=False, rotation=90)
ax.set_yticklabels(label, minor=False)
#plt.show()
# %%
plt.savefig("rho.png")

# %%
