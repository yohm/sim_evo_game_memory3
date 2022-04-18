#%%
import numpy as np
import matplotlib.pyplot as plt

#%%
plt.rcParams['font.family'] ='sans-serif'
plt.rcParams["figure.subplot.left"] = 0.22
plt.rcParams["figure.subplot.right"] = 0.95
plt.rcParams["figure.subplot.bottom"] = 0.20
plt.rcParams["figure.subplot.top"] = 0.95
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.major.width'] = 1.0
plt.rcParams['ytick.major.width'] = 1.0
plt.rcParams['axes.labelsize'] = 30
plt.rcParams['font.size'] = 22
plt.rcParams['axes.linewidth'] = 1.2
# %%
x = np.loadtxt("abundance.dat")
x
#%%
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
ax.bar(x_position, x)
ax.set_xticks(x_position)
ax.set_xticklabels(label, minor=False, rotation=90)
#plt.ylim((0,1))

# %%
plt.savefig("abundance.png")
# %%
psi = np.loadtxt("fixation_probs.dat")
psi

# %%
plt.clf()
plt.rcParams["figure.subplot.left"] = 0.18
plt.rcParams["figure.subplot.right"] = 0.95
plt.rcParams["figure.subplot.bottom"] = 0.05
plt.rcParams["figure.subplot.top"] = 0.8
fig, ax = plt.subplots()
heatmap = ax.pcolor(psi, cmap=plt.cm.Blues, vmin=0, vmax=1)
ax.set_xticks(np.arange(psi.shape[0]) + 0.5, minor=False)
ax.set_yticks(np.arange(psi.shape[1]) + 0.5, minor=False)
ax.invert_yaxis()
ax.xaxis.tick_top()
ax.set_xticklabels(label, minor=False, rotation=90)
ax.set_yticklabels(label, minor=False)
# %%
plt.savefig("fixation_probs.png")
