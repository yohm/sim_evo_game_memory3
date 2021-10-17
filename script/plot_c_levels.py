#%%
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors

plt.rcParams['font.family'] ='sans-serif'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.major.width'] = 1.0
plt.rcParams['ytick.major.width'] = 1.0
plt.rcParams['axes.labelsize'] = 30
plt.rcParams['font.size'] = 22
plt.rcParams['axes.linewidth'] = 1.2



# %%
dat = np.loadtxt("cooperation_level.dat")

# %%
fig1, ax1 = plt.subplots()
ax1.set_xscale('log')
ax1.set_xticks([2,4,8,16,32])
ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax1.set_xlabel(r'$N$')
ax1.set_ylabel(r'$b/c$')
benefit = dat[0,1:]
n = dat[1:,0]
extent = (n[0],n[-1], 1.0,benefit[-1])
im = plt.imshow(dat[-1:0:-1,1:], vmin=0.0, vmax=1.0, extent=extent, aspect='auto')
fig1.colorbar(im)
# %%
dat[0,:], extent
# %%
