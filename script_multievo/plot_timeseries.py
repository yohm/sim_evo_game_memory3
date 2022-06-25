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

digit = (np.floor(np.log10(dat[-1,0])) // 3 * 3)
xscale = 10**digit

plt.ylim((0,1.0))
plt.xlabel( r'time ($\times 10^{' + f"{digit:.0f}" + r'}$)')
plt.ylabel(r'fraction')
plt.plot(dat[:,0]/xscale, dat[:,1], '.-', label='cooperation level')
plt.plot(dat[:,0]/xscale, dat[:,2], '.-', label='friendly rival')
plt.plot(dat[:,0]/xscale, dat[:,3], '.-', label='efficient')
plt.plot(dat[:,0]/xscale, dat[:,4], '.-', label='rival')
# plt.plot(dat[:,0]/xscale, dat[:,5], '.-', label='WSLS-like')
plt.legend()
# %%
plt.savefig("timeseries.png")

# %%
plt.clf()
xscale = 10**digit
plt.xlabel( r'time ($\times 10^{' + f"{digit:.0f}" + r'}$)')
plt.ylabel(r'diversity')
plt.plot(dat[:,0]/xscale, dat[:,6], '.-', label='diversity')
plt.savefig("diversity.png")


# %%
plt.clf()

digit = (np.floor(np.log10(dat[-1,0])) // 3 * 3)
xscale = 10**digit

plt.ylim((0,3.0))
plt.xlabel( r'time ($\times 10^{' + f"{digit:.0f}" + r'}$)')
plt.ylabel(r'memory length')
plt.plot(dat[:,0]/xscale, dat[:,7], '.-', label=r'$m_1$')
plt.plot(dat[:,0]/xscale, dat[:,8], '.-', label=r'$m_2$')
plt.legend()
# %%
plt.savefig("memory_lengths.png")

# %%
plt.clf()

digit = (np.floor(np.log10(dat[-1,0])) // 3 * 3)
xscale = 10**digit

plt.ylim((0,32.0))
plt.xlabel( r'time ($\times 10^{' + f"{digit:.0f}" + r'}$)')
plt.ylabel(r'automaton size')
plt.plot(dat[:,0]/xscale, dat[:,9], '.-', label=r'$simple$')
plt.plot(dat[:,0]/xscale, dat[:,10], '.-', label=r'$full$')
plt.legend()
# %%
plt.savefig("automaton_sizes.png")
