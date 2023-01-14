#%%
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
plt.rcParams['axes.labelsize'] = 22
plt.rcParams['font.size'] = 16
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['savefig.dpi'] = 200
plt.rcParams['figure.facecolor'] = 'white'
cmap = plt.rcParams['axes.prop_cycle'].by_key()['color']
# %%
def plot(dat, figpath=None):
  plt.clf()

  digit = 5
  xscale = 10**digit

  plt.ylim((-0.002,1.002))
  plt.xlim((0,10))
  plt.xlabel( r'time ($\times 10^{' + f"{digit:.0f}" + r'}$)')
  plt.ylabel(r'fraction')
  # plt.plot(dat[:,0]/xscale, dat[:,1], '.-', label='cooperation level')
  plt.plot(dat[:,0]/xscale, dat[:,2], '.-', label='efficient')
  plt.plot(dat[:,0]/xscale, dat[:,3], '.-', label='rival')
  plt.plot(dat[:,0]/xscale, dat[:,4], '.-', label='friendly rival')
  plt.legend(loc='upper right')
  if figpath:
    plt.savefig(figpath)
  plt.show()
#%%
# http://localhost:3333/runs/631015437a3aca0272a3fbf6
# (N=2, M=1000, benefit=1.5, error_rate=1.0e-06, sigma_in_b=30.0, sigma_out_b=3.0, strategy_space=[3, 3], weighted_sampling=1, initial_condition=random, excluding_strategies=[], T_max=1000000, T_print=1000, T_init=100000)
dat = np.loadtxt('../../oacis/Result/63100b1c7a3aca0272a3fb33/6310112d7a3aca0272a3fb7a/631015437a3aca0272a3fbf6/timeseries.dat')
plot(dat, 'm3_grouped_lowmut_b15_timeseries.pdf')
# %%
# http://localhost:3333/runs/631015437a3aca0272a3fbec
# (N=2, M=1000, benefit=3.0, error_rate=1.0e-06, sigma_in_b=30.0, sigma_out_b=3.0, strategy_space=[3, 3], weighted_sampling=1, initial_condition=random, excluding_strategies=[], T_max=1000000, T_print=1000, T_init=100000)
dat = np.loadtxt('../../oacis/Result/63100b1c7a3aca0272a3fb33/6310112d7a3aca0272a3fb7c/631015437a3aca0272a3fbec/timeseries.dat')
plot(dat, 'm3_grouped_lowmut_b3_timeseries.pdf')
# %%
# http://localhost:3333/runs/631015437a3aca0272a3fbe1
# (N=2, M=1000, benefit=6.0, error_rate=1.0e-06, sigma_in_b=30.0, sigma_out_b=3.0, strategy_space=[3, 3], weighted_sampling=1, initial_condition=random, excluding_strategies=[], T_max=1000000, T_print=1000, T_init=100000)
dat = np.loadtxt('../../oacis/Result/63100b1c7a3aca0272a3fb33/6310112d7a3aca0272a3fb7e/631015437a3aca0272a3fbe1/timeseries.dat')
plot(dat, 'm3_grouped_lowmut_b6_timeseries.pdf')
# %%
