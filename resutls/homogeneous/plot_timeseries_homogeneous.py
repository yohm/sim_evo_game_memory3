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
  plt.xlim((5,10))
  plt.xlabel( r'time ($\times 10^{' + f"{digit:.0f}" + r'}$)')
  plt.ylabel(r'fraction')
  # plt.plot(dat[:,0]/xscale, dat[:,1], '.-', label='cooperation level')
  plt.plot(dat[:,0]/xscale, dat[:,2], '.-', label='non-FR efficient')
  plt.plot(dat[:,0]/xscale, dat[:,3], '.-', label='non-FR rival')
  plt.plot(dat[:,0]/xscale, dat[:,4], '.-', label='FR')
  plt.legend()
  if figpath:
    plt.savefig(figpath)
  plt.show()
#%%
# http://localhost:3333/runs/631049f37a3aca027357d291
# (N=64, M=1, benefit=6.0, error_rate=1.0e-06, sigma_in_b=30.0, sigma_out_b=0.0, strategy_space=[3, 3], weighted_sampling=1, initial_condition=random, excluding_strategies=[], T_max=1000000, T_print=1000, T_init=100000)
dat = np.loadtxt("../../oacis/Result/6310429b7a3aca0272a3fc39/631049f27a3aca027357d162/631049f37a3aca027357d291/timeseries.dat")
plot(dat, "m3_homogeneous_b6_timeseries.pdf")
# %%
# http://localhost:3333/runs/631049f37a3aca027357d28f
# (N=64, M=1, benefit=1.5, error_rate=1.0e-06, sigma_in_b=30.0, sigma_out_b=0.0, strategy_space=[3, 3], weighted_sampling=1, initial_condition=random, excluding_strategies=[], T_max=1000000, T_print=1000, T_init=100000)
dat = np.loadtxt("../../oacis/Result/6310429b7a3aca0272a3fc39/631049f27a3aca027357d160/631049f37a3aca027357d28f/timeseries.dat")
plot(dat, "m3_homogeneous_b15_timeseries.pdf")
# %%
# http://localhost:3333/runs/631054e57a3aca027357d3f0
# (N=64, M=1, benefit=1.5, error_rate=1.0e-06, sigma_in_b=30.0, sigma_out_b=0.0, strategy_space=[1, 1], weighted_sampling=1, initial_condition=random, excluding_strategies=[], T_max=1000000, T_print=1000, T_init=100000)
dat = np.loadtxt("../../oacis/Result/6310429b7a3aca0272a3fc39/631054e57a3aca027357d2a6/631054e57a3aca027357d3f0/timeseries.dat")
plot(dat, "m1_homogeneous_b15_timeseries.pdf")
# %%
# http://localhost:3333/runs/631054e57a3aca027357d3f2
# (N=64, M=1, benefit=6.0, error_rate=1.0e-06, sigma_in_b=30.0, sigma_out_b=0.0, strategy_space=[1, 1], weighted_sampling=1, initial_condition=random, excluding_strategies=[], T_max=1000000, T_print=1000, T_init=100000)
dat = np.loadtxt("../../oacis/Result/6310429b7a3aca0272a3fc39/631054e57a3aca027357d2a8/631054e57a3aca027357d3f2/timeseries.dat")
dat[:,4] = dat[:,4] - 0.002
#dat[:,3] = dat[:,3] + 0.01
plot(dat, "m1_homogeneous_b6_timeseries.pdf")
# %%
