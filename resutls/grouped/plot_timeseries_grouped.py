#%%
import os
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
plt.rcParams['axes.labelsize'] = 18
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
  plt.xlabel( r'time ($\times 10^{8}$)')
  plt.ylabel(r'fraction')
  # plt.plot(dat[:,0]/xscale, dat[:,1], '.-', label='cooperation level')
  plt.plot(dat[:,0]/xscale, dat[:,3], '.-', label='non-FR efficient')
  plt.plot(dat[:,0]/xscale, dat[:,4], '.-', label='non-FR rival')
  plt.plot(dat[:,0]/xscale, dat[:,2], '.-', label='FR')
  plt.legend()
  if figpath:
    plt.savefig(figpath)
  plt.show()

# %%
# Results for strong out-group selection

# (N=2, M=1000, benefit=3.0, error_rate=1.0e-06, sigma_in_b=30.0, sigma_out_b=30.0, p_nu=0.01, strategy_space=[3, 3], weighted_sampling=1, parallel_update=0, initial_condition=random, T_max=1000000, T_print=1000, T_init=1000, excluding_strategies=[], alld_mutant_prob=0.0)
oacis_root = os.path.expanduser('~/work/sim_evo_game_memory3/oacis/Result')
dat = np.loadtxt(f"{oacis_root}/62d9ffb67a3aca0271947506/6327c35b7a3aca027def9e52/6327c35b7a3aca027def9e54/timeseries.dat")
plot(dat, "m3_grouped_b3_strong_out_timeseries.pdf")

# %%
# http://localhost:3333/runs/632868517a3aca027def9e95
# (N=2, M=1000, benefit=3.0, error_rate=1.0e-06, sigma_in_b=30.0, sigma_out_b=30.0, p_nu=0.0001, strategy_space=[3, 3], weighted_sampling=1, parallel_update=0, initial_condition=random, T_max=1000000, T_print=1000, T_init=1000, excluding_strategies=[], alld_mutant_prob=0.0)
dat = np.loadtxt(f"{oacis_root}/62d9ffb67a3aca0271947506/632868517a3aca027def9e90/632868517a3aca027def9e95/timeseries.dat")
#plot(dat, "m3_grouped_b3_strong_out_mut1e-4_timeseries.pdf")
figpath = "m3_grouped_b3_strong_out_mut1e-4_timeseries.pdf"
plt.clf()

digit = 5
xscale = 10**digit

plt.ylim((-0.003,1.003))
plt.xlim((2,5))
plt.xticks([2,3,4,5])
plt.xlabel(r'time ($\times 10^{8}$)')
plt.ylabel(r'fraction')
# plt.plot(dat[:,0]/xscale, dat[:,1], '.-', label='cooperation level')
plt.plot(dat[:,0]/xscale, dat[:,3], '.-', label='non-FR efficient')
plt.plot(dat[:,0]/xscale, dat[:,4], '.-', label='non-FR rival')
plt.plot(dat[:,0]/xscale, dat[:,2], '.-', label='FR')
plt.legend(loc='right')
if figpath:
  plt.savefig(figpath)
plt.show()


# %%
# (N=2, M=1000, benefit=3.0, error_rate=1.0e-06, sigma_in_b=30.0, sigma_out_b=30.0, p_nu=0.01, strategy_space=[1, 1], weighted_sampling=1, parallel_update=0, initial_condition=random, T_max=1000000, T_print=1000, T_init=1000, excluding_strategies=[], alld_mutant_prob=0.0)
# http://localhost:3333/runs/6327c3687a3aca027def9e5f
dat = np.loadtxt(f"{oacis_root}/62d9ffb67a3aca0271947506/6327c3687a3aca027def9e5e/6327c3687a3aca027def9e5f/timeseries.dat")
plot(dat, "m1_grouped_b3_strong_out_timeseries.pdf")

# Results for weak out-group selection

#%%
# http://localhost:3333/runs/630977a27a3aca027e386639
# (N=2, M=1000, benefit=1.5, error_rate=1.0e-06, sigma_in_b=30.0, sigma_out_b=3.0, p_nu=0.01, strategy_space=[3, 3], weighted_sampling=1, parallel_update=0, initial_condition=random, T_max=1000000, T_print=1000, T_init=1000, excluding_strategies=[], alld_mutant_prob=0.0)
oacis_root = os.path.expanduser('~/work/sim_evo_game_memory3/oacis/Result')
dat = np.loadtxt(f"{oacis_root}/62d9ffb67a3aca0271947506/630977a07a3aca027cf8db6c/630977a27a3aca027e386639/timeseries.dat")
plot(dat, "m3_grouped_b15_timeseries.pdf")
# %%
# http://localhost:3333/runs/62da37177a3aca027b303ada
# (N=2, M=1000, benefit=3.0, error_rate=1.0e-06, sigma_in_b=30.0, sigma_out_b=3.0, p_nu=0.01, strategy_space=[3, 3], weighted_sampling=1, parallel_update=0, initial_condition=random, T_max=1000000, T_print=1000, T_init=1000, excluding_strategies=[], alld_mutant_prob=0.0)
dat = np.loadtxt(f"{oacis_root}/62d9ffb67a3aca0271947506/62da37127a3aca027194755e/62da37177a3aca027b303ada/timeseries.dat")
plot(dat, "m3_grouped_b3_timeseries.pdf")
# %%
# http://localhost:3333/runs/630977a27a3aca027e386648
# (N=2, M=1000, benefit=6.0, error_rate=1.0e-06, sigma_in_b=30.0, sigma_out_b=3.0, p_nu=0.01, strategy_space=[3, 3], weighted_sampling=1, parallel_update=0, initial_condition=random, T_max=1000000, T_print=1000, T_init=1000, excluding_strategies=[], alld_mutant_prob=0.0)
dat = np.loadtxt(f"{oacis_root}/62d9ffb67a3aca0271947506/630977a27a3aca027e38662a/630977a27a3aca027e386648/timeseries.dat")
plot(dat, "m3_grouped_b6_timeseries.pdf")

# %%
# http://localhost:3333/runs/62dab7de7a3aca0271947574
# (N=2, M=1000, benefit=3.0, error_rate=1.0e-06, sigma_in_b=30.0, sigma_out_b=3.0, p_nu=1.0e-05, strategy_space=[3, 3], weighted_sampling=1, parallel_update=0, initial_condition=random, T_max=1000000, T_print=1000, T_init=1000, excluding_strategies=[], alld_mutant_prob=0.0)
dat = np.loadtxt(f"{oacis_root}/62d9ffb67a3aca0271947506/62dab7de7a3aca0271947571/62dab7de7a3aca0271947574/timeseries.dat")
plot(dat, "m3_grouped_b3_pnu1e-5_timeseries.pdf")
# %%
# http://localhost:3333/runs/631736d27a3aca027357d688
# (N=2, M=1000, benefit=3.0, error_rate=1.0e-06, sigma_in_b=30.0, sigma_out_b=3.0, p_nu=0.01, strategy_space=[1, 1], weighted_sampling=1, parallel_update=0, initial_condition=random, T_max=1000000, T_print=1000, T_init=1000, excluding_strategies=[], alld_mutant_prob=0.0)
dat = np.loadtxt(f"{oacis_root}/62d9ffb67a3aca0271947506/631736d27a3aca027357d661/631736d27a3aca027357d688/timeseries.dat")
plot(dat, "m1_grouped_b3_timeseries.pdf")
# %%
