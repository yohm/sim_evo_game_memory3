# %%
import matplotlib.pyplot as plt
import numpy as np
import json
# %%
with open("results.json") as f:
  loaded = json.load(f)
loaded
# %%
def calculate_avg_yerr_n(y_key, b, p_nu, s_space, sigma_out_b):
  dat = [l['output'][y_key] for l in loaded if l['input']['benefit'] == b and l['input']['p_nu'] == p_nu and l['input']['strategy_space'] == s_space and l['input']['sigma_out_b'] == sigma_out_b]
  n = len(dat)
  avg = np.array(dat).sum() / len(dat)
  err= np.array(dat).std() / np.sqrt(n)
  return avg, err, n

# %%
def prepare_y_yerr(y_key, p_nu_array, b, s_space, sigma_out_b):
  y = []
  yerr = []
  na = []
  for p_nu in p_nu_array:
    avg, err, n = calculate_avg_yerr_n(y_key, b, p_nu, s_space, sigma_out_b)
    y.append(avg)
    yerr.append(err)
    na.append(n)
  return y, yerr, na

# %%
x = [0.1, 0.04,0.02,0.01, 0.004,0.002,0.001, 0.0004,0.0002,0.0001, 0.00004,0.00002,0.00001]
y, yerr, na = prepare_y_yerr('cooperation_level', x, 1.5, [3,3], 3.0)
y, yerr, na
# %%

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)

plt.rcParams['font.family'] ='sans-serif'
plt.rcParams["figure.subplot.left"] = 0.2
plt.rcParams["figure.subplot.right"] = 0.95
plt.rcParams["figure.subplot.bottom"] = 0.20
plt.rcParams["figure.subplot.top"] = 0.95
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.major.width'] = 1.0
plt.rcParams['ytick.major.width'] = 1.0
plt.rcParams['axes.labelsize'] =18
plt.rcParams['font.size'] = 16
plt.rcParams['axes.linewidth'] = 1.2
mpl.rcParams['lines.markersize'] = 6
plt.rcParams['savefig.dpi'] = 200
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams["errorbar.capsize"] = 2
plt.rcParams['lines.marker'] = 'o'
plt.rcParams['lines.linestyle'] = '-'

# %%
def make_plot(s_space, sigma_out_b, y_key, outfilepath=None, legend_loc=None):
  plt.clf()
  fig1, ax1 = plt.subplots()
  ax1.set_xscale("log")
  ax1.set_ylim(-0.02, 1.02)
  ax1.set_xlabel("relative mutation probability, $r$")
  if y_key == 'cooperation_level':
    ax1.set_ylabel("cooperation level")
  else:
    ax1.set_ylabel("fraction")
  ax1.set_yticks([0, 0.5, 1])

  x = [0.1, 0.04,0.02,0.01, 0.004,0.002,0.001, 0.0004,0.0002,0.0001, 0.00004,0.00002,0.00001]
  y, yerr, na = prepare_y_yerr(y_key, x, 6, s_space, sigma_out_b)
  ax1.errorbar(x, y, yerr=yerr, label=r"$b=6$", marker='^')

  y, yerr, na = prepare_y_yerr(y_key, x, 3, s_space, sigma_out_b)
  ax1.errorbar(x, y, yerr=yerr, label=r"$b=3$", marker='o')

  y, yerr, na = prepare_y_yerr(y_key, x, 1.5, s_space, sigma_out_b)
  ax1.errorbar(x, y, yerr=yerr, label=r"$b=1.5$", marker='v')

  if legend_loc:
    ax1.legend(loc=legend_loc)
  if outfilepath:
    fig1.savefig(outfilepath)
  plt.show()

# %%
make_plot([3,3], 3.0, 'cooperation_level', 'm3_cooperation_level.pdf', 'lower right')

# %%
make_plot([3,3], 3.0, 'efficient_fraction', 'm3_efficient_fraction.pdf')

# %%
make_plot([3,3], 3.0, 'defensible_fraction', 'm3_rival_fraction.pdf')

# %%
make_plot([3,3], 3.0, 'friendly_rival_fraction', 'm3_friendly_rival_fraction.pdf')

# %%
make_plot([3,3], 30.0, 'cooperation_level', 'm3_cooperation_level_sigma_out_30.pdf', 'lower right')
make_plot([3,3], 30.0, 'efficient_fraction', 'm3_efficient_fraction_sigma_out_30.pdf')
make_plot([3,3], 30.0, 'defensible_fraction', 'm3_rival_fraction_sigma_out_30.pdf')
make_plot([3,3], 30.0, 'friendly_rival_fraction', 'm3_friendly_rival_fraction_sigma_out_30.pdf')

# %%
make_plot([1,1], 3.0, 'cooperation_level', 'm1_cooperation_level.pdf', 'lower right')
make_plot([1,1], 3.0, 'efficient_fraction', 'm1_efficient_fraction.pdf')
make_plot([1,1], 3.0, 'defensible_fraction', 'm1_rival_fraction.pdf')
make_plot([1,1], 3.0, 'friendly_rival_fraction', 'm1_friendly_rival_fraction.pdf')

# %%
make_plot([1,1], 30.0, 'cooperation_level', 'm1_cooperation_level_sigma_out_30.pdf', 'lower right')
make_plot([1,1], 30.0, 'efficient_fraction', 'm1_efficient_fraction_sigma_out_30.pdf')
make_plot([1,1], 30.0, 'defensible_fraction', 'm1_rival_fraction_sigma_out_30.pdf')
make_plot([1,1], 30.0, 'friendly_rival_fraction', 'm1_friendly_rival_fraction_sigma_out_30.pdf')
# %%
def plot_mem_length(s_space, sigma_out_b, benefit, outfilepath=None):
  plt.clf()
  fig1, ax1 = plt.subplots()
  ax1.set_xscale("log")
  ax1.set_xlim(7e-6,1.5e-1)
  ax1.set_ylim(-0.02, 1.02)
  ax1.set_xlabel("relative mutation probability, $r$")
  ax1.set_ylabel("memory length")
  if s_space[0] == 1:
    ax1.set_yticks([0, 0.5, 1.0])
    ax1.plot([0, 1], [0.5, 0.5], color='gray', linestyle='dotted')
  elif s_space[0] == 3:
    ax1.set_yticks([0, 1.5, 3])
    ax1.plot([0, 1], [1.5, 1.5], color='gray', linestyle='dotted')
  else:
    raise ValueError("s_space[0] must be 1 or 3")

  x = [0.1, 0.04,0.02,0.01, 0.004,0.002,0.001, 0.0004,0.0002,0.0001, 0.00004,0.00002,0.00001]
  cmap = plt.rcParams['axes.prop_cycle'].by_key()['color']
  y, yerr, na = prepare_y_yerr('mem_length_self', x, benefit, s_space, sigma_out_b)
  ax1.errorbar(x, y, yerr=yerr, label=r"$m_1$", marker='^', color=cmap[1])

  y, yerr, na = prepare_y_yerr('mem_length_opponent', x, benefit, s_space, sigma_out_b)
  ax1.errorbar(x, y, yerr=yerr, label=r"$m_2$", marker='v', linestyle='dashed', color=cmap[1])

  ax1.legend()
  if outfilepath:
    fig1.savefig(outfilepath)

# %%
def plot_mem_length_both(s_space, benefit, outfilepath=None):
  plt.clf()
  fig1, ax1 = plt.subplots()
  ax1.set_xscale("log")
  ax1.set_xlim(7e-6,1.5e-1)
  ax1.set_ylim(-0.02, 1.02)
  ax1.set_xlabel("relative mutation probability, $r$")
  ax1.set_ylabel("memory length")
  if s_space[0] == 1:
    ax1.set_yticks([0, 0.5, 1.0])
    ax1.plot([0, 1], [0.5, 0.5], color='gray', linestyle='dotted')
  elif s_space[0] == 3:
    ax1.set_yticks([0, 1.5, 3])
    ax1.set_yticks([0.5,1.0,2.0,2.5], minor=True)
    ax1.plot([0, 1], [1.5, 1.5], color='gray', linestyle='dotted')
  else:
    raise ValueError("s_space[0] must be 1 or 3")

  ax1.plot([], [], color='gray', label=r"$m_1$", marker='^', linestyle='solid')
  ax1.plot([], [], color='gray', label=r"$m_2$", marker='v', linestyle='dashed')

  x = [0.1, 0.04,0.02,0.01, 0.004,0.002,0.001, 0.0004,0.0002,0.0001, 0.00004,0.00002,0.00001]
  cmap = plt.rcParams['axes.prop_cycle'].by_key()['color']
  sigma_out_b = 30.0
  y, yerr, na = prepare_y_yerr('mem_length_self', x, benefit, s_space, sigma_out_b)
  ax1.errorbar(x, y, yerr=yerr, marker='^', color=cmap[0])

  y, yerr, na = prepare_y_yerr('mem_length_opponent', x, benefit, s_space, sigma_out_b)
  ax1.errorbar(x, y, yerr=yerr, marker='v', linestyle='dashed', color=cmap[0])

  sigma_out_b = 3.0
  y, yerr, na = prepare_y_yerr('mem_length_self', x, benefit, s_space, sigma_out_b)
  ax1.errorbar(x, y, yerr=yerr, marker='^', color=cmap[1])

  y, yerr, na = prepare_y_yerr('mem_length_opponent', x, benefit, s_space, sigma_out_b)
  ax1.errorbar(x, y, yerr=yerr, marker='v', linestyle='dashed', color=cmap[1])

  ax1.text(1.5e-5, 2.45, r'$\sigma_{\rm out}=15$', color=cmap[0])
  ax1.text(1.5e-5, 0.95, r'$\sigma_{\rm out}=1.5$', color=cmap[1])

  ax1.legend(loc='lower right')
  if outfilepath:
    fig1.savefig(outfilepath)

plot_mem_length_both([3,3], 3.0, 'm3_memory_length.pdf')
# %%
plot_mem_length([3,3], 3.0, 3, 'm3_memory_length_sigma_out_3.pdf')

# %%
plot_mem_length([1,1], 3.0, 3, 'm1_memory_length_sigma_out_3.pdf')

# %%
plot_mem_length([3,3], 30.0, 3, 'm3_memory_length_sigma_out_30.pdf')
# %%
plot_mem_length([1,1], 30.0, 3, 'm1_memory_length_sigma_out_30.pdf')
# %%
