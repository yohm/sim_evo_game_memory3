# %%
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
plt.rcParams['font.size'] = 22
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['savefig.dpi'] = 200
plt.rcParams['figure.facecolor'] = 'white'

# %%
def plot_bar_chart(mem1, mem3, ylabel, figname = None):
  plt.clf()
  cmap = list(mpl.colors.TABLEAU_COLORS)
  fig, ax1 = plt.subplots()
  ax1.set_ylim(-0.0, 1.02)
  ax1.set_yticks([0,0.5,1.0])
  ax1.set_ylabel(ylabel)
  bar_width = 0.45
  index = np.arange(3)
  ax1.bar(index-bar_width/2, mem1, bar_width-0.03, alpha=0.8, color=cmap[0], label="memory-1")
  ax1.bar(index+bar_width/2, mem3, bar_width-0.03, alpha=0.8, color=cmap[1], label="memory-3")
  ax1.set_xticks((0, 1, 2))
  ax1.set_xticklabels((r'$b=1.5$', r'b=3', r'b=6'))
  ax1.legend(fontsize = 16)
  if figname:
    fig.savefig(figname)
  plt.show()

#%%
# http://localhost:3333/parameter_sets/631d638b7a3aca027097d78b
# http://localhost:3333/parameter_sets/6310112d7a3aca0272a3fb7b
mem1 = [0.2950388672433676, 0.42923348693292906, 0.9999969999856321]
mem3 = [0.4566493199762133, 0.5608140735195349, 0.6602537025013502]
plot_bar_chart(mem1, mem3, "cooperation level", "cooperation_level_bar.pdf")

# %%
mem1 = [0.06403673781859757, 0.2330904812116458, 1.0]
mem3 = [0.030186144651271833, 0.11389601544001718, 0.24623249581388426]
plot_bar_chart(mem1, mem3, "fractions", "efficient_fraction.pdf")
# %%
mem1 = [0.7784968649965167, 0.616502240558045, 0.0]
mem3 = [0.36318984798872, 0.28587142874603194, 0.20996456662729623]
plot_bar_chart(mem1, mem3, "fractions", "rival_fraction.pdf")
# %%
mem1 = [0.0, 0.0, 0.0]
mem3 = [0.2333122592358436, 0.27452752725280805, 0.2626500696111885]
plot_bar_chart(mem1, mem3, "fractions", "fr_fraction.pdf")

# %%
