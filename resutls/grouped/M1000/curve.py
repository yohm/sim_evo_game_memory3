#%%
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
x = np.arange(0, 10, 0.1)
y = np.sin(x)
plt.plot(x, y)
# %%
import matplotlib.patches as mpatches
from matplotlib.path import Path
d1 = 0.02 # X軸のはみだし量
d2 = 0.05 # ニョロ波の高さ
wn = 21   # ニョロ波の数（奇数値を指定）

fig, ax = plt.subplots()
pp = (0,d2,0,-d2)
px = np.linspace(-d1,1+d1,wn)
py = np.array([0.1+pp[i%4] for i in range(0,wn)])
p = Path(list(zip(py,px)), [Path.MOVETO]+[Path.CURVE3]*(wn-1))

line1 = mpatches.PathPatch(p, lw=6, edgecolor='black',
                          facecolor='None', clip_on=False,
                          transform=ax.transAxes, zorder=10)
line2 = mpatches.PathPatch(p,lw=4, edgecolor='white',
                           facecolor='None', clip_on=False,
                           transform=ax.transAxes, zorder=10,
                           capstyle='round')
a = ax.add_patch(line1)
a = ax.add_patch(line2)
p

# %%
