import scienceplots
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

plt.style.use(['ieee'])
plt.rcParams['xtick.labelsize'] = 40
plt.rcParams['ytick.labelsize'] = 40
plt.rcParams['axes.titlesize'] = 40
plt.rcParams['axes.labelsize'] = 40
plt.rcParams['lines.linewidth'] = 5
plt.rcParams['lines.markersize'] = 15
plt.rcParams['legend.fontsize'] = 40
path = 'util/calibri.ttf'
fm.fontManager.addfont(path)
prop = fm.FontProperties(fname=path)
plt.rcParams['font.family'] = prop.get_name()
plt.rcParams['mathtext.fontset'] = 'dejavusans'
colors = [(2/255,48/255,71/255), (255/255,202/255,95/255), (26/255,134/255,163/255), (253/255,152/255,2/255), (70/255,172/255,202/255), (14/255,91/255,118/255), (155/255,207/255,232/255), (251/255,132/255,2/255)]
color_bar = [(2/255,48/255,71/255), (14/255,91/255,118/255), (26/255,134/255,163/255), (70/255,172/255,202/255), (155/255,207/255,232/255), (243/255,249/255,252/255), (255/255,202/255,95/255), (254/255,168/255,9/255), (253/255,152/255,2/255), (251/255,132/255,2/255)]
markers = ['*', 'o', 's', 'D', 'v', '^', 'h']
from matplotlib.colors import LinearSegmentedColormap
my_cmap = LinearSegmentedColormap.from_list("mycmap", color_bar)