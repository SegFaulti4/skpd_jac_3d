from run_polus import DATA_SIZES, PARALLEL_NUMS, EXPERIMENTS

import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.ticker as mticker
import numpy as np

y = DATA_SIZES
x = PARALLEL_NUMS
x, y = np.meshgrid(x, y)

for experiment in EXPERIMENTS:
    if experiment != 'single':
        z = np.genfromtxt(fname='./polus_res/' + experiment + '_res.csv', dtype=float, delimiter=',')

        fig = plt.figure(figsize=(11, 8))
        ax = fig.add_subplot(projection='3d')
        ax.title.set_text(experiment.upper() + ' perfomance')
        ax.set_xlabel('Parallel num')
        ax.set_ylabel('Data size')
        ax.set_zlabel('Mean time')

        # ax.plot_wireframe(x, y, np.log10(z), rstride=1, cstride=3)
        surf = ax.plot_surface(x, y, np.log10(z), cmap=cm.coolwarm, linewidth=0, antialiased=True)

        def log_tick_formatter(val, pos=None):
            return f"$10^{{{int(val)}}}$"

        ax.zaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
        ax.zaxis.set_major_locator(mticker.MaxNLocator(integer=True))

        plt.savefig('./img/' + experiment + '.png')
