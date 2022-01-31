import numpy as np
import matplotlib.pyplot as plt


def plot_running_avg(totalrewards):
    N = len(totalrewards)
    N = 7500
    running_avg = np.empty(N)
    for t in range(N):
      running_avg[t] = totalrewards[max(0, t-100):(t+1)].mean()
    plt.plot(running_avg)
    plt.title("Running Average")
    plt.xlabel("Epizody")
    plt.show()


buf_iters = np.load('plot_data/iters{}.npz'.format(7500))
iters = buf_iters['arr_0']

iters = iters.astype('int16')

plot_running_avg(iters)