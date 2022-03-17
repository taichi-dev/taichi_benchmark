from matplotlib import pyplot as plt
from benchmark_taichi import benchmark_taichi
import math
import os

def plot_roofline_log_scale(ax):
    bw = 760
    cc = 29700
    ridge = cc / 760.0
    xlimit = 1e4
    ax.plot([0, ridge, xlimit], [0, cc, cc], color='black')
    ax.set_xscale('log', base=10)
    ax.set_yscale('log', base=10)
    ax.set_xlim([2**-3, xlimit])
    ax.set_ylim([1, cc * 2])
    ax.grid(True, "minor")
    membound_str = "DRAM bandwidth=760 GB/s"
    compbound_str = "FP32 peak={} GFLOPs".format(cc)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.2)
    ax.text(0.09, 0.95, membound_str, fontsize=12, rotation = 37, transform=ax.transAxes,
            verticalalignment='top')
    ax.text(0.55, 0.98, compbound_str, fontsize=12, transform=ax.transAxes,
            verticalalignment='top')

def plot_roofline_linear_scale(ax):
    bw = 760
    cc = 29700
    ridge = cc / 760.0
    xlimit = 100
    ax.plot([0, ridge, xlimit], [0, cc, cc], color='black')
    #ax.set_xscale('log', base=10)
    #ax.set_yscale('log', base=10)
    ax.set_xlim([2**-3, xlimit])
    ax.set_ylim([1, cc * 1.2])
    ax.grid(True, "minor")
    membound_str = "DRAM bandwidth=760 GB/s"
    compbound_str = "FP32 peak={} GFLOPs".format(cc)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.2)
    ax.text(0.05, 0.75, membound_str, fontsize=12, rotation = 57, transform=ax.transAxes,
            verticalalignment='top')
    ax.text(0.55, 0.88, compbound_str, fontsize=12, transform=ax.transAxes,
            verticalalignment='top')

def get_arithmetic_intensity(flops, bw):
    return flops / bw

def get_color_marker(N):
    color_dict = {256:'gold', 512:'darkorange', 1024:'green', 2048:'blue', 4096:'red', 8192:'darkgrey',}
    return color_dict[N]
def plot_results(ax, results):
    x = []
    y = []
    for rd in results:
        gflops = rd["gflops"]
        gbs = rd["gbs"]
        x.append(gflops / gbs)
        y.append(gflops)
        color = get_color_marker(rd["N"])
        ax.plot([gflops / gbs], [gflops], marker="+", color=color, markersize=6, mew=2)

if __name__ == '__main__':
    try:
        os.makedirs('fig')
    except FileExistsError:
        pass
    taichi_results = benchmark_taichi(max_nesting=512)
    fig, ax = plt.subplots()
    plot_roofline_linear_scale(ax)
    plot_results(ax, taichi_results)
    fig.savefig('fig/roofline_linear_scale.png',dpi=200)
    fig.clf()
    ax.cla()
    fig, ax = plt.subplots()
    plot_roofline_log_scale(ax)
    plot_results(ax, taichi_results)
    fig.savefig('fig/roofline_log_scale.png',dpi=200)
