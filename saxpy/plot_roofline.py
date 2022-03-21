from matplotlib import pyplot as plt
from src.taichi.benchmark import benchmark as benchmark_taichi
import math
import os

def plot_nesting_factor_lines(ax, ylimit, nesting_factors=[]):
    for nf in nesting_factors:
        ax.vlines(x=[nf/6.0] , ymin=0, ymax=min(nf / 6.0  * 760, 29700), label="TEST", color='grey', linestyle='dashed')
        ax.text(nf / 6.0, 40, "Nest={}".format(nf), rotation=270, verticalalignment='top')

def plot_roofline(ax, scale='log'):
    bw = 760
    cc = 29700
    ridge = cc / 760.0
    membound_str = "DRAM bandwidth=760 GB/s"
    compbound_str = "FP32 peak={} GFLOPs".format(cc)
    if scale == 'log':
        xlimit = 1e4
        ylimit = cc * 2
        plot_nesting_factor_lines(ax, ylimit, [2**x for x in range(10)])
        ax.set_xscale('log', base=10)
        ax.set_yscale('log', base=10)
        ax.text(0.09, 0.95, membound_str, fontsize=12, rotation = 37, transform=ax.transAxes,
            verticalalignment='top')
        ax.text(0.55, 0.98, compbound_str, fontsize=12, transform=ax.transAxes,
            verticalalignment='top')
    elif scale == 'linear':
        xlimit = 100
        ylimit = cc * 1.2
        ax.text(0.05, 0.75, membound_str, fontsize=12, rotation = 57, transform=ax.transAxes,
            verticalalignment='top')
        ax.text(0.55, 0.88, compbound_str, fontsize=12, transform=ax.transAxes,
            verticalalignment='top')
    else:
        raise Exception("Unknown plot scale {}".format(scale))
    ax.set_xlim([2**-3, xlimit])
    ax.set_ylim([1, ylimit])
    ax.plot([0, ridge, xlimit], [0, cc, cc], color='black')
    ax.grid(True, "minor")
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.2)
    ax.set_xlabel("Arithmetic Intensity")
    ax.set_ylabel("Performance (GFLOPS)")
    ax.set_title("Taichi nested SAXPY roofline plot on 4096x4096 arrays")

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
        if rd["N"] != 4096:
            continue
        ax.plot([gflops / gbs], [gflops], marker="+", color=color, markersize=6, mew=2)

if __name__ == '__main__':
    try:
        os.makedirs('fig')
    except FileExistsError:
        pass
    taichi_results = benchmark_taichi(max_nesting=512)
    fig, ax = plt.subplots()
    #plot_roofline(ax, scale='linear')
    #plot_results(ax, taichi_results)
    #fig.savefig('fig/roofline_linear_scale.png',dpi=200)
    #fig.clf()
    #ax.cla()
    #fig, ax = plt.subplots()
    plot_roofline(ax, scale='log')
    plot_results(ax, taichi_results)
    fig.savefig('fig/roofline_log_scale.png',dpi=200)
