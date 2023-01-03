# stdlib
import os
import sys
import re
import json

# plotter
from matplotlib import pyplot as plt

class Visualizer:
    def __init__(self, log_dir, fig_dir=None):
        self.log_dir = log_dir
        self.fig_dir = fig_dir
        if fig_dir is None:
            self.fig_dir = os.path.join(log_dir, 'fig')
        self.benchmark_raw_data = {}
        self.plot_data = {}
        self.ti_versions = []
        self.max_lines_per_fig = 18

        self.load_data_from_log_dir()
        self.restructure_data_for_plots()
        self.plot_figures()

    def load_data_from_log_dir(self):
        has_master = False
        for (_,_,files) in os.walk(self.log_dir, topdown=True):
            for log_fn in files:
                if re.search("benchmark_v.*\.log", log_fn):
                    re_match = re.search("v\d\.\d\.\d", log_fn)
                    # version_string = re.search("v\d\.\d\.\d", log_fn).group()
                    if not has_master and re_match is None:
                        has_master = log_fn == "benchmark_vmaster.log"
                        version_string = "master"
                    else:
                        version_string = re_match.group()
                        self.ti_versions.append(version_string)

                    print(f"Found benchmark log file for Taichi {version_string}")
                    log_file = open(os.path.join(self.log_dir, log_fn)) 
                    self.benchmark_raw_data[version_string] = json.load(log_file)
        self.ti_versions.sort()
        if has_master:
            self.ti_versions.append("master")
       

    def restructure_data_for_plots(self):
        def tags_to_str(tags):
            tag_str=tags['impl'] + '-' + tags['arch']
            if tags.get('variant'):
                tag_str += '-' + tags['variant']
            for key in tags:
                if key in ['impl', 'arch', 'variant']:
                    continue
                tag_str += '-'
                tag_str += key
                tag_str += '-'
                tag_str += str(tags[key])
            return tag_str
        for version in self.benchmark_raw_data:
            for record in self.benchmark_raw_data[version]:
                name = record['name']
                tags = record['tags']
                tags_str = tags_to_str(tags)
                value = record['value']
                if self.plot_data.get(name) is None:
                    self.plot_data[name] = {}
                if self.plot_data[name].get(tags_str) is None:
                    self.plot_data[name][tags_str] = {}
                self.plot_data[name][tags_str][version] = value
    
    def create_new_subplot(self, fig, n_charts, subplot_id, fig_name):
        subplot_id = n_charts * 100 + 10 + subplot_id
        ax = fig.add_subplot(subplot_id)
        ax.set_title(fig_name)
        ax.set_xticks(self.x_pos, self.ti_versions)
        return ax

    def plot_figures(self):
        os.makedirs(self.fig_dir, exist_ok=True)
        self.x_pos = [i + 1 for i in range(len(self.ti_versions))]

        for fig_name in self.plot_data:
            labels = []
            values = []
            nlines = len(self.plot_data[fig_name])
            n_charts = 1
            if nlines > self.max_lines_per_fig:
                n_charts = (nlines + self.max_lines_per_fig - 1) // self.max_lines_per_fig
            fig = plt.figure(figsize=(24, 6 * n_charts), dpi=80)
            subplot_id = 1
            ax = self.create_new_subplot(fig, n_charts, subplot_id, fig_name)
            line_id = 0
            for line_label in self.plot_data[fig_name]:
                if line_id >= self.max_lines_per_fig:
                    subplot_id += 1
                    labels = []
                    ax = self.create_new_subplot(fig, n_charts, subplot_id, fig_name)
                    line_id = 0
                labels.append(line_label)
                line_data = self.plot_data[fig_name][line_label]
                values = [line_data.get(ti_ver) for ti_ver in self.ti_versions]
                ax.plot(self.x_pos, values)
                ax.legend(labels)

                line_id += 1
            plt.show()
        # plt.plot()
        
if __name__ == '__main__':
    Visualizer(sys.argv[1])
 
