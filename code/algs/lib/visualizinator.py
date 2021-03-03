import numpy as np
from collections import deque
from itertools import product
from .optional_args import process_kwargs
from matplotlib.ticker import MultipleLocator
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})

# Visualize data by tracking it using labels that hold a
# list of all points (x,y) passed to it for graphing or to
# give back requested information about it as well
class Visualizinator:
    def __init__(self,
                 labels=('default'),
                 windowed_labels=None,
                 window_size=None,
                 **kwargs):
        self.enable_visual = False
        process_kwargs(self, kwargs, acceptable_kws=["enable_visual"])

        self.tracked_values = {}
 
        self.labels = labels
        for label in labels:
            self.tracked_values[label] = []

        self.windowed_values = {}
        if windowed_labels:
            assert (window_size != None)
            self.windowed_labels = windowed_labels
            self.window_size = window_size
            for windowed_label in windowed_labels:
                assert (windowed_label in labels)
                self.windowed_values[windowed_label] = deque([0] * window_size)

    # Get the current data for a given label
    def get(self, label):
        assert (label in self.tracked_values)
        return self.tracked_values[label]

    # Add values with the given labels in the passed dictionary
    # Should be in the form of:
    #   example.add({
    #       'example_label1': (x_value, y_value),
    #       'example_label2': (x_value, y_value)})
    # This allows for the x values to be different, but can have
    # all things graphed in the same plot potentially
    def add(self, label_values):
        if self.enable_visual:
            for label in label_values:
                assert (label in self.tracked_values)
                self.tracked_values[label].append(label_values[label])

    def addWindow(self, label_values, time, ticks):
        if self.enable_visual:
            for label, value in label_values.items():
                assert (label in self.windowed_values)
                window = self.windowed_values[label]
                window.popleft()
                window.append(value)
                self.tracked_values[label].append((time, np.mean(window), ticks))

    # Get the summation for a given label
    def sum(self, label, axis='y'):
        assert (axis == 'x' or axis == 'y')
        assert (label in self.tracked_values)
        x, y = self.tracked_values[label]
        if axis == 'y':
            return sum(y)
        else:
            return sum(x)

    # Use a passed in graph, creates a line graph using the data for the given
    # labels and colors using the given colors
    # NOTE: Default colors only goes up to 4, beyond that I'm not sure
    #       what will occur, but I believe an error would occur
    def visualize(self,
                  graph,
                  algorithms=[""],
                  labels=['default'],
                  xlabel=None,
                  ylabel=None,
                  show_xticks=True,
                  show_yticks=True,
                  colors=('k--', 'k-', 'rs', 'k^', 'g-'),
                  **kwargs):
        x_min, x_max = None, None
        y_min, y_max = None, None

        tmp_labels = []
        for label, alg in product(labels, algorithms):
            tmp_labels.append("{} {}".format(alg, label))

        for label in tmp_labels:
            assert (label in self.tracked_values)
            x, y, ticks = zip(*self.tracked_values[label])
            x_min = min(x_min, min(x)) if x_min else min(x)
            x_max = max(x_max, max(x)) if x_max else max(x)
            y_min = min(y_min, min(y)) if y_min else min(y)
            y_max = max(y_max, max(y)) if y_max else max(y)
            
        if xlabel:
            graph.set_xlabel(xlabel)
        if ylabel:
            graph.set_ylabel(ylabel)

        if not show_yticks:
            graph.set_yticks([])
        
        if not show_xticks:
            graph.set_xticks([])
        else:
            graph.xaxis.set_major_locator(plt.MaxNLocator(np.max(ticks) + 1))
            tick_labels = sorted(list(set(ticks)))
            tick_labels = [str(i) for i in tick_labels]
            graph.set_xticklabels(tick_labels)
        
        graph.set_xlim(x_min, x_max)
        for label, color in zip(tmp_labels, colors):
            print(label)
            x, y, _ = zip(*self.tracked_values[label])
            graph.plot(x, y, color, label=label.split(' ', 1)[0], linewidth=0.95)
            
        graph.legend(loc="upper left", prop={"size": "8"}, bbox_to_anchor=(1, 1))
