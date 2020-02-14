from scipy import stats
import inspect
import itertools
from matplotlib.ticker import FormatStrFormatter
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib
from numpy import *
matplotlib.use('GTK3Agg')

sns.set_context("paper")


class Function(object):

    def __init__(self, func):
        self.func = func
        self.args = inspect.getfullargspec(func)[0]
        self.dims = len(self.args)
        self.defaults = func.__defaults__ if func.__defaults__ else ()
        self.name = func.__name__

    def __call__(self, *args, **kwargs):
        if len(args) > len(self.args):
            raise TypeError("{}() takes {} arguments but {} were given".format(
                self.func.__name__, len(self.args), len(args)))
        params = {self.args[i]: val for i, val in enumerate(args)}
        for i, val in enumerate(self.defaults):
            if self.args[-i - 1] not in params.keys():
                params[self.args[-i - 1]] = val
        for arg in self.args:
            if arg not in params:
                raise TypeError(
                    "{}() missing 1 required positional argument: '{}'".format(
                        self.func.__name__, arg))
        return self.func(**params)

    def sample(self, *args, samples=50, **kwargs):
        ranges = {self.args[i]: val for i, val in enumerate(args)}
        for key, val in kwargs.items():
            ranges[key] = val
        for i, val in enumerate(self.defaults):
            if self.args[-i - 1] not in ranges.keys():
                ranges[self.args[-i - 1]] = (val, val)
        for arg in self.args:
            if arg not in ranges:
                raise TypeError(
                    "{}() missing 1 required positional argument: '{}'".format(
                        self.func.__name__, arg))
        for key, arg in ranges.items():
            if len(arg) == 1:
                ranges[key] = np.linspace(0, arg[0], samples)
            elif len(arg) == 2:
                ranges[key] = np.linspace(arg[0], arg[1], samples)
            else:
                ranges[key] = np.linspace(arg[0], arg[1], arg[2])
        samples = list(
            itertools.product(*ranges.values()))
        data = []
        for sample in samples:
            data.append((*sample,
                         self.func(*sample)))
        return data


def func(function):
    return Function(function)


class Plot(object):

    def __init__(self, *funcs, steps=100, **kwargs):
        self.funcs = funcs
        self.steps = steps
        self.ranges = {}
        for key, val in kwargs.items():
            self.ranges[key] = val

    def generate_dataframe(self, steps=None, **kwargs):
        if steps is None:
            steps = self.steps
        print(self.ranges.values())
        if len(self.funcs) != 1:
            return pd.merge(*[pd.DataFrame(x.sample(*self.ranges.values(), samples=steps), columns=[*x.args, x.name]) for x in self.funcs], on=self.funcs[0].args[0])
        x = self.funcs[0]
        return pd.DataFrame(x.sample(*self.ranges.values(), samples=steps), columns=[*x.args, x.name])

    def show_1d(self, **kwargs):
        df = self.generate_dataframe(**kwargs)
        print(df)
        ax = sns.lineplot(
            x=self.funcs[0].args[0], y="haar",
            data=df)
        major_formatter = FormatStrFormatter('%.2f')
        ax.xaxis.set_major_formatter(major_formatter)
        ax.yaxis.set_major_formatter(major_formatter)

    def show_2d(self, **kwargs):
        df = self.generate_dataframe(**kwargs)
        ax = sns.heatmap(data=df.pivot("x", "y", "val"), cmap='viridis')
        major_formatter = FormatStrFormatter('%.2f')
        ax.xaxis.set_major_formatter(major_formatter)
        ax.yaxis.set_major_formatter(major_formatter)

    def show(self, **kwargs):
        if self.funcs[0].dims == 1:
            self.show_1d(**kwargs)
        elif self.funcs[0].dims == 2:
            self.show_2d(**kwargs)
        plt.show()

    def save(self, file):
        pass


@func
def haar(x):
    if 0 <= x < 0.5:
        return 1
    elif 0.5 <= x < 1:
        return -1
    return 0


@func
def meyer(x):
    phi1 = (4/(3*pi)*(x-0.5)*cos(2*pi/3*(x-0.5))-1/pi *
            sin(4*pi/3*(x-0.5)))/((x-0.5)-16/9*(x-0.5)**3)
    phi2 = (8/(3*pi)*(x-0.5)*cos(8*pi/3*(x-0.5))-1/pi *
            sin(4*pi/3*(x-0.5)))/((x-0.5)-64/9*(x-0.5)**3)
    return phi1 + phi2


Plot(haar, meyer, x=(-5, 5)).show()

# mean, cov = [0, 1], [(1, .5), (.5, 1)]
# data = np.random.multivariate_normal(mean, cov, 200)
# df = pd.DataFrame(data, columns=["x", "y"])
#

#
# sns.jointplot(x="x", y="y", data=df, kind="hex", space=0)
# plt.show()
