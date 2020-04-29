#!/usr/bin/env python3
import pandas as pd
import numpy as np
from tabulate import tabulate

from bokeh.io import curdoc
from bokeh.plotting import figure, output_file, show
import bokeh.palettes as palettes
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, Slider, Select, HoverTool, DataTable, TableColumn, Label

from copy import deepcopy
import itertools


def decompose(u):
    n = np.log2(u.shape[0])
    c = deepcopy(u)
    for j in np.arange(int(n - 1), 0 - 1, -1):
        c_next = deepcopy(c)
        for i in np.arange(0, int(2**j)):
            c_next[i] = (c[2 * i] + c[2 * i + 1]) / 2.0
            c_next[2**j + i] = (c[2 * i] - c[2 * i + 1]) / 2.0
        c = deepcopy(c_next)
    return c


def reconstruct(c):
    n = np.log2(c.shape[0])
    u = c
    for j in np.arange(0, int(n - 1) + 1):
        u_next = deepcopy(u)
        for i in np.arange(0, int(2**j)):
            u_next[2 * i] = u[i] + u[2**j + i]
            u_next[2 * i + 1] = u[i] - u[2**j + i]
        u = deepcopy(u_next)
    return u


j = 8
x = np.linspace(0, 2 * np.pi, 2**j)
y = np.sin(x)

haar_coef = decompose(y)
comp_haar_coef = deepcopy(haar_coef)
comp_haar_coef[np.abs(comp_haar_coef) < 0.0] = 0
comp_y = reconstruct(comp_haar_coef)

source = ColumnDataSource(
    data={
        'X': x,
        'Signal': y,
        'Haar': haar_coef,
        'Comp. Haar': comp_haar_coef,
        'Comp. Signal': comp_y
    })

plot = figure(title="1D Haar Wavelet Compression",
              tools="crosshair,pan,reset,save,wheel_zoom,,box_zoom,hover",
              x_axis_label='x',
              y_axis_label='y')
hover = plot.select(dict(type=HoverTool))
colors = itertools.cycle(palettes.Category10[10])
for key, color in zip(source.data, colors):
    if key == 'X':
        continue
    plot.line(x='X',
              y=key,
              source=source,
              legend_label=key,
              color=color,
              line_width=2)
plot.legend.click_policy = "hide"

start_slider = Slider(title="Start X", value=0.0, start=-100, end=100, step=1)
end_slider = Slider(title="End X",
                    value=2 * np.pi,
                    start=-100,
                    end=100,
                    step=1)
min_cutoff_slider = Slider(title="MinCutoff",
                           value=0.0,
                           start=0.0,
                           end=2 * np.max(y),
                           step=0.01)
max_cutoff_slider = Slider(title="MaxCutoff",
                           value=np.max(y) * 2,
                           start=0.0,
                           end=2 * np.max(y),
                           step=0.01)
start_reconstruct = Slider(title='Reconstruct Start',
                           value=0,
                           start=0,
                           end=2**j,
                           step=1)
stop_reconstruct = Slider(title='Reconstruct Stop',
                          value=2**j,
                          start=0,
                          end=2**j,
                          step=1)
signal_select = Select(
    title="Signal",
    value="sine",
    options=["sine", "sawtooth", "square", "triangle", "x^2"])
error_label = Label(x=0,
                    y=0,
                    x_units='screen',
                    y_units='screen',
                    text="Error: {:2.5f}".format(
                        np.square(np.subtract(y, comp_y)).mean()))
plot.add_layout(error_label)


def update_data(attrname, old, new):
    start = start_slider.value
    end = end_slider.value
    min_cutoff = min_cutoff_slider.value
    max_cutoff = max_cutoff_slider.value
    rec_start = start_reconstruct.value
    rec_stop = stop_reconstruct.value
    signal = signal_select.value
    x = np.linspace(start, end, 2**j)
    if signal == "sine":
        y = np.sin(x)
    elif signal == "sawtooth":
        y = x - np.floor(x)
    elif signal == "square":
        y = np.sign(np.sin(2 * np.pi * x))
    elif signal == "triangle":
        y = 4 * (x - 0.5 * np.floor(2 * x + 0.5)) * \
            np.power(-1, np.floor(2*x+0.5))
    elif signal == "x^2":
        y = x**2
    min_cutoff_slider.end = np.max(y)
    max_cutoff_slider.end = np.max(y)
    haar_coef = decompose(y)
    comp_haar_coef = deepcopy(haar_coef)
    comp_haar_coef[np.abs(comp_haar_coef) < min_cutoff] = 0
    comp_haar_coef[np.abs(comp_haar_coef) > max_cutoff] = 0
    comp_haar_coef[:rec_start] = 0
    comp_haar_coef[rec_stop:] = 0
    comp_y = reconstruct(comp_haar_coef)
    source.data = {
        'X': x,
        'Signal': y,
        'Haar': haar_coef,
        'Comp. Haar': comp_haar_coef,
        'Comp. Signal': comp_y
    }
    error_label.text = "Error: {:2.5f}".format(
        np.square(np.subtract(y, comp_y)).mean())


min_cutoff_slider.on_change('value', update_data)
max_cutoff_slider.on_change('value', update_data)
signal_select.on_change('value', update_data)
start_slider.on_change('value', update_data)
end_slider.on_change('value', update_data)
start_reconstruct.on_change('value', update_data)
stop_reconstruct.on_change('value', update_data)

data_table = DataTable(source=source,
                       columns=[
                           TableColumn(field="X", title="X"),
                           TableColumn(field="Signal", title="Signal"),
                           TableColumn(field="Haar", title='Haar'),
                           TableColumn(field='Comp. Haar', title="Comp. Haar"),
                           TableColumn(field="Comp. Signal",
                                       title="Comp. Signal")
                       ])
curdoc().add_root(
    column(plot, signal_select, row(min_cutoff_slider, max_cutoff_slider),
           row(start_reconstruct, stop_reconstruct),
           row(start_slider, end_slider)))
curdoc().title = "1D Haar Compression"
