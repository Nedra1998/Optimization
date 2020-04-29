#!/usr/bin/env python3
import numpy as np
import matplotlib.image as mpimg

from bokeh.io import curdoc
from bokeh.plotting import figure, output_file, show
import bokeh.palettes as palettes
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, Slider, Select, HoverTool, DataTable, TableColumn, Label, Button

from copy import deepcopy
from glob import glob


def load_img(file):
    return mpimg.imread(file)


def bokeh_image(raw):
    scaled_raw = np.nan_to_num((raw - raw.min()) / ((raw.max() - raw.min()) if (raw.max() - raw.min()) != 0 else 1))
    img = np.empty((raw.shape[0], raw.shape[1]), dtype=np.uint32)
    view = img.view(dtype=np.uint8).reshape((raw.shape[0], raw.shape[1], 4))
    for a in range(raw.shape[0]):
        for b in range(raw.shape[1]):
            for c in range(raw.shape[2]):
                view[a, b, c] = 255 * np.clip(scaled_raw[a, b, c], 0.0, 1.0)
            view[a, b, 3] = 255
    return np.flip(img)


def decompose_vector(u):
    n = np.log2(u.shape[0])
    c = np.copy(u)
    for j in np.arange(int(n-1), 0 - 1, -1):
        c_next = np.copy(c)
        for i in np.arange(0, int(2**j)):
            c_next[i] = (c[2*i] + c[2*i + 1]) / 2.0
            c_next[2**j+i] = (c[2*i] - c[2*i + 1]) / 2.0
        c = np.copy(c_next)
    return c


def reconstruct_vector(c):
    n = np.log2(c.shape[0])
    u = c
    for j in np.arange(0, int(n-1)+1):
        u_next = np.copy(u)
        for i in np.arange(0, int(2**j)):
            u_next[2*i] = u[i] + u[2**j+i]
            u_next[2*i+1] = u[i] - u[2**j+i]
        u = np.copy(u_next)
    return u


def decompose(u):
    B = np.apply_along_axis(decompose_vector, 0, u)
    return np.apply_along_axis(decompose_vector, 1, B)


def reconstruct(c):
    B = np.apply_along_axis(reconstruct_vector, 1, c)
    return np.apply_along_axis(reconstruct_vector, 0, B)


plot = figure(title="2D Haar Wavelet Compression",
              tools="crosshair,pan,reset,save,wheel_zoom,,box_zoom,hover", x_axis_label='x', y_axis_label='y')

hover = plot.select(dict(type=HoverTool))
images = glob("*.png")
current_source = "Lenna-64.png"

y = load_img("./Lenna-64.png")
j = np.log2(y.shape[0])
haar_coef = decompose(y)
comp_haar_coef = np.copy(haar_coef)
comp_haar_coef[np.abs(comp_haar_coef) < 0.0] = 0
comp_y = reconstruct(comp_haar_coef)
plot.image_rgba(image=[bokeh_image(y), bokeh_image(haar_coef), bokeh_image(comp_haar_coef), bokeh_image(comp_y)], x=[0, 0, 1, 1], y=[
                0, -1, -1, 0], dw=[1, 1, 1, 1], dh=[1, 1, 1, 1])

error_label = Label(x=0, y=0, x_units='screen',
                    y_units='screen', text="Error: {:2.5f}".format((np.square(y - comp_y)).mean()))
plot.add_layout(error_label)


min_cutoff_slider = Slider(title="MinCutoff", value=0.0, start=0.0,
                           end=np.max(haar_coef), step=0.01)
max_cutoff_slider = Slider(title="MaxCutoff", value=np.max(haar_coef), start=0.0,
                           end=np.max(haar_coef), step=0.01)
start_reconstruct = Slider(title='Reconstruct Start',
                           value=0, start=0, end=2**j, step=1)
stop_reconstruct = Slider(title='Reconstruct Stop',
                          value=2**j, start=0, end=2**j, step=1)
recompute_button = Button(label="Recompute")
signal_select = Select(title="Source", value="Lenna-64.png", options=list(images))


def update_data():
    global current_source
    global y
    global j
    global haar_coef
    global comp_y
    global comp_haar_coef
    min_cutoff = min_cutoff_slider.value
    max_cutoff = max_cutoff_slider.value
    rec_start = start_reconstruct.value
    rec_stop = stop_reconstruct.value
    error_label.text = "Recomputing..."

    if signal_select.value != current_source:
        current_source = signal_select.value
        y = load_img(current_source)
        j = np.log2(y.shape[0])
        haar_coef = decompose(y)
        start_reconstruct.end = 2**j
        stop_reconstruct.end = 2**j

    min_cutoff_slider.end = np.max(haar_coef)
    max_cutoff_slider.end = np.max(haar_coef)

    comp_haar_coef = np.copy(haar_coef)
    comp_haar_coef[np.abs(comp_haar_coef) < min_cutoff] = 0
    comp_haar_coef[np.abs(comp_haar_coef) > max_cutoff] = 0
    comp_haar_coef[0:rec_start, 0:rec_start] = 0
    comp_haar_coef[int(rec_stop):, int(rec_stop):] = 0
    comp_y = reconstruct(comp_haar_coef)
    error_label.text = "Error: {:2.5f}".format((np.square(y - comp_y)).mean())
    plot.image_rgba(image=[bokeh_image(y), bokeh_image(haar_coef), bokeh_image(comp_haar_coef), bokeh_image(comp_y)], x=[0, 0, 1, 1], y=[
                    0, -1, -1, 0], dw=[1, 1, 1, 1], dh=[1, 1, 1, 1])


# start_reconstruct.on_change('value', update_data)
# stop_reconstruct.on_change('value', update_data)
# min_cutoff_slider.on_change('value', update_data)
# max_cutoff_slider.on_change('value', update_data)
recompute_button.on_click(update_data)

curdoc().add_root(column(plot,
                         row(signal_select, recompute_button),
                         row(min_cutoff_slider, max_cutoff_slider),
                         row(start_reconstruct, stop_reconstruct)))
curdoc().title = "2D Haar Compression"
