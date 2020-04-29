#!/usr/bin/env python3
import numpy as np
import time

from bokeh.io import curdoc, show
from bokeh.plotting import figure, show
import bokeh.palettes as palettes
from bokeh.layouts import column, row
from bokeh.layouts import gridplot
from bokeh.models import Slider, Select, HoverTool, Label

from opensimplex import OpenSimplex


def normalize(v):
    return v / np.sqrt(np.sum(v**2))


PLOTS = []


def plot_img(source, title, x, y, palette="Viridis256", plt=None):
    if plt is None:
        PLOTS.append(
            figure(title=title,
                   plot_width=400,
                   plot_height=400,
                   tooltips=[("x", "$x"), ("y", "$y"), ("value", "@image")],
                   tools="crosshair,pan,reset,save,wheel_zoom,box_zoom,hover"))
        plt = PLOTS[-1]
    plt.image(image=[source],
              x=x[0],
              y=y[0],
              dw=x[1] - x[0],
              dh=y[1] - y[0],
              palette=palette)
    return plt


def show_plots(control=None, width=700, height=700):
    nplots = len(PLOTS)
    nsquare = 0
    while (nsquare + 1)**2 < nplots:
        nsquare += 1
    nsquare = (nsquare + 1)
    plot_width = int(width / nsquare)
    plot_height = int(height / nsquare)
    rows = []
    for i, plt in enumerate(PLOTS):
        plt.plot_height = plot_height
        plt.plot_width = plot_width
        if i % nsquare == 0:
            rows.append([plt])
        else:
            rows[-1].append(plt)
    grid = gridplot(rows)
    if control:
        curdoc().add_root(column(grid, control))
    else:
        curdoc().add_root(column(grid))


def rasterize(x0, y0, x1, y1, limit_min=-np.inf, limit_max=np.inf):
    points = set()
    dx = np.abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -np.abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    started = False
    while True:
        if limit_min < int(x0) < limit_max and limit_min < int(y0) < limit_max:
            points.add((int(x0), int(y0)))
            started = True
        elif started == True:
            break
        if np.abs(x0 - x1) < 1.0 and np.abs(y0 - y1) < 1.0:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
    return list(points)


def cast_ray(offset, angle, radius, origin=[0, 0]):
    origin = np.array(origin)
    pt = np.array([np.cos(angle) * radius, np.sin(angle) * radius])
    tan = normalize(np.array([-pt[1], pt[0]]))
    res = [*(pt + offset * tan + origin), *(offset * tan - pt + origin)]
    return res


def sum_pixels(source, pixels):
    res = np.sum([source[x] for x in pixels])
    return res


def decompose_vector(u):
    n = np.log2(u.shape[0])
    c = np.copy(u)
    for j in np.arange(int(n - 1), 0 - 1, -1):
        c_next = np.copy(c)
        for i in np.arange(0, int(2**j)):
            c_next[i] = (c[2 * i] + c[2 * i + 1]) / 2.0
            c_next[2**j + i] = (c[2 * i] - c[2 * i + 1]) / 2.0
        c = np.copy(c_next)
    return c


def reconstruct_vector(c):
    n = np.log2(c.shape[0])
    u = c
    for j in np.arange(0, int(n - 1) + 1):
        u_next = np.copy(u)
        for i in np.arange(0, int(2**j)):
            u_next[2 * i] = u[i] + u[2**j + i]
            u_next[2 * i + 1] = u[i] - u[2**j + i]
        u = np.copy(u_next)
    return u


def decompose(u):
    B = np.apply_along_axis(decompose_vector, 0, u)
    return np.apply_along_axis(decompose_vector, 1, B)


def reconstruct(c):
    B = np.apply_along_axis(reconstruct_vector, 1, c)
    return np.apply_along_axis(reconstruct_vector, 0, B)


def haar_filter(v, min_v=0.0):
    width = v.shape[0]
    height = v.shape[1]
    size = max(np.ceil(np.log2(width)), np.ceil(np.log2(height)))
    pad_width = 2**size - v.shape[0]
    pad_height = 2**size - v.shape[1]
    v = np.pad(v, [(0, int(pad_width)), (0, int(pad_height))])
    v = decompose(v)
    v[np.abs(v) < np.max(v) * min_v] = 0
    v = reconstruct(v)
    return v[:width, :height]


def radon_transform(img, nangle=50, nray=50):
    l2 = np.sqrt(2) * np.max(img.shape)
    trans = lambda x, y: sum_pixels(
        img,
        rasterize(*cast_ray(
            x, y, l2 / 2, origin=[img.shape[0] / 2, img.shape[1] / 2]),
                  limit_min=0,
                  limit_max=np.min(img.shape)))
    xx, yy = np.meshgrid(np.linspace(-l2 / 2, l2 / 2, nray),
                         np.linspace(0, np.pi, nangle),
                         sparse=True)
    return np.vectorize(trans)(xx, yy)


def inverse_radon_transform(shape, sinograph):
    l2 = np.sqrt(2) * np.max(shape)
    out = np.zeros(shape)
    for i, angle in enumerate(np.linspace(0, np.pi, sinograph.shape[0])):
        for j, offset in enumerate(
                np.linspace(-l2 / 2, l2 / 2, sinograph.shape[1])):
            pixels = rasterize(*cast_ray(
                offset, angle, l2 / 2, origin=[shape[0] / 2, shape[1] / 2]),
                               limit_min=0,
                               limit_max=np.min(shape))
            for px in pixels:
                out[px] += sinograph[i][j]
    return out / sinograph.shape[0]


def gen_img_from_func(func, x_range, y_range, num=50):
    xx, yy = np.meshgrid(np.linspace(x_range[0], x_range[1], num),
                         np.linspace(y_range[0], y_range[1], num),
                         sparse=True)
    return np.vectorize(func)(xx, yy)


def indicator_func(x, y):
    if x < 0 and y < 1 and x > -1 and y > 0:
        return 1
    elif x > 0 and y > -1 and x < 1 and y < 0:
        return 1
    return 0


def mexican_hat(x, y):
    r = np.sqrt(x**2 + y**2)
    return np.sin(2 * r) / r


simplex_noise = OpenSimplex()
func = indicator_func

x = (-2, 2)
y = (-2, 2)
z = gen_img_from_func(func, x, y)

sinograph = radon_transform(z, nangle=32, nray=32)
filtered_sinograph = haar_filter(sinograph, 0.25)
backprojection = inverse_radon_transform(z.shape, sinograph)
filtered_backprojection = inverse_radon_transform(z.shape, filtered_sinograph)

x_range_min = Slider(title="XMin", value=-2.0, start=-100, end=100, step=1)
x_range_max = Slider(title="XMax", value=2.0, start=-100, end=100, step=1)
y_range_min = Slider(title="YMin", value=-2.0, start=-100, end=100, step=1)
y_range_max = Slider(title="YMax", value=2.0, start=-100, end=100, step=1)

ntheta_slider = Slider(title="NTheta",
                       value=32,
                       start=1,
                       end=256,
                       step=1,
                       callback_policy="throttle",
                       callback_throttle=1000)
nray_slider = Slider(title="NRays",
                     value=32,
                     start=1,
                     end=256,
                     step=1,
                     callback_policy="throttle",
                     callback_throttle=1000)

filter_slider = Slider(title="Filter",
                       value=0.0,
                       start=0.0,
                       end=1.0,
                       step=0.01,
                       callback_policy="throttle",
                       callback_throttle=1000)

src_plt = plot_img(z, "Source", x, y)
sin_plt = plot_img(sinograph, "Sinograph", x, (0, np.pi))
fsin_plt = plot_img(filtered_sinograph, "Filtered Sinograph", x, (0, np.pi))
bak_plt = plot_img(backprojection, "BackProjection", x, y)
fbak_plt = plot_img(filtered_backprojection, "Filtered BackProjection", x, y)

error_bak = Label(x=0,
                  y=0,
                  x_units='screen',
                  y_units='screen',
                  text="Error: {:2.5f}".format(
                      (np.square(z - backprojection)).mean()))
error_fbak = Label(x=0,
                   y=0,
                   x_units='screen',
                   y_units='screen',
                   text="Error: {:2.5f}".format(
                       (np.square(z - filtered_backprojection)).mean()))

bak_plt.add_layout(error_bak)
fbak_plt.add_layout(error_fbak)

state = [ntheta_slider.value, nray_slider.value, filter_slider.value]


def update_plots(attr, old, new):
    global state
    global sinograph
    global z
    global filtered_sinograph
    global x, y
    print("==> Processing: filter: {}, radial: {}, rays: {}".format(
        filter_slider.value, ntheta_slider.value, nray_slider.value))
    # x = (x_range_min.value, x_range_max.value)
    # y = (y_range_min.value, y_range_max.value)
    # z = gen_img_from_func(func, x, y)
    # plot_img(z, "Source", x, y, plt=src_plt)
    if ntheta_slider.value != state[0] or nray_slider.value != state[1]:
        state = [ntheta_slider.value, nray_slider.value, -1]
        sinograph = radon_transform(z,
                                    nangle=ntheta_slider.value,
                                    nray=nray_slider.value)
        plot_img(sinograph, "Sinograph", x, (0, np.pi), plt=sin_plt)
        backprojection = inverse_radon_transform(z.shape, sinograph)
        plot_img(backprojection, "BackProjection", x, y, plt=bak_plt)
        error_bak.text = "Error: {:2.5f}".format(
            (np.square(z - backprojection)).mean())
    if filter_slider.value != state[2]:
        state[2] = filter_slider.value
        filtered_sinograph = haar_filter(sinograph, filter_slider.value)
        filtered_backprojection = inverse_radon_transform(z.shape,
                                                          filtered_sinograph)
        plot_img(filtered_sinograph,
                 "Filtered Sinograph",
                 x, (0, np.pi),
                 plt=fsin_plt)
        plot_img(filtered_backprojection,
                 "Filtered BackProjection",
                 x,
                 y,
                 plt=fbak_plt)
        error_fbak.text = "Error: {:2.5f}".format(
            (np.square(z - filtered_backprojection)).mean())
    print("==> Done")


x_range_min.on_change('value_throttled', update_plots)
x_range_max.on_change('value_throttled', update_plots)
y_range_min.on_change('value_throttled', update_plots)
y_range_max.on_change('value_throttled', update_plots)
ntheta_slider.on_change('value_throttled', update_plots)
nray_slider.on_change('value_throttled', update_plots)
filter_slider.on_change('value_throttled', update_plots)

show_plots(control=column(row(x_range_min, x_range_max),
                          row(y_range_min, y_range_max),
                          row(ntheta_slider, nray_slider), row(filter_slider)))
