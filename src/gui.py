import os
import random
import string

import PySimpleGUI as sg
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

matplotlib.use('TkAgg')
def draw_figure(canvas, figure):
    if not hasattr(draw_figure, 'canvas_packed'):
        draw_figure.canvas_packed = {}
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    widget = figure_canvas_agg.get_tk_widget()
    if widget not in draw_figure.canvas_packed:
        draw_figure.canvas_packed[widget] = figure
        widget.pack(side='top', fill='both', expand=1)
    return figure_canvas_agg


def delete_figure_agg(figure_agg):
    figure_agg.get_tk_widget().forget()
    try:
        draw_figure.canvas_packed.pop(figure_agg.get_tk_widget())
    except Exception as e:
        print(f'Error removing {figure_agg} from list', e)
    # plt.close('all')


def run_gui(epochs, figs):
    layout = [[sg.Text("Epoch:"), sg.Slider(range=(1,epochs), orientation='h', key='SLIDER', enable_events=True)],
              [sg.Canvas(key="-CANVAS-", size=(1500, 1000))]]

    window = sg.Window("MLP", layout, finalize=True)

    figure_agg = draw_figure(window['-CANVAS-'].TKCanvas, figs[0])
    while True:
        event, values = window.read()
        if event in (sg.WINDOW_CLOSED, "Cancel"):
            break
        if figure_agg:
            delete_figure_agg(figure_agg)

        figure_agg = draw_figure(window['-CANVAS-'].TKCanvas, figs[int(values["SLIDER"] - 1)])
    window.close()
    plt.close("all")