from pyvirtualdisplay import Display

Display(visible=0, size=(1400, 900)).start()

import imageio

from IPython.display import display
from IPython.display import Image

import pandas as pd
import plotly.graph_objects as go


def save_gif(frames, filename):
    imageio.mimsave(filename, frames, duration=1 / 35.)


def display_gif(filename):
    with open(filename, 'rb') as f:
        display(Image(data=f.read(), format='png'))


def visualize(solver_class, *args, **kwargs):
    fig = go.Figure()

    solver = solver_class()
    all_stats = pd.DataFrame()
    sample_count = 10
    for i in range(sample_count):
        solver = solver_class()
        print('attempt {}'.format(i + 1))
        stats = solver.train(*args, **kwargs)
        all_stats = all_stats.append(stats)
        fig.add_trace(go.Scatter(x=stats['Episode'],
                                 y=stats['Score'],
                                 mode='markers',
                                 name='attempt {}'.format(i + 1),
                                 opacity=0.5
                                 ))

    stats_mean = all_stats.groupby('Episode').mean()
    fig.add_trace(go.Scatter(x=stats_mean.index,
                             y=stats_mean['Score'],
                             mode='lines+markers',
                             name='Mean'
                             ))

    solver.display()
    fig.show()
    return stats_mean
