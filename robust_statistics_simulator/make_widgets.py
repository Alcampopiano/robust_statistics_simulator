import ipywidgets as widgets
from ipywidgets import Output
import numpy as np
from scipy.stats import trim_mean

def make_population_widgets():

    output = Output()

    slider = widgets.FloatSlider(min=0, max=3, step=.01, value=1, description='Sigma',
                                       style=dict(description_width='100px'), layout=dict(width='300px'))

    dropdown = widgets.Dropdown(options=['normal', 'lognormal', 'contaminated chi-squared'], value='lognormal',
                                      description='Shape',
                                      style=dict(description_width='100px'), layout=dict(width='300px'))

    return {'output': output, 'slider': slider, 'dropdown': dropdown}

def make_sampling_distribution_widgets():

    output = Output()

    slider = widgets.IntSlider(min=1, max=300, step=1, value=30, description='Sample size')

    est_values = [('mean', {'func': np.mean}), ('trim_mean', {'func': trim_mean, 'args': .2}),
                  ('median', {'func': np.median}),
                  ('one-step', {'func': np.mean}), ('variance', {'func': np.var})]

    dropdown = widgets.Dropdown(options=est_values, value={'func': np.mean}, description='Estimator')

    #est_dropdown.layout={'margin': '50px 0px 10px 0px'}

    button = widgets.Button(description="run simulation")
    button.layout = {'width': 'auto'}

    label = widgets.Label(value="")

    label.layout = {'margin': '50px 0px 10px 0px'}

    return {'output': output, 'slider': slider, 'dropdown': dropdown, 'button': button, 'label': label}

def make_comparison_widgets():

    output = Output()
    button = widgets.Button(description="run comparisons")
    button.layout = {'width': 'auto'}

    return {'output': output, 'button': button}

def make_sampling_distribution_of_t_widgets():

    output = Output()

    slider = widgets.IntSlider(min=1, max=300, step=1, value=30, description='Sample size')

    button = widgets.Button(description="run simulation")
    button.layout = {'width': 'auto'}

    return {'output': output, 'slider': slider, 'button': button}

