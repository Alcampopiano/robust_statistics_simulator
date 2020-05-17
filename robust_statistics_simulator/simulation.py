import pandas as pd
import altair as alt
import ipywidgets as widgets
import numpy as np
from IPython.display import clear_output, display
from ipywidgets import Output, VBox

sigma_slider = widgets.FloatSlider(min=0, max=3, step=.01, value=1,  description='Sigma') 
shape_dropdown=widgets.Dropdown(options=['normal', 'log', 'g-and-h'], value='normal', description='Shape')
shape_dropdown.layout={'margin': '50px 0px 10px 0px'}

sample_slider = widgets.IntSlider(min=1, max=300, step=1, value=30,  description='sample size')
est_values=[('mean', np.mean), ('trim_mean', np.mean), ('median', np.median),
            ('one-step', np.mean), ('variance', np.var)]

est_dropdown = widgets.Dropdown(options=est_values, value=np.mean, description='Estimator')
est_dropdown.layout={'margin': '50px 0px 10px 0px'}
go_button = widgets.Button(description="run simulation")

def make_distribution(sig, shape):
    
    if shape=='normal':
        data=np.random.normal(0, sig, 1000)
        
    elif shape=='log':
        data = np.random.lognormal(0, sig, 1000)
        
    return data

def make_population_chart(data):
    
    df=pd.DataFrame(data, columns=['data'])
    
    c=alt.Chart(df).transform_density('data', as_=['data', 'density']).mark_area().encode(
        x="data:Q",
        y='density:Q',
    )
        
    return c

def initialize_population_visualization():
    
    output_widget = Output(layout={'width': '1000px', 'height': '300px'})
        
    def update_display(change):
    
        sigma_value=sigma_slider.value
        population_shape=shape_dropdown.value

        with output_widget: 
            clear_output(wait=True)
            data=make_distribution(sigma_value, population_shape)
            display(make_population_chart(data))
    
    sigma_slider.observe(update_display, names='value')
    shape_dropdown.observe(update_display, names='value')

    update_display(None)
    display(VBox([output_widget, shape_dropdown, sigma_slider]))

def initialize_sampling_distribution_visualization():

    output_widget = Output()

    def on_button_clicked(change):

        output_widget.layout = {'width': '1000px', 'height': '300px'}

        with output_widget:
            clear_output(wait=True)
            data = make_distribution(sigma_slider.value, shape_dropdown.value)

            samp=[]
            for i in range(1000):
                rand_samp=np.random.choice(data, size=sample_slider.value)
                est=est_dropdown.value(rand_samp)
                samp.append(est)

            print('make new charting func? How to display SE?')
            display(make_population_chart(samp))


    go_button.on_click(on_button_clicked)
    display(VBox([output_widget, est_dropdown, sample_slider, go_button]))

def population():
    initialize_population_visualization()

def sampling_distribution():
    initialize_sampling_distribution_visualization()

