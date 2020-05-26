import pandas as pd
import altair as alt
import numpy as np
from IPython.display import clear_output, display
from ipywidgets import VBox, HBox
import ipywidgets
from scipy.stats import lognorm, norm, chi2, trim_mean, gaussian_kde
from robust_statistics_simulator.make_widgets import \
    make_population_widgets, make_sampling_distribution_widgets, make_comparison_widgets

population_widget_dict = make_population_widgets()
sampling_distribution_widgets=make_sampling_distribution_widgets()

def make_pdf(param, shape):
    
    if shape=='normal':

        x = np.linspace(norm.ppf(0.01, 0, param), norm.ppf(0.99, 0, param), 1000)
        y = norm.pdf(x)
        df = pd.DataFrame({'data': x, 'density': y})

    elif shape=='lognormal':

        x = np.linspace(lognorm.ppf(0.01, param), lognorm.ppf(0.99, param), 1000)
        y = lognorm.pdf(x, param)
        df = pd.DataFrame({'data': x, 'density': y})

    elif shape=='contaminated chi-squared':

        size=1000
        x = np.linspace(0, 13, size)
        chi_rand_values = chi2.rvs(4, size=size)
        contam_inds=np.random.randint(size, size=int(param*size))
        chi_rand_values[contam_inds] *= 10
        kernel=gaussian_kde(chi_rand_values)
        y=kernel.pdf(x)

        df = pd.DataFrame({'data': x, 'density': y})
        
    return df

def make_population_chart(df):

    c=alt.Chart(df).mark_line().encode(
        x=alt.X('data', axis=alt.Axis(titleFontSize=15)),
        y=alt.Y('density', axis=alt.Axis(titleFontSize=15))
    )
        
    return c.interactive()

def population():

    dropdown = population_widget_dict['dropdown']
    slider = population_widget_dict['slider']
    output=population_widget_dict['output']

    slider.observe(popluation_slider_callback, names='value')
    dropdown.observe(popluation_dropdown_callback, names='value') # dropdown.observe(update_population, names='value')

    with output:
        clear_output(wait=True)
        df = make_pdf(slider.value, dropdown.value)
        display(make_population_chart(df))


    # box_layout = widgets.Layout(display='flex',
    #                             flex_flow='row',
    #                             align_items='center',
    #                             )

    comps_vbox=VBox([dropdown, slider])
    display(HBox([comps_vbox, output]))

def popluation_dropdown_callback(widget_info):

    dropdown=population_widget_dict['dropdown']
    slider=population_widget_dict['slider']
    output=population_widget_dict['output']

    if dropdown.value=='contaminated chi-squared':
        slider.description='Contamination'
        slider.min=0
        slider.max = .5
        slider.value=.1

    else:
        slider.description = 'Sigma'
        slider.min=0
        slider.max = 3
        slider.value=1

    with output:
        clear_output(wait=True)
        df=make_pdf(slider.value, dropdown.value)
        display(make_population_chart(df))

def popluation_slider_callback(widget_info):

    dropdown = population_widget_dict['dropdown']
    slider = population_widget_dict['slider']
    output=population_widget_dict['output']

    with output:
        clear_output(wait=True)
        df=make_pdf(slider.value, dropdown.value)
        display(make_population_chart(df))

def sampling_distribution():

    dropdown = sampling_distribution_widgets['dropdown']
    slider = sampling_distribution_widgets['slider']
    output = sampling_distribution_widgets['output']
    button = sampling_distribution_widgets['button']
    label = sampling_distribution_widgets['label']

    button.on_click(sampling_distribution_button_callback)
    comps_vbox=VBox([dropdown, slider, button, label])
    display(HBox([comps_vbox, output]))

def sampling_distribution_button_callback(widget_info):

    population_dropdown = population_widget_dict['dropdown']
    sample_dropdown = sampling_distribution_widgets['dropdown']
    sample_slider = sampling_distribution_widgets['slider']
    population_slider = population_widget_dict['slider']
    output = sampling_distribution_widgets['output']
    label = sampling_distribution_widgets['label']

    with output:
        clear_output(wait=True)

        sample=[]
        for i in range(1000):
            data = generate_random_data_from_dist(population_slider.value, population_dropdown.value, sample_slider.value)
            est=sample_dropdown.value['func'](data) if not sample_dropdown.value.get('args') else \
                sample_dropdown.value['func'](data, sample_dropdown.value['args'])

            sample.append(est)

        display(make_sampling_distribution_chart(sample))
        label.value=f'SE = {np.std(sample, ddof=1).round(2)} based on the {population_dropdown.value} population'

def generate_random_data_from_dist(param, shape, size):


    if shape=='normal':
       data = norm.rvs(0, param, size=size)

    elif shape=='lognormal':
        data = lognorm.rvs(param, size=size)

    elif shape=='contaminated chi-squared':

        ### this should be the building of the whole contaminated population
        # then sample from it.
        # should this be outside of this function or is there a mathematical equivalent to doing it like this
        data = chi2.rvs(4, size=size)
        contam_inds=np.random.randint(size, size=int(param*size))
        data[contam_inds] *= 10


    return data

def make_sampling_distribution_chart(sample):

    df=pd.DataFrame({'data': sample})

    c=alt.Chart(df).transform_density('data', as_=['data', 'density']).mark_area().encode(
        x=alt.X('data', axis=alt.Axis(titleFontSize=15)),
        y=alt.Y('density:Q', axis=alt.Axis(titleFontSize=15))
    )

    return c.interactive()

