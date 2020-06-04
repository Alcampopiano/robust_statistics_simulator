import pandas as pd
import altair as alt
import numpy as np
from IPython.display import clear_output, display
from ipywidgets import VBox, HBox
import ipywidgets
from scipy.stats.mstats import winsorize
from scipy.stats import lognorm, norm, chi2, trim_mean, gaussian_kde, t
from robust_statistics_simulator.make_widgets import \
    make_population_widgets, make_sampling_distribution_widgets, \
    make_comparison_widgets, make_sampling_distribution_of_t_widgets, \
    make_type_I_error_widgets

population_widget_dict = make_population_widgets()
sampling_distribution_widgets=make_sampling_distribution_widgets()
comparison_widgets=make_comparison_widgets()
t_sampling_distribution_widgets=make_sampling_distribution_of_t_widgets()
type_I_error_widgets=make_type_I_error_widgets()

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

    elif shape=='t':

        x = np.linspace(t.ppf(0.01, param), t.ppf(0.99, param), 1000)
        y = t.pdf(x, param)
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

def comparisons():

    output = comparison_widgets['output']
    button = comparison_widgets['button']

    button.on_click(comparison_button_callback)
    #comps_vbox = VBox([button, output])
    #display(comps_vbox)
    #comps_vbox = VBox([dropdown, slider, button, label])
    display(HBox([button, output]))

def comparison_button_callback(widget_info):

    output = comparison_widgets['output']

    sample_size=30
    dists=['normal', 'lognormal', 'contaminated chi-squared']
    estimators=[({'name': 'mean', 'func': np.mean}),
                ({'name': 'trim_mean', 'func': trim_mean, 'args': .2}),
                ({'name': 'median', 'func': np.median}),
                ({'name': 'one-step', 'func': np.mean}),
                ] # ({'name': 'variance', 'func': np.var})

    params=[1,1,.1]
    results=[]

    with output:
        clear_output(wait=True)

        for param, dist in zip(params, dists):
            for est in estimators:
                sample = []
                for i in range(1000):
                    data = generate_random_data_from_dist(param, dist, sample_size)
                    est_val=est['func'](data) if not est.get('args') else \
                        est['func'](data, est['args'])

                    sample.append(est_val)

                results.append({'dist': dist, 'est': est['name'], 'se': np.std(sample, ddof=1)})

        display(make_comparison_chart(results))

def make_comparison_chart(results):

    df = pd.DataFrame(results)

    c=alt.Chart(df).mark_bar(tooltip=True, size=30).encode(
        y=alt.Y('est', title='Estimator', axis=alt.Axis(titleFontSize=15, labelFontSize=12)),
        x=alt.X('se', title='Standard error', axis=alt.Axis(titleFontSize=15, labelFontSize=12)),
        color=alt.Color('dist', title='Population shape', legend=alt.Legend(labelFontSize=12, titleFontSize=15))
    ).properties(height=300)


    return c.interactive()

def t_sampling_distribution():

    slider = t_sampling_distribution_widgets['slider']
    output = t_sampling_distribution_widgets['output']
    button = t_sampling_distribution_widgets['button']

    button.on_click(t_sampling_distribution_button_callback)
    comps_vbox = VBox([slider, button])
    display(HBox([comps_vbox, output]))

def t_sampling_distribution_button_callback(widget_info):

    population_dropdown = population_widget_dict['dropdown']
    sample_slider = t_sampling_distribution_widgets['slider']
    population_slider = population_widget_dict['slider']
    output = t_sampling_distribution_widgets['output']

    mu=get_population_average_estimate(population_slider.value,
                population_dropdown.value)

    with output:
        clear_output(wait=True)

        sample=[]
        for i in range(1000):
            data = generate_random_data_from_dist(population_slider.value,
                        population_dropdown.value, sample_slider.value)

            est=(np.sqrt(sample_slider.value)*(np.mean(data)-mu))/np.std(data, ddof=1)
            sample.append(est)

        display(make_sampling_distribution_of_t_chart(sample))
        #print(sample[:20])

def make_sampling_distribution_of_t_chart(sample):

    sample_slider = t_sampling_distribution_widgets['slider']
    freedom=sample_slider.value-1
    df_assumed = make_pdf(freedom, 't')
    df_actual=pd.DataFrame({'actual': sample})

    actual=alt.Chart(df_actual).transform_density('actual',
            as_=['actual', 'density']).mark_line().encode(
        x=alt.X('actual', title='T',
                axis=alt.Axis(titleFontSize=15)),
        y=alt.Y('density:Q', axis=alt.Axis(titleFontSize=15)),
    )

    assumed=alt.Chart(df_assumed).mark_line(color='red').encode(
        x=alt.X('data', title='T',
                axis=alt.Axis(titleFontSize=15)),
        y=alt.Y('density', axis=alt.Axis(titleFontSize=15)),
    )

    c=alt.layer(actual, assumed)

    return c.interactive()

def get_population_average_estimate(param, shape):

    size=100000

    if shape=='normal':
        mu=0

    elif shape=='lognormal':
        mu = lognorm.stats(param, moments='m')

    elif shape=='contaminated chi-squared':

        ### this should be the building of the whole contaminated population
        # then sample from it.
        # should this be outside of this function or is there a mathematical equivalent to doing it like this
        data = chi2.rvs(4, size=size)
        contam_inds=np.random.randint(size, size=int(param*size))
        data[contam_inds] *= 10
        mu=np.mean(data)

    return mu

def type_I_error():

    slider = type_I_error_widgets['slider']
    output = type_I_error_widgets['output']
    button = type_I_error_widgets['button']

    button.on_click(type_I_error_button_callback)
    comps_vbox = VBox([slider, button])
    display(HBox([comps_vbox, output]))

def type_I_error_button_callback(widget_info):

    dropdown = population_widget_dict['dropdown']
    output = type_I_error_widgets['output']
    dists = dropdown.options
    params = [1, 1, .1]

    results = []
    with output:
        clear_output(wait=True)

        for param, dist in zip(params, dists):
            error_rate=simulate_t_type_I_error(param, dist)
            results.append({'dist': dist, 'error': error_rate, 'test': 't'})

        for param, dist in zip(params, dists):
            error_rate = simulate_tt_type_I_error(param, dist)
            results.append({'dist': dist, 'error': error_rate, 'test': 'tt'})

        for param, dist in zip(params, dists):
            error_rate = simulate_pb_type_I_error(param, dist)
            results.append({'dist': dist, 'error': error_rate, 'test': 'pb'})

        display(make_type_I_error_chart(results))
        #print(results)

def simulate_t_type_I_error(param, dist):

    nsamples=10000
    mu = get_population_average_estimate(param, dist)
    slider = type_I_error_widgets['slider']

    tvals=[]
    for _ in range(nsamples):

        data = generate_random_data_from_dist(param, dist, slider.value)
        tval = (np.sqrt(slider.value) * (np.mean(data) - mu)) / np.std(data, ddof=1)
        tvals.append(tval)

    t_crit = t.ppf(.975, slider.value - 1)
    prob = (np.sum(tvals < -t_crit) + np.sum(tvals > t_crit)) / len(tvals)

    return prob

def simulate_pb_type_I_error(param, dist):

    slider = type_I_error_widgets['slider']
    nboot = 599
    bools = []
    mu = get_trimmed_mu_estimate(param, dist)

    for _ in range(1000):
        effects = []
        data = generate_random_data_from_dist(param, dist, slider.value)

        for _ in range(nboot):
            bdat = np.random.choice(data, slider.value)
            effects.append(trim_mean(bdat, .2) - mu)

        l = round(.05 * nboot / 2) - 1
        u = nboot - l - 2
        up = sorted(effects)[u]
        low = sorted(effects)[l]
        bools.append((low < 0 < up))

    prob = 1 - (np.sum(bools) / len(bools))

    return prob

def simulate_tt_type_I_error(param, dist):

    nsamples=10000
    crit_nsamples=100000
    n = type_I_error_widgets['slider'].value
    g=int(.2*n)
    df = n - 2 * g - 1
    ts=np.sort(t.rvs(df, size=crit_nsamples))
    l = round(.05 * crit_nsamples / 2) - 1
    u = crit_nsamples - l - 2
    up = ts[u]
    low = ts[l]

    mu = get_trimmed_mu_estimate(param, dist)

    bools=[]
    for _ in range(nsamples):

        data = generate_random_data_from_dist(param, dist, n)
        t_stat=(1-2*.2)*np.sqrt(n)*(trim_mean(data, .2) - mu) / np.sqrt(winvar(data))
        bools.append((t_stat < low or t_stat > up))


    prob = np.sum(bools) / len(bools)

    return prob

def winvar(x, tr=.2):
    """
    Compute the gamma Winsorized variance for the data in the vector x.
    tr is the amount of Winsorization which defaults to .2.
    Nan values are removed.

    :param x:
    :param tr:
    :return:
    """

    y=winsorize(x, limits=(tr,tr))
    wv = np.var(y, ddof=1)

    return wv

def get_trimmed_mu_estimate(param, shape):

    size=100000

    if shape == 'normal':
        mu=0

    elif shape=='lognormal':
        mu = trim_mean(lognorm.rvs(param, size=size), .2)

    elif shape=='contaminated chi-squared':

        data = chi2.rvs(4, size=size)
        contam_inds=np.random.randint(size, size=int(param*size))
        data[contam_inds] *= 10
        mu=trim_mean(data, .2)

    return mu

def make_type_I_error_chart(results):

    df = pd.DataFrame(results)
    df['test'] = df['test'].replace({'t': 't-test', 'pb': 'trimmed percentile bootstrap', 'tt': 'trimmed t-test'})

    bars=alt.Chart().mark_bar(tooltip=True, size=30).encode(
        y=alt.Y('test', sort=['-x'], title='Type of test', axis=alt.Axis(titleFontSize=15, labelFontSize=12)),
        x=alt.X('error', title='Estimated type I error', axis=alt.Axis(titleFontSize=15, labelFontSize=12)),
        color=alt.Color('dist', title='Population shape', legend=alt.Legend(labelFontSize=12, titleFontSize=15))
    )

    text = alt.Chart().mark_text(dx=-15, dy=3).encode(
        y=alt.Y('test', sort=['-x']),
        x=alt.X('error', stack='zero'),
        text=alt.Text('error', format='.3f')
    )


    return alt.layer(bars,text, data=df).properties(height=300)
