#import numba
import pandas as pd
import altair as alt
import numpy as np
import streamlit as st
from scipy.stats.mstats import winsorize
from scipy.stats import expon, lognorm, norm, chi2, trim_mean, gaussian_kde, t
from scipy.integrate import quad

dists=['normal', 'lognormal', 'contaminated chi-squared', 't', 'exponential', 'contaminated normal']
est_dict = {'mean': np.mean,
              'trim_mean': {'func': trim_mean, 'args': .2},
              'median': np.median,
              'one-step': np.mean,
              'variance': np.var}

def get_trimmed_mu_estimate(param, shape):

    size=100000

    if shape == 'normal' or shape=='contaminated normal':
        mu=0

    elif shape=='lognormal':
        mu = trim_mean(lognorm.rvs(param, size=size), .2)

    elif shape=='contaminated chi-squared':

        data = chi2.rvs(4, size=size)
        contam_inds=np.random.randint(size, size=int(param*size))
        data[contam_inds] *= 10
        mu=trim_mean(data, .2)

    elif shape=='exponential':
        mu = trim_mean(expon.rvs(0, param, size=size), .2)

    return mu

@st.cache(show_spinner=False)
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

        # x = np.linspace(chi2.ppf(0.01, 4, 0, param), chi2.ppf(0.99, 4, 0, param), 1000)
        # y = chi2.pdf(x, 4, 0, param)
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

    elif shape=='exponential':

        x = np.linspace(expon.ppf(0.01, 0, param), expon.ppf(0.99, 0, param), 1000)
        y = expon.pdf(x, 0, param)
        df = pd.DataFrame({'data': x, 'density': y})

    elif shape=='contaminated normal':

        total_pop_size = 100000
        sub_pop_size = round(param * total_pop_size)
        norm_pop_size = int(total_pop_size - sub_pop_size)
        standard_norm_values = norm.rvs(0, 1, size=norm_pop_size)
        contam_values = norm.rvs(0, 10, size=sub_pop_size)
        values = np.concatenate([standard_norm_values, contam_values])

        x = np.linspace(-3, 3, 1000)
        kernel = gaussian_kde(values)
        y = kernel.pdf(x)

        df = pd.DataFrame({'data': x, 'density': y})


    # elif shape=='argus':
    #
    #     chi=3
    #     x = np.linspace(argus.ppf(0.01, chi, 0, param), argus.ppf(0.99, chi, 0, param), 1000)
    #     y = argus.pdf(x, chi, 0, param)
    #     df = pd.DataFrame({'data': x, 'density': y})


    return df

def generate_random_data_from_dist(param, shape, nrows, ncols):

    if shape=='normal':
       data = norm.rvs(0, param, size=(nrows, ncols))

    # link the two sliders and make the param for t dfs (yolked to sample size in other slider)
    # elif shape=='t':
    #     data = t.rvs(df=ncols-1)

    elif shape=='lognormal':
        data = lognorm.rvs(param, size=(nrows, ncols))

    elif shape=='contaminated chi-squared':

        # data = chi2.rvs(4, 0, param, size=size)
        data = chi2.rvs(4, size=(nrows, ncols))
        contam_inds=np.random.randint(ncols, size=int(param*ncols))
        data[:, contam_inds] *= 10

    elif shape=='contaminated normal':

        sub_size = round(param * ncols)
        norm_size = int(ncols - sub_size)
        standard_norm_values = norm.rvs(0, 1, size=(nrows, norm_size))
        contam_values = norm.rvs(0, 10, size=(nrows, sub_size))
        #print(standard_norm_values.shape)
        #print(contam_values.shape)
        data = np.concatenate([standard_norm_values, contam_values], axis=1)
        #print(data.shape)

    elif shape=='exponential':
       data = expon.rvs(0, param, size=(nrows, ncols))

    return data

def get_population_average_estimate(param, shape):

    size=100000

    if shape=='normal' or shape=='contaminated normal':
        mu=0

    elif shape=='lognormal':
        mu = lognorm.stats(param, moments='m')

    elif shape=='contaminated chi-squared':

        # mu = chi2.stats(4, 0, param, moments='m')
        data = chi2.rvs(4, size=size)
        contam_inds=np.random.randint(size, size=int(param*size))
        data[contam_inds] *= 10
        mu=np.mean(data)

    elif shape=='exponential':
       mu = expon.stats(0, param, moments='m')

    return mu

#@st.cache(show_spinner=False)
def comparison_button_callback():

    sample_size=30
    exclude_est='variance'
    #exclude_dist='t'

    # dists and params are hard coded here since their paired order really matters
    # I tried pulling from global dists but after removing t (which I may not need to do eventually),
    # the pairing with the parmas and dists got messed up
    params_for_sim=[1., 1., 0.1, 0.1, 1.] # add t eventually
    dists_for_sim=['normal', 'lognormal', 'contaminated chi-squared', 'contaminated normal', 'exponential']
    #dists_for_sim=[i for i in dists if i != exclude_dist]
    ests_for_sim={k: est_dict[k] for k in est_dict if k != exclude_est}

    results=[]
    for param, dist in zip(params_for_sim, dists_for_sim):
        print(dist)
        for est_key, est_val in ests_for_sim.items():
            sample = []
            for i in range(1000):
                data = generate_random_data_from_dist(param, dist, 1, sample_size)
                if type(est_val) is dict:
                    func=est_val.get('func')
                    arg=est_val.get('args')
                    est_res=func(data.squeeze(), arg)

                else:
                    func=est_val
                    est_res=func(data.squeeze())

                sample.append(est_res)

            results.append({'dist': dist, 'est': est_key, 'se': np.std(sample, ddof=1)})

    return results

def make_population_chart(df):

    c=alt.Chart(df).mark_line().encode(
        x=alt.X('data', axis=alt.Axis(titleFontSize=15)),
        y=alt.Y('density', axis=alt.Axis(titleFontSize=15))
    )
        
    return c.interactive()

# def sampling_distribution_button_callback(widget_info):
#
#     population_dropdown = population_widget_dict['dropdown']
#     sample_dropdown = sampling_distribution_widgets['dropdown']
#     sample_slider = sampling_distribution_widgets['slider']
#     population_slider = population_widget_dict['slider']
#     output = sampling_distribution_widgets['output']
#     label = sampling_distribution_widgets['label']
#
#     with output:
#         clear_output(wait=True)
#
#         sample=[]
#         for i in range(1000):
#             data = generate_random_data_from_dist(population_slider.value, population_dropdown.value, 1, sample_slider.value)
#             #print(data[0])
#             est=sample_dropdown.value['func'](data) if not sample_dropdown.value.get('args') else \
#                 sample_dropdown.value['func'](np.squeeze(data), sample_dropdown.value['args'])
#                 #trim_mean(data, .2, axis=0)
#
#             #print(est)
#             sample.append(est)
#
#         display(make_sampling_distribution_chart(sample))
#         label.value=f'SE = {np.std(sample, ddof=1).round(2)} based on the {population_dropdown.value} population'

@st.cache(show_spinner=False)
def sampling_distribution_loop(est_param, scale_param, shape_param, samp_param):

    sample=[]
    for i in range(1000):
        data = generate_random_data_from_dist(scale_param, shape_param, 1, samp_param)

        est_func=est_dict[est_param]

        if type(est_func) is dict:
            est=est_func['func'](np.squeeze(data), est_func['args'])

        else:
            est=est_func(data)

        sample.append(est)

    return sample

def make_sampling_distribution_chart(sample):

    #print(type(sample[0]))
    df=pd.DataFrame({'data': sample})
    #print(df.head())

    c=alt.Chart(df).transform_density('data', as_=['data', 'density']).mark_area().encode(
        x=alt.X('data', axis=alt.Axis(titleFontSize=15)),
        y=alt.Y('density:Q', axis=alt.Axis(titleFontSize=15))
    )

    return c.interactive()

def make_comparison_chart(results):

    df = pd.DataFrame(results)

    # bars=alt.Chart(df).mark_bar(tooltip=True, size=30).encode(
    #     y=alt.Y('est', title='Estimator', axis=alt.Axis(titleFontSize=15, labelFontSize=12)),
    #     x=alt.X('se', title='Standard error', axis=alt.Axis(titleFontSize=15, labelFontSize=12)),
    #     color=alt.Color('dist', title='Population shape', legend=alt.Legend(labelFontSize=12, titleFontSize=15))
    # )#.properties(height=300)
    #
    # text = alt.Chart().mark_text(dx=-15, dy=3, color='white').encode(
    #     y=alt.Y('est', sort=['-x']),
    #     x=alt.X('sum(se)', stack='zero'),
    #     detail='dist',
    #     text=alt.Text('sum(se)', format='.2f')
    # )

    bars=alt.Chart().mark_rect(tooltip=True).encode(
        y=alt.Y('est', title='Estimator', axis=alt.Axis(titleFontSize=18, labelFontSize=15)),
        x=alt.X('dist', title='Population shape', axis=alt.Axis(titleFontSize=18, labelFontSize=15)),
        color=alt.Color('se', title='Standard error')
    )

    text = alt.Chart().mark_text(tooltip=True, color='black', size=15).encode(
        y=alt.Y('est', title='Estimator',),
        x=alt.X('dist', title='Population shape'),
        text=alt.Text('se', format='.3f', title='Standard error')
    )

    #properties(height=220, width=400)
    return alt.layer(bars, text, data=df).properties(height=500, width=600).configure_scale(bandPaddingInner=0)

@st.cache(show_spinner=False)
def t_sampling_distribution_loop(scale_param, shape_param, samp_param):

    mu = get_population_average_estimate(scale_param, shape_param)

    sample=[]
    for i in range(1000):
        data = generate_random_data_from_dist(scale_param,
                    shape_param, 1, samp_param)

        est=(np.sqrt(samp_param)*(np.mean(data)-mu))/np.std(data, ddof=1)
        sample.append(est)

    return sample

def make_sampling_distribution_of_t_chart(sample, samp_param):

    freedom=samp_param-1
    df_assumed = make_pdf(freedom, 't')
    df_actual=pd.DataFrame({'actual': sample})

    actual=alt.Chart(df_actual).transform_density('actual',
            as_=['actual', 'density']).mark_line().encode(
        x=alt.X('actual', title='T',
                axis=alt.Axis(titleFontSize=15)),
        y=alt.Y('density:Q', axis=alt.Axis(titleFontSize=15)),
    )

    assumed=alt.Chart(df_assumed).mark_line(color='lightgrey').encode(
        x=alt.X('data', title='T',
                axis=alt.Axis(titleFontSize=15)),
        y=alt.Y('density', axis=alt.Axis(titleFontSize=15)),
    )

    c=alt.layer(assumed, actual)

    return c.interactive()

def type_I_error_button_callback(g, h):

    samp_size=12
    #g=.8
    #h=0
    sample_data=sample_from_g_and_h_distribution(g,h)
    t_error_low, t_error_up=simulate_t_type_I_error(sample_data, samp_size, g, h)
    #t_error_low + t_error_up
    pb_error_low, pb_error_up=simulate_pb_type_I_error(sample_data, samp_size, g, h)
    #print(pb_error_low + pb_error_up)

    results=[{'test': 't-test', 'error': t_error_low, 'direction': 'P(test_stat < .025 quantile)'},
             {'test': 't-test', 'error': t_error_up, 'direction':  'P(test_stat > .975 quantile)'},
             {'test': 'percentile boot', 'error': pb_error_low, 'direction':  'P(test_stat < .025 quantile)'},
             {'test': 'percentile boot', 'error': pb_error_up, 'direction':  'P(test_stat > .975 quantile)'},
             ]

    return results

def sample_from_g_and_h_distribution(g,h):

    # g=0
    # h=0
    #Zs=generate_random_data_from_dist(1, 'normal', 100000, samp_size) #nsamples x samp_size
    Zs=generate_random_data_from_dist(1, 'normal', 100000, 1) #nsamples x samp_size

    if g>0:
        Xs=((np.exp(g*Zs)-1)/g) * np.exp(h*(Zs**2)/2)

    else:
        Xs=Zs*np.exp(h*(Zs**2)/2)

    return Xs.squeeze()

def simulate_t_type_I_error(data, samp_size, g, h): #param, dist, samp_size

    nboot=599
    samples=np.random.choice(data.squeeze(), size=(nboot, samp_size))
    mu=ghmean(g,h)
    tvals = (np.sqrt(samp_size) * (np.mean(samples, axis=1) - mu)) / np.std(samples, ddof=1, axis=1)

    t_crit = t.ppf(.975, samp_size - 1)
    prob_up = (np.sum(tvals >= t_crit)) / len(tvals)
    prob_low = (np.sum(tvals <= -t_crit)) / len(tvals)

    return prob_low, prob_up

def ghmean(g,h):

    if h==0 and g>0:

        val=(np.exp(g**2/2)-1) / g
        #val2 = (1 - 2 * np.exp(g ** 2 / 2) + np.exp(2 * g ** 2)) / g ** 2
        #val2 = val2 - val ** 2

    elif h != 0 and g>0:
        #val2=np.nan
        if h<1:
            val=(np.exp(g ** 2 / (2 * (1 - h))) - 1) / (g * np.sqrt(1 - h))

        # elif 0 < h < .5:
        #     val2 = (np.exp(2 * g ** 2 / (1 - 2 * h)) - 2 * np.exp(g ** 2 / (2 * (1 - 2 * h))) +
        #           1) / (g ** 2 * np.sqrt(1 - 2 * h)) - (np.exp(g ** 2 / (2 * (1 - h))) - 1) ** 2 / (g ** 2 * (1 - h))

    elif g==0:
        val=0
        #val2 = 1 / (1 - 2 * h) ** 1.5   #Headrick et al. (2008)

    return val#, val2

# @numba.jit(nopython=True)
# def vendored_trim_mean(a, proportiontocut, axis=0):
#
#     a = np.asarray(a)
#
#     if a.size == 0:
#         return np.nan
#
#     # if axis is None:
#     #     a = a.ravel()
#     #     axis = 0
#
#     nobs = a.shape[axis]
#     lowercut = int(proportiontocut * nobs)
#     uppercut = nobs - lowercut
#
#     if (lowercut > uppercut):
#         raise ValueError("Proportion too big.")
#
#     atmp = np.partition(a, (lowercut, uppercut - 1), axis)
#
#     sl = [slice(None)] * atmp.ndim
#     sl[axis] = slice(lowercut, uppercut)
#     return np.mean(atmp[tuple(sl)], axis=axis)

# @numba.jit(nopython=True)
# def percentile_bootstrap_tests(data, nboot, mu, samp_size):
#
#     l = round(.05 * nboot / 2) - 1
#     u = nboot - l - 2
#
#     bools=[]
#     for sample in data:
#         bdat = np.random.choice(sample, size=(nboot, samp_size))
#         #effects=trim_mean(bdat, .2, axis=1) - mu
#
#         effects=[]
#         for row in bdat:
#             effects.append(np.mean(row) - mu)
#
#         #effects = np.mean(bdat, 1) - mu
#         up = sorted(effects)[u]
#         low = sorted(effects)[l]
#         # up = np.sort(effects)[u]
#         # low = np.sort(effects)[l]
#         bools.append((low < 0 < up))
#
#     arr_bools=np.array(bools)
#     #prob = 1 - (np.sum(bools) / len(bools))
#     prob = 1 - (np.sum(arr_bools) / len(arr_bools))
#
#     return prob

def ghtrim(g,h):

    tr=.2

    if g==0:
        val=0
    elif g>0:
        low=norm.ppf(tr)
        up = -1 * low
        val = quad(ftrim, low, up, args=(g,h))[0]
        val = val / (1-2*tr)

    return val

def ftrim(z,g,h):
    gz = (np.exp(g * z) - 1) * np.exp(h * z ** 2 / 2) / g
    res= norm.pdf(z) * gz

    return res

def simulate_pb_type_I_error(data, samp_size, g, h): #param, dist, samp_size

    nboot = 1000
    nsims= 599
    l = round(.05 * nboot / 2) - 1
    u = nboot - l - 2
    mu = ghtrim(g, h)

    sig_ups=[]
    sig_lows=[]
    for s in range(nsims):
        experiment_data = np.random.choice(data, size=samp_size)
        bdat = np.random.choice(experiment_data, size=(nboot, samp_size))
        effects = trim_mean(bdat, .2, axis=1) - mu
        up = np.sort(effects)[u]
        low = np.sort(effects)[l]

        if low >= 0:
            sig_lows.append(1)

        elif up <=0:
            sig_ups.append(1)

        # if (low>0 and up>0) or (low<0 and up<0):
        #     print('found sig')

    prob_low = (np.sum(sig_lows) / nsims)
    prob_up = (np.sum(sig_ups) / nsims)

    return prob_low, prob_up

# def simulate_tt_type_I_error(param, dist, samp_size):
#
#     mu = get_trimmed_mu_estimate(param, dist)
#     nboot = 599
#     l=round(.025 * nboot)
#     u=round(.975 * nboot)
#
#     bools=[]
#     for i in range(100):
#         data = generate_random_data_from_dist(param, dist, 1, samp_size)
#
#         bdat = np.random.choice(data[0], size=(nboot, samp_size))
#         t_stat = (trim_mean(bdat, .2, axis=1) - mu) / (winvar(bdat, axis=1)) / (0.6 * np.sqrt(samp_size))
#         sorted_t_stat=np.sort(t_stat)
#         Tlow = sorted_t_stat[l]
#         Tup = sorted_t_stat[u]
#         CI_low=mu - (Tup * (winvar(data[0])) / (0.6 * np.sqrt(samp_size)))
#         CI_up= mu - (Tlow * (winvar(data[0])) / (0.6 * np.sqrt(samp_size)))
#
#         bools.append((CI_low < mu < CI_up))
#
#     prob = 1 - (np.sum(bools) / len(bools))
#
#     return prob

# @numba.jit(nopython=True)
# def my_trimmed_mean(data, percentile):
#
#     for i in range(data.shape[0]):
#         data[i].sort()
#
#     low = int(percentile * data.shape[1])
#     high = int((1. - percentile) * data.shape[1])
#
#     results=np.zeros(data.shape[0])
#     for i in range(data.shape[0]):
#         results[i]=np.mean(data[i, low:high])
#
#     return results

def winvar(x, tr=.2, axis=0):

    y=winsorize(x, limits=(tr,tr), axis=axis)
    wv = np.var(y, ddof=1, axis=axis)

    return wv

def make_type_I_error_chart(results):

    df = pd.DataFrame(results)

    bars=alt.Chart().mark_bar(size=30).encode(
        y=alt.Y('test:N', title='Type of test', axis=alt.Axis(titleFontSize=18, labelFontSize=15)),
        x=alt.X('sum(error):Q', title='Probability of Type I error', axis=alt.Axis(titleFontSize=18, labelFontSize=15), stack='zero'),
        color=alt.Color('direction:N', legend=alt.Legend(title=None, labelFontSize=18, labelLimit=1000)),
        order = alt.Order('direction:N'),
        tooltip = alt.Tooltip(['test', 'direction', 'error'])
    )

    text = alt.Chart().mark_text(color='black', size=15, dx=-20).encode(
        y=alt.Y('test:N', title='Type of test',),
        x=alt.X('error:Q', title='Probability of Type I error', stack='zero'),
        text=alt.Text('error:Q', format='.3f'),
        order=alt.Order('direction:N'),
        tooltip=alt.Tooltip(['test', 'direction', 'error'])
    )

    return alt.layer(bars,text, data=df).properties(height=300, width=600)
