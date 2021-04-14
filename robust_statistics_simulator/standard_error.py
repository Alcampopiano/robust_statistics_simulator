import numpy as np
import streamlit as st
from funcs.simulation import make_pdf, make_population_chart, sampling_distribution_loop, \
    make_sampling_distribution_chart, make_comparison_chart, dists, est_dict, comparison_button_callback

#######################################

st.title('Population Characteristics')
st.markdown('<br>', unsafe_allow_html=True)
shape_param=st.selectbox('Choose population shape', dists)
st.markdown('<br>', unsafe_allow_html=True)

if shape_param in ['contaminated normal', 'contaminated chi-squared']:
    scale_param = st.slider('Choose contamination', .01, 1.0, .1, step=.01)
else:
    scale_param = st.slider('Choose standard deviation', 1.0, 5.0, 1.0, step=.01)

df=make_pdf(scale_param, shape_param)
c=make_population_chart(df)
st.write(c)
st.title('Estimating Standard Error')
est_param=st.selectbox('Estimator', list(est_dict.keys()))
samp_param = st.slider('Choose sample size', 1, 100, 20, step=1)
sample=sampling_distribution_loop(est_param, scale_param, shape_param, samp_param)
c2=make_sampling_distribution_chart(sample)
st.write(c2)
st.write(f'SE = {np.std(sample, ddof=1).round(2)} based on the {shape_param} population')

st.title('SE Estimates by Shape and Estimator')
st.markdown('<br>', unsafe_allow_html=True)

if st.button('Run simulations'):
    results=comparison_button_callback()
    c3=make_comparison_chart(results)
    st.write(c3)









